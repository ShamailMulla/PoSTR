"""
Neural-linear Bayesian head for BTRL v3 — replaces BayesFormer dropout as the
posterior over world models.

The transformer encoder provides features Φ = transition_network.features(...).
A Bayesian linear regression maps [Φ, 1] -> [next-state logits (obs_space), reward]
with a Gaussian posterior over the weights. Thompson sampling draws one weight
matrix W per episode (sample()); acting uses Ŷ = [Φ,1] · W.

Unlike dropout, the posterior precision is built from the data (Λ = σ⁻²XᵀX + λI),
so uncertainty is HIGH where data is scarce and shrinks as data accumulates —
the self-widening property that lets exploration stay directed and forgetting
self-correct (the defect diagnosed in v1/v2). Ported/adapted from
PSDRL/bayes/neural_linear_model.py.
"""
import numpy as np
import torch

from ..common.utils import extract_episode_data


class NeuralLinearHead:
    def __init__(self, config, transition_network, terminal_network,
                 obs_space, actions, feat_dim, device):
        self.tn = transition_network
        self.en = terminal_network
        self.device = device
        self.actions = torch.tensor(actions, device=device)
        self.num_actions = len(actions)
        self.obs_space = obs_space

        self.noise_variance = float(config.get("noise_variance", 1.0))     # β, likelihood precision
        self.blr_coeff = float(config.get("blr_coefficient", 100.0))       # α, prior precision (the real Bayesian prior)
        # Sampling temperatures: 1.0 => draw from the TRUE posterior N(mu, cov)
        # (correct Thompson sampling). Values != 1 temper the per-block spread —
        # an optional HPO knob, NOT the default (PSDRL used <1 here, which under-
        # explores the transition). Kept separate from blr_coeff, which is the
        # actual prior precision entering the posterior.
        self.transition_temp = float(config.get("transition_temp", 1.0))   # state block sampling spread
        self.reward_temp = float(config.get("reward_temp", 1.0))           # reward sampling spread
        # Decoupled exploration: acting draws a WIDER state sample than value
        # training. This keeps value targets stable (true-posterior temp=1,
        # exactly as PSDRL trains its value net) while widening only the
        # exploration sample so directed deep exploration stays alive where the
        # posterior would otherwise collapse. explore_temp defaults to
        # transition_temp (no decoupling) for backward compatibility.
        self.explore_temp = float(config.get("explore_temp", self.transition_temp))
        # v9 fix: query the terminal net on one-hot-projected predicted states
        self.onehot_terminal = bool(config.get("onehot_terminal_inputs", False))

        self.feat_dim = feat_dim
        self.out_dim = obs_space + 1               # state logits + reward
        self.D = feat_dim + 1                      # + bias
        self.eye = torch.eye(self.D, device=device)

        self.mu = torch.zeros(self.D, self.out_dim, device=device)
        self.cov = self.eye.clone()
        self.chol_cov = self.eye.clone()           # lower-triangular chol(cov) for sampling
        self.w = torch.zeros(self.D, self.out_dim, device=device)

        # per-output sampling-spread scale (1.0 = exact posterior sampling).
        # col_scale = value-training / true-posterior spread; explore_col_scale =
        # wider spread used only for acting. Reward block is not widened for
        # exploration (the transition posterior is what drives deep exploration).
        self.col_scale = torch.full((self.out_dim,), self.transition_temp ** 0.5, device=device)
        self.col_scale[obs_space] = self.reward_temp ** 0.5
        self.explore_col_scale = torch.full((self.out_dim,), self.explore_temp ** 0.5, device=device)
        self.explore_col_scale[obs_space] = self.reward_temp ** 0.5

        self.sample()  # draw from the prior initially

    # ---- feature construction -------------------------------------------------
    def _bias(self, phi):
        return torch.cat([phi, torch.ones(phi.shape[0], 1, device=self.device)], dim=1)

    def _features(self, timestep, observations, actions):
        return self._bias(self.tn.features(timestep, observations, actions))  # (N, D)

    def compute_feature_maps(self, dataset, chunk: int = 512):
        """
        Assemble (X, Y) over every transition in the replay buffer. The features
        are computed in CHUNKS so peak GPU memory is bounded by `chunk`
        transitions, not the whole buffer (which OOMs at ~1e4 transitions with
        several runs sharing a GPU).
        """
        pool = dataset.episodes + getattr(dataset, "goal_episodes", [])
        obs_list, ts_list, act_list, y_list = [], [], [], []
        for ep in pool:
            o, a, o1, r, done, ts = extract_episode_data([ep])
            T = o.shape[1]
            obs_list.append(o[0])                       # (T, ctx, obs)
            ts_list.append(ts[0])                       # (T, ctx)
            act_list.append(a[0].reshape(T, 1))         # (T, 1)
            y_list.append(torch.cat([o1[0], r[0].reshape(T, 1)], dim=1))  # (T, obs+1)
        obs = torch.cat(obs_list); ts = torch.cat(ts_list)
        act = torch.cat(act_list); Y = torch.cat(y_list).to(self.device)
        feats = []
        for i in range(0, obs.shape[0], chunk):
            feats.append(self._features(
                ts[i:i + chunk].to(self.device),
                obs[i:i + chunk].to(self.device),
                act[i:i + chunk].to(self.device),
            ))
        X = torch.cat(feats, dim=0)                    # (M, D)
        return X, Y

    # ---- closed-form Bayesian linear regression -------------------------------
    def update_posteriors(self, dataset):
        X, Y = self.compute_feature_maps(dataset)
        Lam = self.noise_variance * (X.T @ X)          # data precision (D,D)
        coeff = self.blr_coeff
        while True:
            A = Lam.double() + self.eye.double() * coeff
            try:
                chol = torch.linalg.cholesky(A)
                cov = torch.cholesky_inverse(chol).float()
            except Exception:
                coeff *= 10; continue
            if cov.isnan().any():
                coeff *= 10; continue
            self.cov = cov
            break
        self.mu = self.noise_variance * (self.cov @ (X.T @ Y))   # (D, out_dim)
        # symmetrise + jitter for a stable sampling factor
        cov_sym = 0.5 * (self.cov + self.cov.T) + self.eye * 1e-6
        self.chol_cov = torch.linalg.cholesky(cov_sym.double()).float()
        self.sample()

    def sample(self, explore: bool = False):
        """
        Draw one weight matrix from the posterior (Thompson sampling).
        explore=True widens the state block (acting/exploration); explore=False
        uses the true-posterior spread (value training — PSDRL-faithful).
        """
        z = torch.randn(self.D, self.out_dim, device=self.device)
        scale = self.explore_col_scale if explore else self.col_scale
        self.w = self.mu + (self.chol_cov @ z) * scale   # per-block spread

    # ---- prediction (matches BayesianTransformer.predict's needs) -------------
    def predict(self, timestep, observations, actions):
        X = self._features(timestep, observations, actions)
        pred = X @ self.w                              # (N, out_dim)
        states = pred[:, : self.obs_space]
        rewards = pred[:, self.obs_space: self.obs_space + 1]
        if self.onehot_terminal:
            # v9 fix: terminal net trains on one-hot buffer states — query it
            # on the one-hot projection of the predicted logits, not raw logits.
            from ..common.utils import project_onehot
            terminals = self.en.predict(project_onehot(states))
        else:
            terminals = self.en.predict(states)
        return states, rewards, terminals
