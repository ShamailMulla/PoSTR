import sys

import torch
import numpy as np

from ..common.logger import Logger
from ..common.replay import Dataset
from ..common.utils import preprocess_image, compute_state_loss, project_onehot
from ..networks.terminal import Network as TerminalNetwork
from ..networks.transition import Network as TransitionNetwork
from ..networks.value import Network as ValueNetwork
from ..bayes.transformer_rl_agent import BayesianTransformer
from ..common import tracing
from ..bayes.neural_linear_head import NeuralLinearHead
from ..training.transition import TransitionModelTrainer
from ..training.policy_btrl import PolicyTrainer
from ..common.settings import TP_THRESHOLD


class BTRL:
    def __init__(self, config: dict, actions: list, logger: Logger, env_dim: int, seed: int = None):
        if config["gpu"] and torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            if config["gpu"]:
                print("WARNING: gpu=true in config but CUDA unavailable — falling back to CPU.")
            self.device = "cpu"
        self.random_state = np.random.RandomState(seed)
        print("Available device:", self.device)
        self.num_actions = len(actions)
        self.actions = torch.tensor(actions, dtype=int, device=self.device)

        # v9 fix: query the value net on one-hot-projected predicted states so
        # acting-time inputs match the (one-hot) training distribution.
        # Only valid for one-hot observation spaces (grid); off by default.
        self.value_onehot = config["value"].get("onehot_inputs", False)
        self.epsilon = config["algorithm"]["policy_noise"]
        self.update_freq = config["algorithm"]["update_freq"]
        self.warmup_length = config["algorithm"]["warmup_length"]
        self.warmup_freq = config["algorithm"]["warmup_freq"]
        self.discount = config["value"]["discount"]

        self.dataset = Dataset(
            logger, config["replay"], config["experiment"]["time_limit"], config["transition"]["context_length"], self.device, env_dim, seed
        )

        terminal_network = TerminalNetwork(
            env_dim, config["terminal"], self.device
        )
        transition_network = TransitionNetwork(
            config["transition"],
            env_dim,
            actions,
            p_seq=config["transition"].get("p_seq", 0.2),
            p_attn=config["transition"].get("p_attn", 0.05),
            max_steps=config["experiment"]["time_limit"],
            device=self.device
        )
        self.value_network = ValueNetwork(
            env_dim,
            config["value"],
            config["transition"]["context_length"],
            self.device
        )

        # v3: neural-linear head owns the posterior over world models when enabled.
        self.neural_linear = config["algorithm"].get("neural_linear", False)
        head = None
        if self.neural_linear:
            head = NeuralLinearHead(
                config["algorithm"],
                transition_network,
                terminal_network,
                env_dim,
                actions,
                feat_dim=config["transition"]["hidden_dim"],  # penultimate feature dim
                device=self.device,
            )

        self.model = BayesianTransformer(
            config["algorithm"],
            states=env_dim,
            actions=actions,
            context_length=config['transition']['context_length'],
            transition_network=transition_network,
            terminal_network=terminal_network,
            device=self.device,
            neural_linear_head=head,
        )

        self.transition_trainer = TransitionModelTrainer(
            config["transition"],
            transition_network,
            terminal_network,
            config["replay"]["batch_size"],
            self.num_actions,
            device=self.device
        )
        self.policy_trainer = PolicyTrainer(
            config["value"],
            config["transition"]["context_length"],
            self.value_network,
            self.device,
            config["replay"]["batch_size"],
            self.actions,
        )

        # v6: frozen-phi. Once the transition model's held-out loss plateaus AND
        # the buffer contains at least one reward-bearing episode, stop all
        # gradient training of the transition/terminal networks. The feature map
        # phi becomes a fixed dictionary; all further model learning happens in
        # the BLR head (exact, refit on all data, cannot forget within H_phi).
        # The freeze criterion is data-observable (early-stopping semantics),
        # not an environment-tuned step count.
        self.freeze_phi_enabled = config["algorithm"].get("freeze_phi", False)
        # v8: freeze_mode "hard" = v6 (stop all transition training at trigger);
        # "lora" = freeze the base but enable low-rank adapters, so gradient
        # training continues within a rank-r subspace per layer.
        self.freeze_mode = config["algorithm"].get("freeze_mode", "hard")
        self.freeze_patience = int(config["algorithm"].get("freeze_patience", 20))
        self.freeze_tol = float(config["algorithm"].get("freeze_min_rel_improvement", 0.02))
        self.phi_frozen = False
        self.freeze_step = None
        self._best_val_loss = None
        self._val_stall = 0
        self._probe = self._build_probes(env_dim, config["transition"]["context_length"])
        self._prev_probe_phi = None

    def _build_probes(self, env_dim: int, ctx: int):
        """
        Fixed probe set for the phi-drift and posterior-alive metrics: the
        diagonal (optimal-path) states with their canonical previous-diagonal
        context, both actions each. Only valid for square grid observations
        (DeepSea); metrics are disabled otherwise.
        """
        n = int(round(env_dim ** 0.5))
        if n * n != env_dim:
            return None
        obs, ts, act = [], [], []
        for r in range(n):
            for a in (0, 1):
                o = torch.zeros(ctx, env_dim, device=self.device)
                o[-1, r * n + r] = 1
                if ctx >= 2 and r > 0:
                    o[-2, (r - 1) * n + (r - 1)] = 1
                obs.append(o)
                ts.append(torch.tensor([max(r - 1, 0)] * (ctx - 1) + [r], device=self.device))
                act.append(torch.tensor([a], device=self.device))
        return (torch.stack(ts), torch.stack(obs), torch.stack(act))

    def _validation_loss(self):
        """Single no-grad pass over a freshly sampled batch: mean next-state
        loss + reward loss. Used only as a plateau signal for the freeze."""
        o, a, o1, r, done, timesteps = self.dataset.sample_sequences()
        total, count = 0.0, 0
        with torch.no_grad():
            for idx in range(len(o[0])):
                observation = o[:, idx].unsqueeze(1).reshape(
                    self.dataset.batch_size, -1, self.model.transition_network.states
                )
                next_state_preds, return_preds, _ = self.model.transition_network.forward(
                    timesteps[:, idx], observation, a[:, idx]
                )
                s_loss = compute_state_loss(
                    next_state_preds, o1[:, idx],
                    obs_type=self.transition_trainer.obs_type,
                    obs_space=self.transition_trainer.obs_space,
                )
                r_loss = torch.nn.functional.mse_loss(return_preds, r[:, idx].view_as(return_preds))
                total += (s_loss + r_loss).item()
                count += 1
        return total / max(count, 1)

    @tracing.traced("BTRL.check_freeze")
    def _check_freeze(self, timestep: int):
        val_loss = self._validation_loss()
        self.dataset.logger.add_scalars("Phi/Val_Loss", val_loss)
        if self._best_val_loss is None or (self._best_val_loss - val_loss) > self.freeze_tol * self._best_val_loss:
            self._best_val_loss = val_loss
            self._val_stall = 0
        else:
            self._val_stall += 1
        goal_seen = any(
            bool((ep["rewards"] > 0.5).any())
            for ep in self.dataset.episodes + self.dataset.goal_episodes
        )
        if self._val_stall >= self.freeze_patience and goal_seen:
            self.phi_frozen = True
            self.freeze_step = timestep
            if self.freeze_mode == "lora":
                # base frozen, adapters enabled; terminal net keeps training
                # (its inputs are observation-space, stationary by nature).
                for p in self.model.transition_network.parameters():
                    p.requires_grad_(False)
                for m in self.model.transition_network.modules():
                    if hasattr(m, "lora_A"):
                        m.lora_A.requires_grad_(True)
                        m.lora_B.requires_grad_(True)
                print(f"*** PHI BASE FROZEN + LoRA ADAPTERS ENABLED at timestep {timestep} "
                      f"(val loss {val_loss:.5f}, best {self._best_val_loss:.5f}) ***")
            else:
                for net in (self.model.transition_network, self.model.terminal_network):
                    for p in net.parameters():
                        p.requires_grad_(False)
                print(f"*** PHI FROZEN at timestep {timestep} "
                      f"(val loss {val_loss:.5f}, best {self._best_val_loss:.5f}, "
                      f"stalled {self._val_stall} update cycles) ***")

    @tracing.traced("BTRL.posterior_probe")
    def _log_phi_metrics(self):
        """Log feature drift on the fixed probe set and the posterior-alive
        readout (distinct sampled next-states / modal fraction), every update."""
        if self._probe is None or not self.neural_linear:
            return
        ts, obs, act = self._probe
        head = self.model.head
        with torch.no_grad():
            X = head._features(ts, obs, act)                     # (2N, D) incl. bias
            phi = X[:, :-1]
            if self._prev_probe_phi is not None:
                drift = 1 - torch.nn.functional.cosine_similarity(
                    phi, self._prev_probe_phi, dim=1
                ).mean().item()
                self.dataset.logger.add_scalars("Phi/Feature_Drift", drift)
            self._prev_probe_phi = phi.clone()

            w_backup = head.w.clone()
            K = 20
            argmaxes = []
            for _ in range(K):
                head.sample()
                pred = X @ head.w
                argmaxes.append(pred[:, : head.obs_space].argmax(-1))
            head.w = w_backup
            am = torch.stack(argmaxes)                           # (K, 2N)
            distinct = np.mean([len(torch.unique(am[:, j])) for j in range(am.shape[1])])
            modal = np.mean([
                torch.bincount(am[:, j]).max().item() / K for j in range(am.shape[1])
            ])
        self.dataset.logger.add_scalars(
            ["Phi/Posterior_Distinct", "Phi/Posterior_Modal", "Phi/Frozen"],
            [float(distinct), float(modal), float(self.phi_frozen)],
        )
        norms = [m.adapter_norm() for m in self.model.transition_network.modules()
                 if hasattr(m, "adapter_norm")]
        if norms:
            self.dataset.logger.add_scalars("Phi/Adapter_Norm", float(np.mean(norms)))

    def _value_probe(self):
        """V over every one-hot state: std ~0 means the value net is a constant (report, 2026-07-19)."""
        try:
            layers = self.value_network.layers
            n = next(m for m in layers.modules() if isinstance(m, torch.nn.Linear)).in_features
            dev = next(layers.parameters()).device
            with torch.no_grad():
                v = layers(torch.eye(n, device=dev)).flatten()
            return {"value_onehot_mean": v.mean().item(), "value_onehot_std": v.std().item()}
        except Exception as e:  # probe must never break training
            return {"value_probe_error": repr(e)}

    def select_action(self, current_state: np.array, step: int):
        """
        Reset the hidden state at the start of a new episode. Return a random action with a probability of epsilon,
        otherwise follow the current policy and sampled model greedily.
        """
        if step == 1:
            self.model.sample_model()   # lock M̂ — one world hypothesis per episode

        # print('select_action\nCurrent timestep:', step)
        if self.random_state.random() < self.epsilon:
            return self.random_state.choice(self.num_actions)

        pred_next_states, rewards, terminals = self.model.predict(step, current_state)
        pred_rewards = rewards.squeeze()
        # 1 new state for each action
        # print('\nNew observation:',pred_next_states.shape)

        # acting greedily with respect to the value network
        v_in = project_onehot(pred_next_states) if self.value_onehot else pred_next_states
        value_preds = self.value_network.predict(v_in).squeeze()
        v = self.discount * (
                value_preds * (terminals.squeeze() < TP_THRESHOLD)
        )
        values = (pred_rewards + v).detach().cpu().numpy()
        # print('values',values)
        action = self.random_state.choice(np.where(np.isclose(values, max(values)))[0])
        # print('action:',action)

        return self.actions[action], self.model.trajectory, self.model.time_transition

    def update(self,
        current_obs: np.array, action: torch.IntTensor, rew: int, obs: np.array,
        done: bool, ep: int, timestep: int, episode_step: int
    ):
        """
        Add new transition to replay buffer and if it is time to update:
         - Update transition model (Equation 2) and terminal models (Equation 3).
         - Update posterior distributions model (Equation 4).
         - Sample new model from posteriors.
         - Update value network based on the new sampled model (Equation 5).
        """
        # current_obs, obs = preprocess_image(current_obs)[0], preprocess_image(obs)[0]
        self.dataset.add_data(current_obs, action, obs, rew, done, episode_step)
        update_freq = (
            self.update_freq if timestep > self.warmup_length else self.warmup_freq
        )
        if ep and timestep % update_freq == 0:
            print('\n\n','*'*30,'Beginning training','*'*30,'\n')
            with tracing.cycle(timestep, phi_frozen=self.phi_frozen) as tc:
                # print('Training dataset\n',self.dataset.tmp_episode)
                # v6: once phi is frozen, the gradient stage is skipped entirely —
                # the BLR refit below carries all further model learning.
                # v8 (freeze_mode "lora"): the gradient stage keeps running after
                # the freeze; only the adapters receive gradients.
                if not self.phi_frozen or self.freeze_mode == "lora":
                    self.transition_trainer.train_(self.dataset)
                    if not self.phi_frozen and self.freeze_phi_enabled:
                        self._check_freeze(timestep)
                # v3: refit the neural-linear posterior on the freshly-trained features
                # (and draw a sample) BEFORE the value network trains under it.
                if self.neural_linear:
                    self.model.head.update_posteriors(self.dataset)
                    self._log_phi_metrics()
                self.policy_trainer.train_(self.model, self.dataset)
                if tc.on:
                    tc.outputs = {k: v for k, v in self.dataset.logger.log["scalars"].items()
                                  if k.startswith(("Loss/", "Phi/")) and isinstance(v, (int, float))}
                    tc.outputs.update(self._value_probe())
            print('\n\n', '*' * 30, 'Finished training', '*' * 30, '\n')