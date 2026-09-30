"""
Stage 1 for the current PoSTR setup — offline search over the Transformer transition
network *and* the neural-linear (BLR) posterior on top of it.

Differences from transition_search.py (the July 2026 search):
  * posterior: world hypotheses are weight matrices W sampled from the BLR posterior
    fitted on the trained network's features (as NeuralLinearHead does in the agent),
    not locked dropout masks — dropout is off in the current agent (p_seq = p_attn = 0);
  * search space: dropout rates removed; BLR prior precision `blr_coefficient` (alpha),
    likelihood precision `noise_variance` (beta) and `context_length` (1 = Markov model,
    as the planner needs; 2 = current config) added;
  * data: randomized DeepSea (see notebook_utils.make_env), rewards squashed with tanh
    as the agent's replay buffer does;
  * test pairs whose next state is terminal (last row) are left out of the grid metrics
    (the all-zero terminal observation has no grid position).

Objectives are unchanged (both minimised, NSGA-II):
  1. mean_grid_dist — mean (row, col) distance between each sampled hypothesis's
     predicted next state and the true one, over all test (state, action) pairs;
  2. overconfident_error_rate — fraction of pairs that are wrong (dist > DIST_TOL)
     while the posterior is near-certain (grid variance < VAR_EPS).
Reward-model diagnostics (R^2, goal-transition prediction and margin) are returned as
extra metrics, since the reward model decides whether a planner can find the goal.
"""
import sys

import numpy as np
import pandas as pd
import torch

from .common import SRC_DIR

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from notebook_utils import early_stopping_train  # noqa: E402
from ..networks.transition import Network as TransitionNetwork  # noqa: E402
from .transition_search import VALID_PAIRS, DIST_TOL, VAR_EPS  # noqa: E402


def suggest_params(trial) -> dict:
    pair_idx = trial.suggest_int("hd_heads_idx", 0, len(VALID_PAIRS) - 1)
    hidden_dim, heads = VALID_PAIRS[pair_idx]
    return {
        "hidden_dim": hidden_dim,
        "heads": heads,
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "num_encoder_layers": trial.suggest_int("num_encoder_layers", 1, 4),
        "forward_expansion": trial.suggest_categorical("forward_expansion", [2, 4, 8]),
        "context_length": trial.suggest_categorical("context_length", [1, 2]),
        "blr_coefficient": trial.suggest_float("blr_coefficient", 1.0, 1000.0, log=True),
        "noise_variance": trial.suggest_float("noise_variance", 0.1, 10.0, log=True),
    }


def _features(model, ts, obs, act, chunk=1024):
    with torch.no_grad():
        f = torch.cat([model.features(ts[i:i + chunk], obs[i:i + chunk], act[i:i + chunk])
                       for i in range(0, len(obs), chunk)])
    return torch.cat([f, torch.ones(len(f), 1, device=f.device)], dim=1)   # + bias


def fit_blr(X, Y, alpha, beta):
    """Posterior over W (D x out) for Y = X W + noise, prior N(0, 1/alpha), noise precision beta."""
    D = X.shape[1]
    A = beta * (X.T @ X).double() + alpha * torch.eye(D, device=X.device, dtype=torch.float64)
    L = torch.linalg.cholesky(A)
    cov = torch.cholesky_inverse(L)
    mu = beta * (cov @ (X.T.double() @ Y.double()))
    cov = 0.5 * (cov + cov.T) + 1e-9 * torch.eye(D, device=X.device, dtype=torch.float64)
    return mu.float(), torch.linalg.cholesky(cov).float()


def train_and_eval(config, params, datasets, device, max_epochs=300, patience=15,
                   min_delta=5e-4, n_hypotheses=100, seed=0):
    """Train one transition network on the dataset for its context length, fit the BLR
    posterior on its features and evaluate n_hypotheses sampled world models."""
    torch.manual_seed(seed)
    data = datasets[params["context_length"]]
    cfg = dict(config["transition"])
    for k in ("hidden_dim", "heads", "learning_rate", "num_encoder_layers", "forward_expansion",
              "context_length"):
        cfg[k] = params[k]
    obs_space, actions = data["obs_space"], data["possible_actions"]
    N = int(round(obs_space ** 0.5))
    max_steps = int(torch.max(data["timesteps"]).item()) + 1
    model = TransitionNetwork(cfg, obs_space, actions, p_seq=0.0, p_attn=0.0,
                              max_steps=max_steps, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    _, epochs_to_converge, best_loss = early_stopping_train(
        model, data["states"], data["next_states"], data["timesteps"], data["actions"],
        data["rewards"], optimizer, obs_type=config["transition"]["obs_type"],
        obs_space=obs_space, max_epochs=max_epochs, patience=patience, min_delta=min_delta,
        device=device)
    model.eval()

    # --- BLR posterior on the training transitions --------------------------------
    S, NS = data["states"].to(device), data["next_states"].to(device).reshape(len(data["states"]), -1)
    T, A, R = data["timesteps"].to(device), data["actions"].to(device), data["rewards"].to(device)
    X = _features(model, T, S, A)
    Y = torch.cat([NS, R.reshape(-1, 1)], dim=1)
    mu, chol = fit_blr(X, Y, params["blr_coefficient"], params["noise_variance"])

    # --- sampled world models on the per-state test set ---------------------------
    tt = data["test_transitions"]
    states = [s for s in tt if s // N < N - 1]          # next state is a grid cell
    n_a = len(actions)
    obs = torch.tensor([tt[s]["trajectory"] for s in states for _ in range(n_a)],
                       dtype=torch.float32, device=device)
    tsq = torch.tensor([tt[s]["timestep"] for s in states for _ in range(n_a)], device=device)
    act = torch.tensor([[a] for _ in states for a in range(n_a)], device=device)
    Xq = _features(model, tsq, obs, act)
    preds = []
    for _ in range(n_hypotheses):
        W = mu + chol @ torch.randn_like(mu)
        preds.append((Xq @ W)[:, :obs_space].argmax(1))
    preds = torch.stack(preds).cpu().numpy()             # (K, pairs)

    rows = []
    for j, (s, a) in enumerate([(s, a) for s in states for a in range(n_a)]):
        ns = tt[s]["next_state"][a]
        if ns is None:
            continue
        tr, tc = divmod(int(ns), N)
        pr, pc = preds[:, j] // N, preds[:, j] % N
        consensus = int(np.bincount(preds[:, j]).argmax())
        rows.append({"state": s, "action": a,
                     "mean_grid_dist": float(np.mean(np.sqrt((pr - tr) ** 2 + (pc - tc) ** 2))),
                     "grid_variance": float(np.var(pr) + np.var(pc)),
                     "exact_match": int(consensus == ns)})
    res = pd.DataFrame(rows)
    mean_grid_dist = float(res["mean_grid_dist"].mean())
    overconf = float(((res["mean_grid_dist"] > DIST_TOL) & (res["grid_variance"] < VAR_EPS)).mean())

    # --- reward model: R^2 on training data, goal transition and its action margin -
    r_hat = (X @ mu)[:, -1]
    r2 = float(1 - ((r_hat - R.reshape(-1)) ** 2).mean() / R.var().clamp_min(1e-12))
    goal = R.reshape(-1) > 0.5
    goal_r_hat = goal_margin = float("nan")
    if goal.any():
        Xf = _features(model, T[goal], S[goal], 1 - A[goal])
        goal_r_hat = float(r_hat[goal].mean())
        goal_margin = float((r_hat[goal] - (Xf @ mu)[:, -1]).mean())

    return {
        "objectives": (mean_grid_dist, overconf),
        "mean_grid_dist": mean_grid_dist,
        "overconfident_error_rate": overconf,
        "exact_match_accuracy": float(res["exact_match"].mean()),
        "mean_grid_variance": float(res["grid_variance"].mean()),
        "goal_path_grid_dist": float(res.loc[res["state"] >= obs_space // 2, "mean_grid_dist"].mean()),
        "reward_r2": r2,
        "goal_r_hat": goal_r_hat,
        "goal_reward_margin": goal_margin,
        "goal_transitions": int(goal.sum()),
        "epochs_to_converge": epochs_to_converge,
        "best_train_loss": best_loss,
        "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
