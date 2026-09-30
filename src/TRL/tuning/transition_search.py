"""
Stage 1 — offline hyperparameter search for the Transformer transition network.

Trials are trained and evaluated on a fixed replay dataset (no environment
interaction), so they are cheap and their score is attributable to the
transition hyperparameters alone.

Objectives (both minimised, NSGA-II Pareto front):
  1. mean_grid_dist — mean Euclidean distance in (row, col) space between each
     sampled world hypothesis M-hat's predicted next state and the true next
     state, over N_HYPOTHESES locked-dropout samples and all (state, action)
     test pairs. Measures world-model accuracy.
  2. overconfident_error_rate — fraction of (state, action) pairs where the
     posterior is both wrong (mean_grid_dist > DIST_TOL) and near-certain
     (grid_variance < VAR_EPS). An overconfident-wrong posterior is the exact
     failure mode that breaks Thompson Sampling: every sampled M-hat agrees on
     the wrong transition, so no episode ever explores the correction.
     Measures posterior quality, which mean accuracy alone cannot.

Downstream-relevant diagnostics (reward-head MSE, goal-path grid distance,
exact-match accuracy) are logged as user attributes for the validation stage.
"""
import sys

import torch

from .common import SRC_DIR

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from notebook_utils import early_stopping_train, test_model_locked  # noqa: E402
from ..networks.transition import Network as TransitionNetwork  # noqa: E402

# All (hidden_dim, heads) combinations with hidden_dim % heads == 0, sampled as
# a single index so head_dim is always a valid integer.
VALID_PAIRS = [
    (dim, heads)
    for dim in [32, 64, 128, 256, 512]
    for heads in [1, 2, 4, 8]
    if dim % heads == 0
]

# Sequence/embedding dropout models observation noise (higher range);
# Q/K/V + attention-map + FFN dropout model epistemic uncertainty.
P_SEQ_CHOICES = [0.05, 0.1, 0.2, 0.3]
P_ATTN_CHOICES = [0.02, 0.05, 0.1, 0.15]

# Miscalibration thresholds: wrong by more than half a grid cell while the
# posterior spread is below VAR_EPS counts as overconfident-wrong.
DIST_TOL = 0.5
VAR_EPS = 0.05


def suggest_transition_params(trial) -> dict:
    pair_idx = trial.suggest_int("hd_heads_idx", 0, len(VALID_PAIRS) - 1)
    hidden_dim, heads = VALID_PAIRS[pair_idx]
    return {
        "hidden_dim": hidden_dim,
        "heads": heads,
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "num_encoder_layers": trial.suggest_int("num_encoder_layers", 1, 4),
        "forward_expansion": trial.suggest_categorical("forward_expansion", [2, 4, 8]),
        "p_seq": trial.suggest_categorical("p_seq", P_SEQ_CHOICES),
        "p_attn": trial.suggest_categorical("p_attn", P_ATTN_CHOICES),
    }


def apply_transition_params(config: dict, params: dict) -> dict:
    """Write trial parameters into config['transition'] (in place)."""
    for key, value in params.items():
        config["transition"][key] = value
    return config


def train_and_eval_transition(
    config: dict,
    params: dict,
    data: dict,
    device,
    max_epochs: int = 300,
    patience: int = 15,
    min_delta: float = 5e-4,
    n_hypotheses: int = 100,
) -> dict:
    """
    Train one transition network on the shared offline dataset and evaluate it
    with locked-dropout posterior sampling.

    data — dict with tensors 'states', 'next_states', 'timesteps', 'actions',
    'rewards' plus 'test_transitions', 'obs_space', 'possible_actions'
    (as produced by notebook_utils.generate_transition_data).

    Returns a metrics dict; 'objectives' holds the two Pareto objectives.
    """
    trial_cfg = dict(config["transition"])
    for key in ["hidden_dim", "heads", "learning_rate", "num_encoder_layers", "forward_expansion"]:
        trial_cfg[key] = params[key]

    obs_space = data["obs_space"]
    max_steps = int(torch.max(data["timesteps"]).item()) + 1

    model = TransitionNetwork(
        trial_cfg,
        obs_space,
        data["possible_actions"],
        p_seq=params["p_seq"],
        p_attn=params["p_attn"],
        max_steps=max_steps,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    _, epochs_to_converge, best_loss = early_stopping_train(
        model,
        data["states"], data["next_states"], data["timesteps"],
        data["actions"], data["rewards"],
        optimizer,
        obs_type=config["transition"]["obs_type"], obs_space=obs_space,
        max_epochs=max_epochs, patience=patience, min_delta=min_delta,
        device=device,
    )

    results = test_model_locked(
        model, data["test_transitions"], data["possible_actions"],
        n_hypotheses=n_hypotheses,
    )

    mean_grid_dist = float(results["mean_grid_dist"].mean())
    overconfident = (results["mean_grid_dist"] > DIST_TOL) & (results["grid_variance"] < VAR_EPS)
    overconfident_error_rate = float(overconfident.mean())

    # Goal-approach transitions: bottom half of the grid, the states that
    # decide whether the optimal path is representable.
    goal_mask = results["current_state"] >= obs_space // 2
    goal_df = results[goal_mask]
    goal_path_grid_dist = float(goal_df["mean_grid_dist"].mean()) if len(goal_df) else float("nan")

    return {
        "objectives": (mean_grid_dist, overconfident_error_rate),
        "mean_grid_dist": mean_grid_dist,
        "overconfident_error_rate": overconfident_error_rate,
        "reward_mse": float(results["reward_mse"].mean()),
        "goal_path_grid_dist": goal_path_grid_dist,
        "exact_match_accuracy": float(results["exact_match"].mean()),
        "mean_grid_variance": float(results["grid_variance"].mean()),
        "max_grid_dist": float(results["mean_grid_dist"].max()),
        "epochs_to_converge": epochs_to_converge,
        "best_train_loss": best_loss,
        "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
