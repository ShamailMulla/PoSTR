"""
Stage 3 — online hyperparameter search for the value network.

The best transition config from validation is frozen; each trial runs the full
BTRL agent and is scored by seed-averaged cumulative regret (single objective,
minimised). Regret integrates both final performance and how quickly the agent
starts reaching the goal, so it is the sample-efficiency metric PSDRL-style
methods report.

discount is deliberately NOT tuned: DeepSea episodes are a fixed N-step chain
with a single terminal reward, so gamma rescales all state values by the same
monotonic factor and cannot change the greedy action ranking. It stays at the
base-config value.
"""

TRAINING_ITER_CHOICES = [5, 10, 20, 50]
HIDDEN_LAYERS_CHOICES = [1, 2, 3, 4]
HIDDEN_DIM_CHOICES = [8, 16, 32, 64, 128, 256]
TARGET_UPDATE_CHOICES = [5, 10, 20, 50]


def suggest_value_params(trial) -> dict:
    return {
        "training_iterations": trial.suggest_categorical("training_iterations", TRAINING_ITER_CHOICES),
        "hidden_layers": trial.suggest_categorical("hidden_layers", HIDDEN_LAYERS_CHOICES),
        "hidden_dim": trial.suggest_categorical("hidden_dim", HIDDEN_DIM_CHOICES),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "target_update_freq": trial.suggest_categorical("target_update_freq", TARGET_UPDATE_CHOICES),
    }


def apply_value_params(config: dict, params: dict) -> dict:
    """Write trial parameters into config['value'] (in place). discount untouched."""
    for key, value in params.items():
        config["value"][key] = value
    return config
