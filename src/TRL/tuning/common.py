"""
Shared infrastructure for the BTRL hyperparameter-tuning notebooks.

Everything the three HPO notebooks need to agree on lives here: artifact
paths, the shared Optuna storage URL, config loading with numeric coercion,
and global seeding.
"""
import random
from pathlib import Path

import numpy as np
import torch
import yaml

# src/hpo_artifacts/ — shared by all three HPO notebooks.
SRC_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = SRC_DIR / "hpo_artifacts"

OPTUNA_DB = f"sqlite:///{ARTIFACTS_DIR / 'optuna_btrl.db'}"

TRANSITION_STUDY_NAME = "btrl_transition_hpo"
VALUE_STUDY_NAME = "btrl_value_hpo"

# Artifacts passed between notebooks (never notebook kernel state).
TRANSITION_PARETO_JSON = ARTIFACTS_DIR / "transition_pareto.json"
VALIDATION_RESULTS_CSV = ARTIFACTS_DIR / "validation_results.csv"
BEST_TRANSITION_CONFIG = ARTIFACTS_DIR / "best_transition_config.yaml"
BEST_VALUE_CONFIG = ARTIFACTS_DIR / "best_value_config.yaml"

DEFAULT_CONFIG_PATH = SRC_DIR / "configs" / "config_vector-setting1.yaml"

# Keys that must be numeric for the agent/harness to work. PyYAML (YAML 1.1)
# parses bare scientific notation such as `1e1` or `1e-3` as *strings*, so a
# config loaded with yaml.safe_load silently carries string values into
# arithmetic. Coerce everything the training loop touches.
_NUMERIC_KEYS = {
    "algorithm": ["update_freq", "warmup_length", "warmup_freq", "policy_noise", "n_samples", "temp"],
    "experiment": ["steps", "time_limit", "test_freq"],
    "replay": ["capacity", "batch_size", "sequence_length", "protected_fraction"],
    "transition": [
        "training_iterations", "hidden_dim", "learning_rate", "window_length",
        "context_length", "num_encoder_layers", "heads", "forward_expansion",
        "p_seq", "p_attn",
    ],
    "value": ["training_iterations", "hidden_layers", "hidden_dim", "learning_rate", "target_update_freq", "discount"],
    "terminal": ["training_iterations", "hidden_layer", "hidden_dim", "learning_rate"],
}

_INT_KEYS = {
    "update_freq", "warmup_length", "warmup_freq", "n_samples",
    "steps", "time_limit", "test_freq",
    "capacity", "batch_size", "sequence_length",
    "training_iterations", "hidden_dim", "window_length", "context_length",
    "num_encoder_layers", "heads", "forward_expansion",
    "hidden_layers", "hidden_layer", "target_update_freq",
}


def coerce_numeric(config: dict) -> dict:
    for section, keys in _NUMERIC_KEYS.items():
        for key in keys:
            if section in config and key in config[section]:
                value = float(config[section][key])
                config[section][key] = int(value) if key in _INT_KEYS else value
    return config


def load_base_config(path=DEFAULT_CONFIG_PATH, env: str = "5", deterministic: bool = True) -> dict:
    """
    Load a BTRL YAML config with numeric coercion and the derived fields that
    main.py normally fills in (time_limit, sequence_length).
    """
    with open(path) as f:
        config = yaml.safe_load(f)
    config = coerce_numeric(config)

    config["experiment"]["env"] = str(env)
    config["experiment"]["deterministic"] = deterministic
    config["experiment"]["seed"] = None
    config["load"] = False
    # main.py derives both from the environment depth
    config["experiment"]["time_limit"] = int(env) + 1
    config["replay"]["sequence_length"] = int(env) + 1
    return config


def save_config(config: dict, path) -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def load_config(path) -> dict:
    with open(path) as f:
        return coerce_numeric(yaml.safe_load(f))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config: dict) -> str:
    if config.get("gpu") and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"
