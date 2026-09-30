"""
Multi-seed runner for the tuned, protected-buffer BTRL agent (v2.3).

Loads the HPO-optimised configuration (hpo_artifacts/best_combined_config.yaml:
512-dim transition, 16-dim value, replay capacity 1e4) and enables the
protected replay partition (replay.protected_fraction = 0.1), i.e. exactly the
v2.3 setup, then runs one full 1e6-step experiment per seed.

Each seed:
  * gets a distinct, reproducible experiment name  ->  its own logdir folder,
    so parallel launches never race on directory allocation and never overwrite
    an existing run (seed 0 lives in BTRL-v4_protected_1M/).
  * seeds numpy AND torch so the sweep is controlled, not just OS-entropy noise.

Usage
-----
  # one seed (use this to launch several in parallel, one process per seed):
  python run_experiments.py --seed 1

  # several seeds sequentially in a single process:
  python run_experiments.py --seeds 1,2,3,4
"""
import argparse
import json
from datetime import datetime as dt

import numpy as np
import torch

import main
from TRL.tuning.common import load_config, ARTIFACTS_DIR

CONFIG_PATH = ARTIFACTS_DIR / "best_combined_config.yaml"
PROTECTED_FRACTION = 0.1
STEPS = 1_000_000
SAVE_FREQ = 100_000


def build_config(seed: int) -> dict:
    # load_config applies numeric coercion (YAML leaves 1e-3, 1e4 as strings).
    config = load_config(CONFIG_PATH)
    config["replay"]["protected_fraction"] = PROTECTED_FRACTION  # -> v2.3
    config["experiment"]["steps"] = STEPS
    config["experiment"]["seed"] = seed
    # Distinct name per seed => distinct logdir folder (BTRL-<name>/0). Avoids the
    # create_directories() race under parallel launch and never overwrites seed 0.
    config["experiment"]["name"] = f"v4_protected_seed{seed}"
    config["save"] = True
    config["save_freq"] = SAVE_FREQ
    return config


def run_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config = build_config(seed)
    print("cuda:", torch.cuda.is_available(), "| seed:", seed, "| name:", config["experiment"]["name"])
    print(json.dumps(config, indent=2))
    begin = dt.now()
    print("\n", "*" * 10, "Starting seed", seed, "@", begin, "*" * 10, "\n")
    main.main(config, seed)
    print("Time taken:", (dt.now() - begin).total_seconds() // 60, "minutes\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="Run a single seed (for launching parallel processes).")
    parser.add_argument("--seeds", type=str, default="1,2,3,4",
                        help="Comma-separated seeds to run sequentially (ignored if --seed given).")
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else [int(s) for s in args.seeds.split(",")]
    for s in seeds:
        run_seed(s)
