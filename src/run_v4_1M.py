"""
BTRL v4: the v3 configuration (HPO-tuned transition + value networks,
capacity 1e4) plus the protected replay partition (protected_fraction 0.1).
Single-variable ablation against v3 — reward-aware retention. Version-history
context: ../wiki/failure-modes.md.

    nohup python3 -u run_v4_1M.py > run_v4_1M.log 2>&1 &
"""
import json

import main
from TRL.tuning.common import load_config, ARTIFACTS_DIR

config = load_config(ARTIFACTS_DIR / "best_combined_config.yaml")

config["replay"]["protected_fraction"] = 0.1  # the one variable changed vs v3
config["experiment"]["steps"] = 1_000_000
config["experiment"]["name"] = "v4_protected_1M"
config["experiment"]["seed"] = 0
config["save"] = True
config["save_freq"] = 100_000

print(json.dumps(config, indent=2), flush=True)
main.main(config, 0)
