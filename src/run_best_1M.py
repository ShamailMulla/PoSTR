"""
Full BTRL run on DeepSea-5 with the current best hyperparameters
(configs/config_vector-setting1.yaml), 1e6 environment steps, single seed.

Loads the config through TRL.tuning.common.load_base_config so scientific-
notation YAML values (1e6, 1e2, 1e-3, ...) are numeric — main.py's own
yaml.safe_load leaves them as strings, which only works when launched via
run_experiments.py (ruamel).

    nohup python3 -u run_best_1M.py > run_best_1M.log 2>&1 &
"""
import json

import main
from TRL.tuning.common import load_base_config

config = load_base_config(env="5", deterministic=True)

config["experiment"]["steps"] = 1_000_000
config["experiment"]["name"] = "best_hparams_1M"
config["experiment"]["seed"] = 0
config["save"] = True
config["save_freq"] = 100_000  # 10 checkpoints across the run

print(json.dumps(config, indent=2), flush=True)
main.main(config, 0)
