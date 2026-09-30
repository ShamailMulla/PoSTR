"""
BTRL v3: 1e6-step run on DeepSea-5 with the fully HPO-tuned configuration
(hpo_artifacts/best_combined_config.yaml — validated transition params from
hpo1+hpo2, value params from hpo3, replay capacity 1e4).

    nohup python3 -u run_combined_1M.py > run_combined_1M.log 2>&1 &
"""
import json

import main
from TRL.tuning.common import load_config, ARTIFACTS_DIR

config = load_config(ARTIFACTS_DIR / "best_combined_config.yaml")

config["experiment"]["steps"] = 1_000_000
config["experiment"]["name"] = "best_combined_1M"
config["experiment"]["seed"] = 0
config["save"] = True
config["save_freq"] = 100_000

print(json.dumps(config, indent=2), flush=True)
main.main(config, 0)
