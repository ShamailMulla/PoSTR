"""
BTRL v3 (neural-linear head) 1e6-step run on DeepSea-5. One seed per process:

    python run_v3_1M.py --seed 1

config_v3_neural_linear.yaml: tuned transition arch, dropout off, BLR posterior,
protected_fraction 0 (clean test of whether the data-coupled posterior alone
holds the plateau where every dropout variant collapsed).
"""
import argparse
import json
import os
from datetime import datetime as dt

import numpy as np
import torch

import main
from TRL.tuning.common import load_config


def run(seed: int, config_path: str, name: str, temp: float = None, steps: int = 1_000_000,
        save_freq: int = 100_000, explore_temp: float = None, resume_from: str = None,
        resume_freq: int = 20_000):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cfg = load_config(config_path)
    cfg["experiment"]["steps"] = steps
    cfg["experiment"]["seed"] = seed
    cfg["experiment"]["name"] = f"{name}_seed{seed}"
    if temp is not None:
        cfg["algorithm"]["transition_temp"] = temp   # value-training posterior sampling spread
    if explore_temp is not None:
        cfg["algorithm"]["explore_temp"] = explore_temp   # wide acting/exploration spread
    cfg["save"] = True
    cfg["save_freq"] = save_freq
    cfg["experiment"]["resume_freq"] = resume_freq   # rolling checkpoints/latest/resume.pt
    if resume_from:
        cfg["resume_from"] = resume_from
    print(f"[{dt.now():%H:%M:%S}] v3 seed={seed} | name={cfg['experiment']['name']} "
          f"| deterministic={cfg['experiment']['deterministic']} "
          f"| neural_linear={cfg['algorithm']['neural_linear']}")
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        main.main(cfg, seed)
        return
    # Attach to the run mlflow_sync.py created, so traces (TRL/common/tracing.py)
    # and system metrics land on it; the sync script logs the training metrics.
    import mlflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_system_metrics_sampling_interval(30)
    mlflow.enable_system_metrics_logging()
    with mlflow.start_run(run_id=run_id):
        main.main(cfg, seed)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--config", type=str, default="configs/config_v3_neural_linear.yaml")
    p.add_argument("--name", type=str, default="v3_neural_linear")
    p.add_argument("--temp", type=float, default=None)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--save_freq", type=int, default=100_000)
    p.add_argument("--explore_temp", type=float, default=None)
    p.add_argument("--resume_from", type=str, default=None, help="path to a resume.pt checkpoint")
    p.add_argument("--resume_freq", type=int, default=20_000)
    a = p.parse_args()
    run(a.seed, a.config, a.name, a.temp, a.steps, a.save_freq, a.explore_temp, a.resume_from, a.resume_freq)
