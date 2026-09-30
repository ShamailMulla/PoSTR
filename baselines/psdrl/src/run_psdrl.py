"""PSDRL 1M-step DeepSea run, one seed per process (mirrors src/run_v3_1M.py):

    python run_psdrl.py --seed 1 --name psdrl_rand

Logs to logdir/<env>/PSDRL-<name>_seed<seed>/0/. If MLFLOW_RUN_ID is set (see
src/run_comparison.sh), the process attaches to that MLflow run so system metrics
land on it; the training metrics are streamed separately by src/mlflow_sync.py.
"""
import argparse
import os

import numpy as np
import torch
import yaml

import main


def run(seed: int, config_path: str, name: str, steps: int):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["experiment"].update(seed=seed, steps=steps, name=f"{name}_seed{seed}")
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        main.main(cfg)
        return
    import mlflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_system_metrics_sampling_interval(30)
    mlflow.enable_system_metrics_logging()
    with mlflow.start_run(run_id=run_id):
        main.main(cfg)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--config", type=str, default="configs/psdrl_deepsea5_prior1e3_1M.yaml")
    p.add_argument("--name", type=str, default="psdrl_rand")
    p.add_argument("--steps", type=int, default=1_000_000)
    a = p.parse_args()
    run(a.seed, a.config, a.name, a.steps)
