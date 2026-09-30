"""
num_encoder_layers ablation: everything fixed (tuned + protected v2.3 config,
same seed), vary ONLY the number of transformer encoder layers. Tests the claim
that one attention layer suffices for DeepSea and deeper layers stay uniform.

  python run_layer_ablation.py --layers 1 --seed 1

Launch 1/2/3 in parallel, each writes to BTRL-ablation_L{n}_seed{s}/.
"""
import argparse
import json
from datetime import datetime as dt

import numpy as np
import torch

import main
from TRL.tuning.common import load_config, ARTIFACTS_DIR

STEPS = 300_000       # enough to plateau + give 100k/200k/300k attention checkpoints
SAVE_FREQ = 100_000


def run(layers: int, seed: int):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cfg = load_config(ARTIFACTS_DIR / "best_combined_config.yaml")
    cfg["replay"]["protected_fraction"] = 0.1
    cfg["transition"]["num_encoder_layers"] = layers   # the only variable
    cfg["experiment"]["steps"] = STEPS
    cfg["experiment"]["seed"] = seed
    cfg["experiment"]["name"] = f"ablation_L{layers}_seed{seed}"
    cfg["save"] = True
    cfg["save_freq"] = SAVE_FREQ
    print(f"[{dt.now():%H:%M:%S}] layers={layers} seed={seed} name={cfg['experiment']['name']}")
    print(json.dumps(cfg["transition"], indent=2))
    main.main(cfg, seed)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    run(a.layers, a.seed)
