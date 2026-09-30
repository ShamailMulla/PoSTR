"""
NMT-style ("Bahdanau/Karpathy") full-sequence attention alignment for a BTRL run.

Feeds the ENTIRE descent trajectory (all N diagonal states as one sequence) to
the encoder and reads the full (2N x 2N) attention map over the interleaved
(s0,a0, s1,a1, ..., sN-1,aN-1) tokens — rows = query (position being computed),
columns = key (position attended to), with the full off-diagonal structure.

IMPORTANT CAVEAT: the model was TRAINED with context_length=2, so a length-N
input is out-of-distribution. The prediction head is also fixed to 2*ctx tokens,
so we call the encoder directly (attention only). This figure therefore shows
what alignment structure the learned attention *generalises* to over a full
trajectory — NOT the operative rollout behaviour (which only ever sees 2 states).
A faithful NMT-style map would require training with context_length = N.

Run: python3 attention_nmt_style.py [run_dir] [name]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from TRL.networks.transition import Network as TransitionNetwork
from TRL.tuning.common import coerce_numeric

PLOTS = Path(__file__).resolve().parents[1] / "plots" / "attention"
PLOTS.mkdir(parents=True, exist_ok=True)
RUN = sys.argv[1] if len(sys.argv) > 1 else "logdir/5-determinsitic/BTRL-best_combined_1M/0"
NAME = sys.argv[2] if len(sys.argv) > 2 else "v2.2_HPO"


def load_model(run_dir):
    cfg = coerce_numeric(json.load(open(Path(run_dir) / "hyper_parameters.txt")))
    ckpts = sorted(int(p.name) for p in (Path(run_dir) / "checkpoints").iterdir())
    env = int(cfg["experiment"]["env"])
    m = TransitionNetwork(cfg["transition"], env * env, [0, 1],
                          p_seq=cfg["transition"]["p_seq"], p_attn=cfg["transition"]["p_attn"],
                          max_steps=cfg["experiment"]["time_limit"], device="cpu")
    m.load_state_dict(torch.load(Path(run_dir) / "checkpoints" / str(ckpts[-1]) / "transition.pt",
                                 map_location="cpu"))
    m.eval(); m.unlock_all_dropouts()
    return m, env, ckpts[-1]


def rollout(maps):
    n = maps[0].shape[-1]
    R = np.eye(n)
    for A in maps:
        A = A + np.eye(n); A = A / A.sum(-1, keepdims=True); R = A @ R
    return R


def full_traj_attention(model, env):
    diag = [r * env + r for r in range(env)]
    obs = torch.zeros(2, env, env * env)
    for i, s in enumerate(diag):
        obs[:, i, s] = 1
    ts = torch.tensor([list(range(env))] * 2)
    act = torch.tensor([[0], [1]])
    with torch.no_grad():
        model.encode(ts, obs, act)  # encoder only (prediction head is fixed-size)
    per_layer = [blk.attention.attention_map.mean(1).numpy() for blk in model.encoder.layers]  # (2,2N,2N)
    # average the two candidate actions
    layers_avg = [m.mean(0) for m in per_layer]
    roll = np.mean([rollout([m[a] for m in per_layer]) for a in range(2)], axis=0)
    return layers_avg, roll, env


def token_labels(env):
    labels = []
    for r in range(env):
        labels += [f"s{r}", f"a{r}"]
    return labels


def draw(ax, M, labels, title, annotate):
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=M.max())
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7, rotation=90)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("key (attended to)", fontsize=8)
    ax.set_ylabel("query (from)", fontsize=8)
    ax.set_title(title, fontsize=9)
    if annotate:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < M.max() * 0.6 else "black", fontsize=5.5)
    return im


def visualize(name, run_dir):
    model, env, step = load_model(run_dir)
    layers_avg, roll, env = full_traj_attention(model, env)
    labels = token_labels(env)
    nL = len(layers_avg)

    fig, axes = plt.subplots(1, nL + 1, figsize=(4.2 * (nL + 1), 4.4))
    for L in range(nL):
        im = draw(axes[L], layers_avg[L], labels, f"layer {L}", annotate=False)
        fig.colorbar(im, ax=axes[L], fraction=0.046)
    im = draw(axes[nL], roll, labels, "rollout (all layers)", annotate=True)
    fig.colorbar(im, ax=axes[nL], fraction=0.046)
    fig.suptitle(f"{name} — NMT-style full-trajectory attention @ {step:,} steps (DeepSea-{env})\n"
                 f"rows=query position, cols=key attended to; interleaved (s0,a0,...,s{env-1},a{env-1}). "
                 f"CAVEAT: trained ctx=2, full-traj input is OOD (encoder-only probe).",
                 fontsize=8.5)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    out = PLOTS / f"nmt_{name}.png"
    for ext in ("png", "pdf"):
        fig.savefig(str(out).replace(".png", f".{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}  (step={step:,}, tokens={2*env})")


if __name__ == "__main__":
    visualize(NAME, RUN)
