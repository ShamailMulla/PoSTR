"""
Two complementary attention views for one finished run (default v2.2):

  (A) PER-LAYER maps: for one descent state, each layer's raw softmax attention
      (heads averaged), query (from) x key (to) over the context tokens
      (s0, a0, s1, a1). Shows how each layer routes information.

  (B) GRID PROJECTION: the key point about interpreting "diagonal focus".
      The transformer attends over the CONTEXT WINDOW (last context_length
      states), not over the 25 grid cells. With context_length=2 the model
      only ever sees 2 cells at a time. We place each state-token's rollout
      attention onto the actual grid cell it represents, one 5x5 grid per
      descent step, so you can see the lit cells walk DOWN THE DIAGONAL across
      time — the diagonal appears over the descent, not inside one map.

Run: python3 attention_perlayer_grid.py [run_dir]
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
    return m, cfg, env, ckpts[-1]


def context_for_row(r, ctx, env):
    diag = [i * env + i for i in range(env)]
    obs = torch.zeros(2, ctx, env * env)
    obs[:, -1, diag[r]] = 1
    if r > 0 and ctx >= 2:
        obs[:, -2, diag[r - 1]] = 1
    ts = torch.tensor([[max(r - 1, 0)] * (ctx - 1) + [r]] * 2)
    return obs, ts, torch.tensor([[0], [1]]), diag


def layer_maps(model, obs, ts, act):
    with torch.no_grad():
        model.forward(ts, obs, act)
    return [blk.attention.attention_map.mean(1).numpy() for blk in model.encoder.layers]  # per layer (2,q,k)


def rollout_received(maps_action):
    n = maps_action[0].shape[-1]
    R = np.eye(n)
    for A in maps_action:
        A = A + np.eye(n); A = A / A.sum(-1, keepdims=True); R = A @ R
    return R.mean(0)


# ---------- (A) per-layer maps for a mid-descent state ----------
model, cfg, env, step = load_model(RUN)
ctx = cfg["transition"]["context_length"]
r_show = env // 2
obs, ts, act, diag = context_for_row(r_show, ctx, env)
maps = layer_maps(model, obs, ts, act)  # list of (2,q,k), averaged over the 2 actions below
labels = []
for i in range(ctx):
    tag = "t" if i == ctx - 1 else f"t-{ctx-1-i}"
    labels += [f"s({tag})", f"a({tag})"]

nL = len(maps)
fig, axes = plt.subplots(1, nL, figsize=(3.6 * nL, 3.4), squeeze=False)
for L in range(nL):
    A = maps[L].mean(0)  # avg the 2 actions
    ax = axes[0][L]
    im = ax.imshow(A, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("key (attended to)", fontsize=8)
    if L == 0:
        ax.set_ylabel("query (from)", fontsize=8)
    ax.set_title(f"layer {L}", fontsize=10)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center",
                    color="white" if A[i, j] < 0.6 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle(f"{NAME} per-layer attention (heads avg) — agent at row {r_show}, "
             f"context tokens over last {ctx} steps @ {step:,} steps", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(PLOTS / f"perlayer_{NAME}.png", dpi=170, bbox_inches="tight")
fig.savefig(PLOTS / f"perlayer_{NAME}.pdf", bbox_inches="tight")
plt.close(fig)
print("saved perlayer_" + NAME + ".png")

# ---------- (B) grid projection across the descent ----------
fig, axes = plt.subplots(1, env, figsize=(2.5 * env, 2.9), squeeze=False)
for r in range(env):
    obs, ts, act, diag = context_for_row(r, ctx, env)
    m2 = layer_maps(model, obs, ts, act)
    recv = np.mean([rollout_received([mm[a] for mm in m2]) for a in range(2)], axis=0)  # 2*ctx vec
    # state tokens are at even indices; map to grid cells they represent
    grid = np.full((env, env), np.nan)
    state_positions = list(range(0, 2 * ctx, 2))  # s(t-ctx+1)...s(t)
    cells = ([diag[r - 1]] if (r > 0 and ctx >= 2) else [None]) * (ctx - 1) + [diag[r]]
    # align: last state token = current cell; earlier = previous cells
    cells = []
    for k in range(ctx):
        rr = r - (ctx - 1 - k)
        cells.append(diag[rr] if rr >= 0 else None)
    action_attn = sum(recv[i] for i in range(1, 2 * ctx, 2))
    for k, sp in enumerate(state_positions):
        if cells[k] is not None:
            grid[cells[k] // env, cells[k] % env] = recv[sp]
    ax = axes[0][r]
    im = ax.imshow(grid, cmap="magma", vmin=0, vmax=np.nanmax(recv))
    ax.set_title(f"row {r}" + (" (start)" if r == 0 else " (GOAL)" if r == env - 1 else "")
                 + f"\naction attn={action_attn:.2f}", fontsize=8)
    ax.set_xticks(range(env)); ax.set_yticks(range(env))
    ax.set_xticklabels(range(env), fontsize=7); ax.set_yticklabels(range(env), fontsize=7)
    ax.set_xlabel("col", fontsize=7)
    if r == 0:
        ax.set_ylabel("row", fontsize=7)
    for cell in cells:
        if cell is not None:
            ax.text(cell % env, cell // env, "•", ha="center", va="center",
                    color="cyan", fontsize=9)
fig.suptitle(f"{NAME} attention projected onto the 5x5 grid, per descent step "
             f"(cyan dot = cell in context). The lit cells walk DOWN THE DIAGONAL across time —\n"
             f"the model sees only {ctx} cells at once, so the full diagonal appears over the descent, not within one map.",
             fontsize=8.5)
plt.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(PLOTS / f"grid_{NAME}.png", dpi=170, bbox_inches="tight")
fig.savefig(PLOTS / f"grid_{NAME}.pdf", bbox_inches="tight")
plt.close(fig)
print("saved grid_" + NAME + ".png")
