"""
Descent attention visualisation for the headline BTRL runs, from each run's
final (1e6-step) transition checkpoint.

WHAT THE TOKENS ARE (verified against the architecture)
-------------------------------------------------------
The transition model's context is `context_length` (state, action) pairs,
interleaved into tokens (s0, a0, s1, a1, ...):
    s0 = previous state s_{t-1}   (zero-padding at episode start)
    s1 = current  state s_t       (where the agent is now)
    a* = the CANDIDATE action being evaluated, injected at each timestep
         position (same action id, different timestep embedding)
The next-state head flattens ALL tokens, so "where the model looks" =
attention RECEIVED by each token, averaged over queries.

THE 3-LAYER PROBLEM
-------------------
Raw per-layer maps don't compose trivially. We therefore use ATTENTION
ROLLOUT (Abnar & Zuidema 2020): per layer, average heads, add the identity
(residual connection), row-normalise, then multiply across layers. This yields
ONE effective (query -> key) map per state; averaging over queries gives, for
each context token, the fraction of the prediction's attention it receives.
Each row of the descent heatmap is thus a distribution over
[s_{t-1}, a_{t-1}, s_t, a_t] that sums to 1.

DESCENT
-------
For each diagonal state on the optimal path (row r -> state r*N+r), we set the
context to (previous diagonal state, current diagonal state) + candidate action
and read the rollout attention, averaged over the two candidate actions.
A correctly-learned Markov model of DeepSea should attend to s_t (current) and
the action, NOT to s_{t-1} (previous) — that is the interpretable signal.

Run: python3 visualize_attention_runs.py
"""
import json
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
LOG = "logdir/5-determinsitic"
K_LOCKED = 20  # locked-M̂ samples to average for a posterior-typical routing

RUNS = {
    "v2.1_replay1e4_forgets": f"{LOG}/BTRL-best_hparams_1M/1",
    "v2.2_HPO_relapses":      f"{LOG}/BTRL-best_combined_1M/0",
}


def resolve_best_v23():
    best, best_gr = None, -1
    for d in sorted(Path(LOG).glob("BTRL-v4_protected_seed*/*")):
        m = d / "metrics.jsonl"
        if not m.exists() or not (d / "checkpoints").exists():
            continue
        if not any((d / "checkpoints").iterdir()):
            continue
        rew = [json.loads(l.replace("NaN", "null")).get("Reward/Train_Reward") for l in open(m)]
        rew = [r for r in rew if r is not None]
        if len(rew) < 500:
            continue
        gr = np.mean([r > 0.5 for r in rew[-500:]])
        if gr > best_gr:
            best, best_gr = str(d), gr
    return best, best_gr


def load_model(run_dir: str):
    cfg = coerce_numeric(json.load(open(Path(run_dir) / "hyper_parameters.txt")))
    ckpts = sorted(int(p.name) for p in (Path(run_dir) / "checkpoints").iterdir())
    ckpt = Path(run_dir) / "checkpoints" / str(ckpts[-1]) / "transition.pt"
    env_size = int(cfg["experiment"]["env"])
    model = TransitionNetwork(cfg["transition"], env_size * env_size, [0, 1],
                              p_seq=cfg["transition"]["p_seq"], p_attn=cfg["transition"]["p_attn"],
                              max_steps=cfg["experiment"]["time_limit"], device="cpu")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model, cfg, env_size, ckpts[-1]


def rollout(maps):
    """Attention rollout over a list of (q,k) head-averaged maps. Returns received-attn vector."""
    n = maps[0].shape[-1]
    R = np.eye(n)
    for A in maps:
        A = A + np.eye(n)
        A = A / A.sum(-1, keepdims=True)
        R = A @ R
    return R.mean(0)  # attention received per key token (sums to 1)


def descent_attention(model, cfg, env_size, locked: bool):
    ctx = cfg["transition"]["context_length"]
    rows_out = []
    diag = [r * env_size + r for r in range(env_size)]
    for r, s_cur in enumerate(diag):
        s_prev = diag[r - 1] if r > 0 else None
        obs = torch.zeros(2, ctx, env_size * env_size)
        obs[:, -1, s_cur] = 1
        if s_prev is not None and ctx >= 2:
            obs[:, -2, s_prev] = 1
        ts = torch.tensor([[max(r - 1, 0)] * (ctx - 1) + [r]] * 2)
        act = torch.tensor([[0], [1]])

        samples = []
        for _ in range(K_LOCKED if locked else 1):
            if locked:
                model.lock_all_dropouts(2, ctx, "cpu")
            else:
                model.unlock_all_dropouts()
            with torch.no_grad():
                model.forward(ts, obs, act)
            maps_per_action = [blk.attention.attention_map.mean(1).numpy()  # (N_act, q, k)
                               for blk in model.encoder.layers]
            # rollout per action then average the two actions
            per_act = [rollout([m[a] for m in maps_per_action]) for a in range(2)]
            samples.append(np.mean(per_act, axis=0))
            if locked:
                model.unlock_all_dropouts()
        rows_out.append(np.mean(samples, axis=0))
    return np.array(rows_out)  # (n_rows, 2*ctx)


def visualize(name: str, run_dir: str):
    model, cfg, env_size, step = load_model(run_dir)
    ctx = cfg["transition"]["context_length"]
    mean_desc = descent_attention(model, cfg, env_size, locked=False)
    lock_desc = descent_attention(model, cfg, env_size, locked=True)

    col_labels = []
    for i in range(ctx):
        tag = "t" if i == ctx - 1 else f"t-{ctx-1-i}"
        col_labels += [f"s({tag})", f"a({tag})"]
    row_labels = [f"row{r}" + (" start" if r == 0 else " GOAL" if r == env_size - 1 else "")
                  for r in range(env_size)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, data, tag in [(axes[0], mean_desc, "dropout OFF (posterior mean)"),
                          (axes[1], lock_desc, f"locked M̂ (mean of {K_LOCKED})")]:
        im = ax.imshow(data, cmap="magma", vmin=0, vmax=data.max(), aspect="auto")
        ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_title(tag, fontsize=10)
        ax.set_xlabel("context token (attention received)", fontsize=9)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                        color="white" if data[i, j] < data.max() * 0.6 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_ylabel("descent (agent position)", fontsize=9)
    fig.suptitle(f"{name} — descent attention rollout @ {step:,} steps (DeepSea-{env_size})\n"
                 "each row = distribution over context tokens the next-state prediction draws from",
                 fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = PLOTS / f"descent_{name}.png"
    for ext in ("png", "pdf"):
        fig.savefig(str(out).replace(".png", f".{ext}"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}  (step={step:,})")


if __name__ == "__main__":
    runs = dict(RUNS)
    best_v23, gr = resolve_best_v23()
    if best_v23:
        runs["v2.3_protected_best"] = best_v23
        print(f"best v2.3 seed with checkpoints: {best_v23} (final-500 goal-rate {gr:.1%})")
    else:
        print("no v2.3 seed has a checkpoint yet — will fill in when the sweep reaches 100k")
    for name, d in runs.items():
        try:
            visualize(name, d)
        except Exception as e:
            print(f"SKIP {name}: {type(e).__name__}: {e}")
