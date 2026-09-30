"""
Thesis figures for the v1/v2/v3 DeepSea-5 runs (PNG + PDF into ../plots/).

  fig_v3_learning_curve   — rolling mean return + rolling goal-rate vs steps
  fig_v3_test_reward      — greedy test-episode reward vs steps
  fig_v123_goal_rate      — rolling goal-rate, all three runs
  fig_v123_regret         — cumulative regret, all three runs

Run: python3 make_thesis_plots.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OPT = 0.982
PLOTS = Path(__file__).resolve().parents[1] / "plots"
PLOTS.mkdir(exist_ok=True)

RUNS = {
    "v2.1 (replay 1e2): world-model collapse": "logdir/5-determinsitic/BTRL-best_hparams_1M/0/metrics.jsonl",
    "v2.1 (replay 1e4): forgetting cycle": "logdir/5-determinsitic/BTRL-best_hparams_1M/1/metrics.jsonl",
    "v2.2 HPO-tuned: relapse recurs": None,  # resolved below (newest best_combined run)
    "v2.3 protected replay (in progress)": None,
}
combined = sorted(Path("logdir/5-determinsitic/BTRL-best_combined_1M").glob("*/metrics.jsonl"))
RUNS["v2.2 HPO-tuned: relapse recurs"] = str(combined[-1])
v4 = sorted(Path("logdir/5-determinsitic/BTRL-v4_protected_1M").glob("*/metrics.jsonl"))
if v4:
    RUNS["v2.3 protected replay (in progress)"] = str(v4[-1])
else:
    del RUNS["v2.3 protected replay (in progress)"]

COLORS = {"v2.1 (replay 1e2)": "tab:red", "v2.1 (replay 1e4)": "tab:orange",
          "v2.2": "tab:green", "v2.3": "tab:blue"}


def load(path):
    steps, train, test_steps, test = [], [], [], []
    for line in open(path):
        try:
            r = json.loads(line.replace("NaN", "null"))
        except Exception:
            continue
        if r.get("Reward/Train_Reward") is not None:
            steps.append(r["Timestep"]); train.append(r["Reward/Train_Reward"])
        if r.get("Reward/Test_Reward") is not None:
            test_steps.append(r["Timestep"]); test.append(r["Reward/Test_Reward"])
    return np.array(steps), np.array(train), np.array(test_steps), np.array(test)


def rolling(x, w):
    if len(x) < w:
        return np.array([]), slice(0, 0)
    return np.convolve(x, np.ones(w) / w, mode="valid"), slice(w - 1, None)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", PLOTS / f"{name}.png/.pdf")


data = {label: load(p) for label, p in RUNS.items()}
W = 1000  # rolling window (episodes)

# ── fig 1: v3 learning curve ────────────────────────────────────────────────
steps, train, tsteps, test = data["v2.2 HPO-tuned: relapse recurs"]
goal = (train > 0.5).astype(float)
fig, ax1 = plt.subplots(figsize=(9, 4.5))
r_ret, sl = rolling(train, W)
ax1.plot(steps[sl], r_ret, color="tab:blue", label=f"mean episode return ({W}-ep window)")
ax1.axhline(OPT, color="grey", ls=":", lw=1, label="optimal return (0.982)")
ax1.set_xlabel("environment steps")
ax1.set_ylabel("episode return", color="tab:blue")
ax2 = ax1.twinx()
r_goal, sl = rolling(goal, W)
ax2.plot(steps[sl], 100 * r_goal, color="tab:green", alpha=0.8,
         label=f"goal-rate ({W}-ep window)")
ax2.set_ylabel("goal-rate (%)", color="tab:green")
ax2.set_ylim(0, 100)
lines = ax1.get_lines() + ax2.get_lines()
ax1.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=8)
ax1.set_title("BTRL v2.2 (HPO-tuned) on DeepSea-5 — learning curve, 10⁶ steps")
save(fig, "fig_v22_learning_curve")

# ── fig 2: v3 test reward ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(tsteps, test, ".", ms=2, alpha=0.4, color="tab:purple")
r_test, sl = rolling(test, 25)
ax.plot(tsteps[sl], r_test, color="tab:purple", lw=1.5, label="25-test rolling mean")
ax.axhline(OPT, color="grey", ls=":", lw=1)
ax.set_xlabel("environment steps"); ax.set_ylabel("greedy test-episode reward")
ax.set_title("BTRL v2.2 — greedy (test) policy performance")
ax.legend(fontsize=8)
save(fig, "fig_v22_test_reward")

# ── fig 3: goal-rate comparison ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
for label, (steps, train, *_ ) in data.items():
    key = next(k for k in COLORS if label.startswith(k))
    r, sl = rolling((train > 0.5).astype(float), W)
    ax.plot(steps[sl], 100 * r, color=COLORS[key], label=label, lw=1.4)
ax.axhline(100 * 2**-5, color="grey", ls=":", lw=1, label="random policy (~3%)")
ax.set_xlabel("environment steps"); ax.set_ylabel(f"goal-rate (%, {W}-episode window)")
ax.set_ylim(-2, 60)
ax.set_title("Goal-reaching rate — replay-capacity and HPO ablation (DeepSea-5)")
ax.legend(fontsize=8)
save(fig, "fig_versions_goal_rate")

# ── fig 4: cumulative regret comparison ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
for label, (steps, train, *_ ) in data.items():
    key = next(k for k in COLORS if label.startswith(k))
    ax.plot(steps, np.cumsum(OPT - train), color=COLORS[key], label=label, lw=1.4)
ax.plot([0, 1e6], [0, OPT * 1e6 / 5], color="grey", ls=":", lw=1,
        label="never-solves slope (upper bound)")
ax.set_xlabel("environment steps"); ax.set_ylabel("cumulative regret")
ax.set_title("Cumulative regret — lower is better; flat = solved")
ax.legend(fontsize=8)
save(fig, "fig_versions_regret")

# ── final v3 stats for the write-up ─────────────────────────────────────────
steps, train, *_ = data["v2.2 HPO-tuned: relapse recurs"]
goal = train > 0.5
n = len(train)
print(f"\nv3 final: {n:,} episodes | goal episodes {goal.sum():,} ({100*goal.mean():.1f}%) | "
      f"cumulative regret {np.sum(OPT-train):,.0f} | last goal episode "
      f"{np.where(goal)[0][-1]:,} at step {steps[np.where(goal)[0][-1]]:,}")
