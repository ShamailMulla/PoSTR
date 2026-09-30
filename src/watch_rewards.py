"""
Live reward monitor for a running BTRL training run.

Reads metrics.jsonl every REFRESH_SECS seconds, plots test and train rewards,
and saves/overwrites a PNG at the same path.

Usage:
    python3 watch_rewards.py [metrics.jsonl path]  [--interval N]

If no path is given, auto-discovers the most recent run under logdir/.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")          # no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


REFRESH_SECS   = 30
MAX_TRAIN_PTS  = 500   # down-sample train curve to keep the plot readable


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def find_latest_metrics(base="logdir"):
    """Walk logdir tree and return the most recently modified metrics.jsonl."""
    candidates = glob.glob(os.path.join(base, "**", "metrics.jsonl"), recursive=True)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_metrics(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def split_series(rows):
    """Return (train_steps, train_rewards, test_steps, test_rewards)."""
    train_steps, train_rewards = [], []
    test_steps,  test_rewards  = [], []

    for r in rows:
        step = r.get("Timestep")
        if step is None:
            continue

        tr = r.get("Reward/Train_Reward")
        te = r.get("Reward/Test_Reward")

        if tr is not None and not math.isnan(tr):
            train_steps.append(step)
            train_rewards.append(tr)

        if te is not None and not math.isnan(te):
            test_steps.append(step)
            test_rewards.append(te)

    return (np.array(train_steps), np.array(train_rewards),
            np.array(test_steps),  np.array(test_rewards))


def smooth(x, window=10):
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def build_plot(metrics_path, out_path):
    rows = load_metrics(metrics_path)
    if not rows:
        return

    tr_steps, tr_rews, te_steps, te_rews = split_series(rows)
    current_step = rows[-1].get("Timestep", 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"BTRL — DeepSea 5×5 deterministic   (step {current_step:,} / 1,000,000)",
        fontsize=13, fontweight="bold"
    )

    # --- left: test reward ---
    ax = axes[0]
    if len(te_steps) > 0:
        ax.plot(te_steps, te_rews, "o-", color="crimson", lw=2, ms=6, label="Test reward")
        ax.axhline(1.0, color="green",  linestyle="--", alpha=0.6, label="Goal (reward = 1)")
        ax.axhline(0.0, color="grey",   linestyle=":",  alpha=0.4)
        # running best
        best = np.maximum.accumulate(te_rews)
        ax.fill_between(te_steps, te_rews, best, alpha=0.12, color="crimson")
        ax.annotate(
            f"Latest: {te_rews[-1]:.3f}",
            xy=(te_steps[-1], te_rews[-1]),
            xytext=(10, 8), textcoords="offset points",
            fontsize=9, color="crimson"
        )
    else:
        ax.text(0.5, 0.5, "Waiting for first\ntest episode (step 5000)…",
                ha="center", va="center", transform=ax.transAxes, fontsize=11, color="grey")

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode return")
    ax.set_title("Test reward (greedy rollout, every 5 000 steps)")
    ax.set_xlim(left=0)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # --- right: train reward + smoothed ---
    ax2 = axes[1]
    if len(tr_steps) > 0:
        # down-sample for readability
        stride = max(1, len(tr_steps) // MAX_TRAIN_PTS)
        ds_steps = tr_steps[::stride]
        ds_rews  = tr_rews[::stride]
        ax2.plot(ds_steps, ds_rews, alpha=0.25, color="steelblue", lw=0.8, label="Train (raw)")

        # smoothed
        sm = smooth(tr_rews, window=min(50, len(tr_rews) // 5 + 1))
        sm_steps = tr_steps[len(tr_rews) - len(sm):]
        ax2.plot(sm_steps, sm, color="steelblue", lw=2, label="Train (smoothed)")

    if len(te_steps) > 0:
        ax2.plot(te_steps, te_rews, "o-", color="crimson", lw=1.5, ms=5, label="Test reward")

    ax2.axhline(1.0, color="green", linestyle="--", alpha=0.6, label="Goal")
    ax2.axhline(0.0, color="grey",  linestyle=":",  alpha=0.4)
    ax2.set_xlabel("Environment steps")
    ax2.set_ylabel("Episode return")
    ax2.set_title("Train reward (per episode) + test overlay")
    ax2.set_xlim(left=0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    # progress bar across bottom
    prog = current_step / 1_000_000
    fig.text(0.15, 0.01, f"Progress: {100*prog:.1f}%  ({current_step:,} / 1,000,000 steps)",
             fontsize=9, color="dimgray")
    bar_ax = fig.add_axes([0.15, 0.0, 0.7, 0.012])
    bar_ax.barh(0, prog, color="steelblue", height=1)
    bar_ax.barh(0, 1 - prog, left=prog, color="#e0e0e0", height=1)
    bar_ax.set_xlim(0, 1)
    bar_ax.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return current_step


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="?", default=None,
                        help="Path to metrics.jsonl (auto-discovers if omitted)")
    parser.add_argument("--interval", type=int, default=REFRESH_SECS,
                        help="Refresh interval in seconds (default: 30)")
    args = parser.parse_args()

    metrics_path = args.metrics or find_latest_metrics()
    if metrics_path is None:
        print("No metrics.jsonl found under logdir/. Pass the path explicitly.")
        sys.exit(1)

    out_path = os.path.join(os.path.dirname(metrics_path), "reward_curve_live.png")
    print(f"Monitoring: {metrics_path}")
    print(f"Output PNG: {out_path}")
    print(f"Refresh interval: {args.interval}s   (Ctrl-C to stop)\n")

    iteration = 0
    while True:
        step = build_plot(metrics_path, out_path)
        ts = time.strftime("%H:%M:%S")
        if step is not None:
            print(f"[{ts}]  iteration {iteration:4d}  —  step {step:>9,}  →  {out_path}")
        else:
            print(f"[{ts}]  metrics empty, retrying…")
        iteration += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
