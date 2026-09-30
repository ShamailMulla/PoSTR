"""
Action-value ("did it learn to go right?") visualisation for a BTRL run.

Replicates exactly the score BTRL.select_action computes for each action:
    score(a) = r_hat(a) + gamma * V(s'_hat(a)) * (terminal_hat(a) < TP_THRESHOLD)
using the run's final transition + terminal + value checkpoints. For each
reachable state we show score(LEFT=action0) vs score(RIGHT=action1) and their
margin. On DeepSea (randomize_actions=False so action 1 = right) a solved agent
should prefer RIGHT on the descent, i.e. margin = score_right - score_left > 0.

Dropout OFF (posterior mean) = the consolidated greedy preference.

Run: python3 visualize_action_values.py [run_dir] [name]
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
from TRL.networks.terminal import Network as TerminalNetwork
from TRL.networks.value import Network as ValueNetwork
from TRL.common.settings import TP_THRESHOLD
from TRL.tuning.common import coerce_numeric

PLOTS = Path(__file__).resolve().parents[1] / "plots" / "attention"
PLOTS.mkdir(parents=True, exist_ok=True)
RUN = sys.argv[1] if len(sys.argv) > 1 else "logdir/5-determinsitic/BTRL-best_combined_1M/0"
NAME = sys.argv[2] if len(sys.argv) > 2 else "v2.2_HPO"


def load(run_dir, ckpt_step=None):
    cfg = coerce_numeric(json.load(open(Path(run_dir) / "hyper_parameters.txt")))
    avail = sorted(int(p.name) for p in (Path(run_dir) / "checkpoints").iterdir())
    ck = ckpt_step if (ckpt_step in avail) else avail[-1]
    d = Path(run_dir) / "checkpoints" / str(ck)
    env = int(cfg["experiment"]["env"]); obs = env * env
    ctx = cfg["transition"]["context_length"]
    tn = TransitionNetwork(cfg["transition"], obs, [0, 1], p_seq=cfg["transition"]["p_seq"],
                           p_attn=cfg["transition"]["p_attn"], max_steps=cfg["experiment"]["time_limit"], device="cpu")
    tn.load_state_dict(torch.load(d / "transition.pt", map_location="cpu"))
    en = TerminalNetwork(obs, cfg["terminal"], "cpu")
    en.load_state_dict(torch.load(d / "terminal.pt", map_location="cpu"))
    vn = ValueNetwork(obs, cfg["value"], ctx, "cpu")
    vn.load_state_dict(torch.load(d / "value.pt", map_location="cpu"))
    for m in (tn, en, vn):
        m.eval()
    tn.unlock_all_dropouts()
    return tn, en, vn, cfg, env, ctx, ck


def action_scores(tn, en, vn, cfg, env, ctx, state, prev):
    """score for [left, right] at `state`, replicating select_action's value calc."""
    obs = torch.zeros(2, ctx, env * env)
    obs[:, -1, state] = 1
    if prev is not None and ctx >= 2:
        obs[:, -2, prev] = 1
    row = state // env
    ts = torch.tensor([[max(row - 1, 0)] * (ctx - 1) + [row]] * 2)
    act = torch.tensor([[0], [1]])
    with torch.no_grad():
        ns, rtg, _ = tn.forward(ts, obs, act)
        rewards = rtg.squeeze(-1)
        terminals = en.predict(ns).squeeze(-1)
        vals = vn.predict(ns).squeeze(-1)
        disc = float(cfg["value"]["discount"])
        scores = rewards + disc * vals * (terminals < TP_THRESHOLD)
    return scores.numpy()  # [left, right]


def visualize(name, run_dir, ckpt_step=None):
    tn, en, vn, cfg, env, ctx, ck = load(run_dir, ckpt_step)
    diag = [r * env + r for r in range(env)]

    left, right = [], []
    for r, s in enumerate(diag):
        sc = action_scores(tn, en, vn, cfg, env, ctx, s, diag[r - 1] if r > 0 else None)
        left.append(sc[0]); right.append(sc[1])
    left, right = np.array(left), np.array(right)
    margin = right - left

    # also a full-grid margin map over all reachable states (row r, col<=r)
    grid = np.full((env, env), np.nan)
    for r in range(env):
        for c in range(r + 1):
            s = r * env + c
            prev = (r - 1) * env + max(c - 1, 0) if r > 0 else None
            sc = action_scores(tn, en, vn, cfg, env, ctx, s, prev)
            grid[r, c] = sc[1] - sc[0]

    fig, (axb, axg) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(env)
    axb.bar(x - 0.2, left, 0.4, label="LEFT (action 0)", color="tab:red", alpha=0.8)
    axb.bar(x + 0.2, right, 0.4, label="RIGHT (action 1)", color="tab:green", alpha=0.8)
    axb.set_xticks(x); axb.set_xticklabels([f"row{r}" + ("\nGOAL-adj" if r == env - 1 else "") for r in x])
    axb.set_ylabel("action score  r̂ + γV(ŝ')"); axb.set_xlabel("agent position on descent")
    axb.set_title("action scores along the diagonal — RIGHT should win"); axb.legend()
    axb.axhline(0, color="grey", lw=0.6)
    for r in range(env):
        pref = "→" if margin[r] > 0 else "←"
        axb.text(r, max(left[r], right[r]) + 0.01, pref, ha="center",
                 color="green" if margin[r] > 0 else "red", fontsize=13, fontweight="bold")

    im = axg.imshow(grid, cmap="RdYlGn", vmin=-np.nanmax(abs(grid)), vmax=np.nanmax(abs(grid)))
    axg.set_title("margin = score(RIGHT) − score(LEFT)\ngreen = prefers right (good)")
    axg.set_xticks(range(env)); axg.set_yticks(range(env))
    axg.set_xlabel("col"); axg.set_ylabel("row")
    for r in range(env):
        for c in range(r + 1):
            axg.text(c, r, f"{grid[r,c]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axg, fraction=0.046)
    fig.suptitle(f"{name} — learned action preference @ {ck:,} steps (DeepSea-{env}, action1=right)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = PLOTS / f"actionval_{name}.png"
    for ext in ("png", "pdf"):
        fig.savefig(str(out).replace(".png", f".{ext}"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    frac_right = np.mean(margin > 0)
    print(f"saved {out}  | diagonal prefers RIGHT at {frac_right:.0%} of positions | margins={np.round(margin,3)}")


if __name__ == "__main__":
    ck = int(sys.argv[3]) if len(sys.argv) > 3 else None
    visualize(NAME, RUN, ck)
