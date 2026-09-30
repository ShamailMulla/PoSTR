"""Reward-signal check: does the BLR reward model still know where the treasure is?

    python reward_check.py [--seeds 1 2 3 4 5] [--steps 100000 300000 500000 700000 1000000]

Task as the pre-2026-09 BTRL runs trained it (now `legacy_goal: true`, `randomize_actions: false`): bsuite DeepSea-5 with randomize_actions=False, plus the
override in TRL/common/utils.env_step - reaching the bottom-right cell (row 4, col 4)
ends the episode with reward 0.99 (stored as tanh(0.99) = 0.757). So the treasure is
paid on the step-4 move right from (3,3), context [(2,2), (3,3)], timesteps [3, 4].
For each checkpoint, with the BLR posterior refit from its replay buffer (as in training):

  * goal transitions in the buffer: how many, and r_hat (posterior mean) on them
  * goal margin: r_hat(goal cell, right) - r_hat(goal cell, left), on the real goal
    contexts, under the mean W and under 20 posterior samples (P[margin > 0.5], P[> 0])
  * non-goal transitions: r_hat for right / left, and the reward R^2 over the buffer
  * canonical goal query: prev (2,2) -> cur (3,3), steps [3, 4], action right vs left

See wiki/planning-in-sampled-model.md (section 11).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import markov_check as mc  # noqa: E402  (build_agent, onehot)

RIGHT, LEFT = 1, 0


@torch.no_grad()
def buffer_arrays(agent):
    """Contexts, timesteps, actions and rewards of every buffered transition."""
    from TRL.common.utils import extract_episode_data
    ds = agent.dataset
    obs, ts, act, rew = [], [], [], []
    for ep in ds.episodes + getattr(ds, "goal_episodes", []):
        o, a, _, r, _, t = extract_episode_data([ep])
        T = o.shape[1]
        obs.append(o[0]); ts.append(t[0]); act.append(a[0].reshape(T, 1)); rew.append(r[0].reshape(T))
    dev = agent.device
    return (torch.cat(obs).to(dev), torch.cat(ts).to(dev),
            torch.cat(act).long().to(dev), torch.cat(rew).float().to(dev))


@torch.no_grad()
def features(head, ts, obs, act, chunk=512):
    return torch.cat([head._features(ts[i:i + chunk], obs[i:i + chunk], act[i:i + chunk])
                      for i in range(0, len(obs), chunk)])


@torch.no_grad()
def check(agent, n_samples=20):
    head, dev = agent.model.head, agent.device
    obs, ts, act, rew = buffer_arrays(agent)
    X = features(head, ts, obs, act)
    X_flip = features(head, ts, obs, 1 - act)            # same context, other action
    r_hat = (X @ head.mu)[:, -1]
    goal = rew > 0.5
    out = {"transitions": len(rew), "goal_transitions": int(goal.sum()),
           "reward_r2": float(1 - ((r_hat - rew) ** 2).mean() / rew.var().clamp_min(1e-12))}
    nongoal = ~goal
    for name, a in (("right", RIGHT), ("left", LEFT)):
        m = nongoal & (act[:, 0] == a)
        out[f"r_hat_nongoal_{name}"] = float(r_hat[m].mean()) if m.any() else None

    # canonical goal query (independent of whether the buffer holds goal transitions)
    ctx = torch.stack([mc.onehot(2, 2), mc.onehot(3, 3)]).unsqueeze(0).repeat(2, 1, 1).to(dev)
    tq = torch.tensor([[3, 4], [3, 4]], device=dev)
    aq = torch.tensor([[RIGHT], [LEFT]], device=dev)
    Xq = head._features(tq, ctx, aq)

    def margins(W):
        rq = (Xq @ W)[:, -1]
        res = {"canon_right": rq[0].item(), "canon_left": rq[1].item(), "canon_margin": (rq[0] - rq[1]).item()}
        if goal.any():   # real goal contexts: right (taken) vs left (counterfactual)
            g_right, g_left = (X[goal] @ W)[:, -1], (X_flip[goal] @ W)[:, -1]
            res.update(goal_r_hat=g_right.mean().item(), goal_margin=(g_right - g_left).mean().item())
        return res

    out["mean_W"] = margins(head.mu)
    samp = []
    for _ in range(n_samples):
        head.sample(explore=False)
        samp.append(margins(head.w))
    key = "goal_margin" if goal.any() else "canon_margin"
    vals = np.array([s[key] for s in samp])
    out["samples"] = {"margin_key": key, "P_margin_gt_0.5": float((vals > 0.5).mean()),
                      "P_margin_gt_0": float((vals > 0).mean()), "margin_mean": float(vals.mean()),
                      "margin_std": float(vals.std())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--steps", type=int, nargs="+", default=[100_000, 300_000, 500_000, 700_000, 1_000_000])
    a = ap.parse_args()
    torch.manual_seed(0)
    results = {}
    for s in a.seeds:
        run_dir = sorted(glob.glob(f"{SRC}/logdir/*/BTRL-postr_det_seed{s}/0"))[0]
        for step in a.steps:
            ckpt = f"{run_dir}/checkpoints/{step}/resume.pt"
            if not os.path.exists(ckpt):
                print(f"seed{s} {step}: no checkpoint"); continue
            r = check(mc.build_agent(run_dir, ckpt))
            results[f"seed{s}_{step}"] = r
            m, sm = r["mean_W"], r["samples"]
            goal_txt = (f"goal r_hat {m['goal_r_hat']:.3f} margin {m['goal_margin']:+.3f}"
                        if "goal_margin" in m else "no goal transitions in buffer")
            print(f"seed{s} {step:>8,} | goals {r['goal_transitions']:>5}/{r['transitions']} | {goal_txt} | "
                  f"canonical margin {m['canon_margin']:+.3f} (right {m['canon_right']:.3f}) | "
                  f"samples P[margin>0.5] {sm['P_margin_gt_0.5']:.0%} P[>0] {sm['P_margin_gt_0']:.0%} | "
                  f"non-goal r_hat R {r['r_hat_nongoal_right']:+.3f} L {r['r_hat_nongoal_left']:+.3f} | "
                  f"reward R2 {r['reward_r2']:.2f}", flush=True)
    json.dump(results, open(os.path.join(SRC, "runlogs", "reward_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
