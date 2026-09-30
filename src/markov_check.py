"""Markov check: does the transformer's prediction depend on history?

    python markov_check.py [--seeds 1 2 3 4 5] [--steps 100000 500000 1000000]

The model's context is [s_{t-1}, s_t] with timesteps [t-1, t]; the candidate action is
broadcast over the context, so the only history it sees is the previous state. For every
DeepSea-5 cell at step t reachable from two different previous cells (written for the
pre-2026-09 non-randomized runs; RIGHT = 1 assumes randomize_actions=False), feed both histories
with the same (s_t, t, a) and compare what the agent would act on:

  * BLR posterior mean W = mu (refit from the checkpoint's replay buffer, as in training)
    - argmax next state: does it change with history?
    - reward: size of the history-induced change vs the right-left reward margin
    - one-step greedy action (r_hat only, V is inert): does it flip?
  * 20 posterior samples of W: same argmax / action-flip rates
  * features phi: relative distance between the two histories

See wiki/planning-in-sampled-model.md (section 11, item 4).
"""
import argparse
import glob
import json
import os
import sys
import tempfile

import numpy as np
import torch

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
N = 5          # DeepSea size
RIGHT = 1      # randomize_actions=False: action 1 moves right everywhere


def onehot(row, col):
    v = torch.zeros(N * N)
    v[row * N + col] = 1.0
    return v


def histories():
    """(t, s_t, [prev_1, prev_2]) for every cell with >1 valid predecessor.
    Step t (1-indexed) is spent in row t-1; a predecessor sits in row t-2."""
    cases = []
    for t in range(3, N + 1):              # row 0 has one cell; row 1 cells have one parent
        row = t - 1
        for col in range(0, row + 1):
            prevs = {c for c in (col - 1, col + 1) if 0 <= c <= row - 1}
            if col == 0:
                prevs.add(0)               # "left" at col 0 stays at col 0
            if len(prevs) > 1:
                cases.append((t, (row, col), sorted(prevs)))
    return cases


def build_agent(run_dir, ckpt):
    from TRL.agent import Agent
    from TRL.common.checkpoint import load_checkpoint
    from TRL.common.data_manager import DataManager
    from TRL.common.logger import Logger
    from TRL.common.utils import init_env

    cfg = json.load(open(os.path.join(run_dir, "hyper_parameters.txt")))
    cfg["load"] = True                      # reuse the dir name; no new numbered logdir
    cwd = os.getcwd()
    os.chdir(tempfile.mkdtemp(prefix="markov_"))   # DataManager writes tensorboard files
    try:
        dm = DataManager(cfg)
    finally:
        os.chdir(cwd)
    _, actions, _, obs_space = init_env(cfg["experiment"]["suite"], cfg["experiment"]["env"],
                                        False, deterministic=cfg["experiment"]["deterministic"])
    agent = Agent(cfg, actions, Logger(dm), obs_space, cfg["experiment"]["seed"])
    load_checkpoint(agent, ckpt)
    agent.model.head.update_posteriors(agent.dataset)   # mu, Sigma from this buffer
    return agent


@torch.no_grad()
def check(agent, n_samples=20):
    head, dev = agent.model.head, agent.device
    rows = []
    for t, (row, col), prevs in histories():
        s = onehot(row, col)
        ctx = torch.stack([torch.stack([onehot(row - 1, p), s]) for p in prevs]).to(dev)   # (H,2,25)
        ts = torch.tensor([[t - 1, t]] * len(prevs), device=dev)
        per_action = {}
        for a in agent.actions:
            act = torch.full((len(prevs), 1), a, dtype=torch.long, device=dev)
            X = head._features(ts, ctx, act)                                              # (H, D)
            per_action[int(a)] = X
        rows.append((t, row, col, prevs, per_action))

    def evaluate(W):
        n_state_diff = n_flip = 0
        rew_gap, margins = [], []
        for t, row, col, prevs, feats in rows:
            preds = {a: X @ W for a, X in feats.items()}                                  # (H, 26)
            for a, P in preds.items():
                n_state_diff += int(len(set(P[:, :N * N].argmax(1).tolist())) > 1)
                rew_gap.append((P[:, -1].max() - P[:, -1].min()).item())
            margin = preds[RIGHT][:, -1] - preds[1 - RIGHT][:, -1]                          # per history
            margins.append(margin.abs().min().item())
            n_flip += int(len(set((margin > 0).tolist())) > 1)
        n_sa = len(rows) * len(agent.actions)
        return dict(state_argmax_changes=n_state_diff / n_sa, greedy_action_flips=n_flip / len(rows),
                    reward_gap_median=float(np.median(rew_gap)), reward_gap_max=float(np.max(rew_gap)),
                    right_left_margin_median=float(np.median(margins)))

    out = {"cases": len(rows), "mean_W": evaluate(head.mu)}
    samp = []
    for _ in range(n_samples):
        head.sample(explore=False)
        samp.append(evaluate(head.w))
    out["samples"] = {k: float(np.mean([s[k] for s in samp])) for k in samp[0]}
    dists = []
    for _, _, _, _, feats in rows:
        for X in feats.values():
            phi = X[:, :-1]
            dists.append(((phi[0] - phi[1]).norm() / phi.norm(dim=1).mean().clamp_min(1e-8)).item())
    out["feature_rel_distance_median"] = float(np.median(dists))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--steps", type=int, nargs="+", default=[100_000, 500_000, 1_000_000])
    a = ap.parse_args()
    torch.manual_seed(0)
    results = {}
    for s in a.seeds:
        run_dir = sorted(glob.glob(f"{SRC}/logdir/*/BTRL-postr_det_seed{s}/0"))[0]
        for step in a.steps:
            ckpt = f"{run_dir}/checkpoints/{step}/resume.pt"
            if not os.path.exists(ckpt):
                print(f"seed{s} {step}: no checkpoint"); continue
            r = check(build_agent(run_dir, ckpt))
            results[f"seed{s}_{step}"] = r
            m, sm = r["mean_W"], r["samples"]
            print(f"seed{s} {step:>8,} | mean W: next-state changes {m['state_argmax_changes']:.0%}, "
                  f"action flips {m['greedy_action_flips']:.0%}, reward gap med {m['reward_gap_median']:.4f} "
                  f"(margin med {m['right_left_margin_median']:.4f}) | samples: next-state {sm['state_argmax_changes']:.0%}, "
                  f"flips {sm['greedy_action_flips']:.0%} | phi rel dist {r['feature_rel_distance_median']:.3f}",
                  flush=True)
    json.dump(results, open(os.path.join(SRC, "runlogs", "markov_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
