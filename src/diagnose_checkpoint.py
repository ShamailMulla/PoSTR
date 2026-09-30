"""
Post-mortem diagnostics for a BTRL checkpoint (DeepSea-5).

For each checkpoint:
  1. Replay buffer — episode count, reward stats, goal transitions present?
  2. Value network — V(s) over all 25 one-hot states, printed as the 5x5 grid.
  3. Transition reward head + posterior diversity — for every reachable
     (state, action): predicted next state and reward across K sampled
     hypotheses M-hat; spread across hypotheses = posterior diversity.
  4. Greedy Thompson rollouts — 20 episodes in the real env with a fresh
     M-hat each; goal-rate and the column reached per episode.

Usage: python3 diagnose_checkpoint.py <checkpoint_dir> [K_hypotheses]
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

from TRL.agent.btrl import BTRL
from TRL.common.logger import Logger
from TRL.common.utils import init_env, env_reset, env_step, load
from TRL.tuning.agent_eval import NullDataManager
from TRL.tuning.common import load_base_config

ckpt_dir = sys.argv[1].rstrip("/") + "/"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 50

config = load_base_config(env="5", deterministic=True)
N = 5
OBS = 25

env, actions, _, obs_space = init_env("bsuite", "5", test=False, deterministic=True)
agent = BTRL(config, actions, Logger(NullDataManager()), obs_space, seed=0)
agent = load(agent, ckpt_dir)
device = agent.device

print("=" * 70)
print("CHECKPOINT:", ckpt_dir)

# ── 1. Replay buffer ─────────────────────────────────────────────────────────
eps = agent.dataset.episodes
all_rewards = torch.cat([ep["rewards"] for ep in eps]).flatten()
goal_transitions = int((all_rewards > 0.5).sum())  # tanh(0.99) ≈ 0.757
next_idx = torch.cat([ep["next_states"] for ep in eps]).argmax(-1).cpu().numpy()
visit_counts = np.bincount(next_idx, minlength=OBS).reshape(N, N)
acts = torch.cat([ep["actions"] for ep in eps]).flatten().cpu().numpy()

print(f"\n[1] REPLAY BUFFER: {len(eps)} episodes, {len(all_rewards)} transitions")
print(f"    goal transitions (reward>0.5): {goal_transitions}")
print(f"    reward range: [{all_rewards.min():.3f}, {all_rewards.max():.3f}]")
print(f"    action counts: {np.bincount(acts, minlength=2).tolist()}")
print("    next-state visitation (5x5 grid, goal = bottom-right):")
for row in visit_counts:
    print("      " + " ".join(f"{c:5d}" for c in row))

# ── 2. Value network on one-hot states ──────────────────────────────────────
with torch.no_grad():
    V = agent.value_network.predict(torch.eye(OBS, device=device)).cpu().numpy().reshape(N, N)
print("\n[2] VALUE NETWORK V(s) on one-hot states (5x5 grid):")
for row in V:
    print("      " + " ".join(f"{v:8.4f}" for v in row))
print(f"    V(goal-adjacent s=18): {V[3,3]:.4f}   V(goal s=24): {V[4,4]:.4f}")
print(f"    value spread across states: {V.max()-V.min():.4f}")

# ── 3. Posterior diversity + reward head over reachable states ──────────────
# Reachable states in DeepSea: row r, col c<=r. Build a 1-step context per
# state the same way the live agent does (context = [0, state]).
print(f"\n[3] TRANSITION MODEL — {K} sampled hypotheses per (state, action)")
print("    state (row,col) | action | pred next (mode, agreement%) | mean pred reward")
interesting = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (3, 0), (4, 0)]
report = {}
for r, c in interesting:
    s = r * N + c
    onehot = np.zeros(OBS, dtype=np.float32); onehot[s] = 1.0
    preds = {0: [], 1: []}; rews = {0: [], 1: []}
    for _ in range(K):
        agent.model.sample_model()
        # reset context exactly like a fresh episode, then feed this state
        next_states, rewards, terminals = agent.model.predict(1, onehot)
        idx = next_states.argmax(-1).cpu().numpy().flatten()
        rw = rewards.detach().cpu().numpy().flatten()
        for a in (0, 1):
            preds[a].append(int(idx[a])); rews[a].append(float(rw[a]))
    for a in (0, 1):
        vals, counts = np.unique(preds[a], return_counts=True)
        mode = int(vals[counts.argmax()]); agree = 100 * counts.max() / K
        report[(s, a)] = (mode, agree, np.mean(rews[a]))
        print(f"    s={s:2d} ({r},{c})     |   {a}    | {mode:2d} ({mode//N},{mode%N})  {agree:5.1f}%"
              f"        | {np.mean(rews[a]):+.4f}")
mode18, agree18, rew18 = report[(18, 1)]
print(f"    >>> critical transition s=18 -a=1-> should be 24 with reward ~0.75(tanh):")
print(f"        predicted {mode18} with {agree18:.0f}% agreement, reward {rew18:+.4f}")

# ── 4. Greedy Thompson rollouts in the real environment ─────────────────────
print("\n[4] GREEDY THOMPSON ROLLOUTS (20 episodes, fresh M-hat each):")
returns, final_cols = [], []
for _ in range(20):
    obs = env_reset(env, "bsuite")
    step, ret, done = 1, 0.0, False
    last_state = int(np.argmax(obs))
    while not done:
        action, _, _ = agent.select_action(obs, step)
        obs, rew, done = env_step(env, "bsuite", action)
        if np.max(obs) > 0:
            last_state = int(np.argmax(obs))
        ret += rew
        done = done or step == config["experiment"]["time_limit"]
        step += 1
    returns.append(ret)
    final_cols.append(last_state % N)
goal_hits = sum(1 for r in returns if r > 0.5)
print(f"    goal-rate: {goal_hits}/20 | mean return {np.mean(returns):+.4f}")
print(f"    final column reached per episode: {final_cols}")
print(f"    (column 4 = goal side; column 0 = agent always moved left)")
