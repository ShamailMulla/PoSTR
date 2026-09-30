"""
Task-20 validation of the neural-linear head, on a *learned* checkpoint
(v2.3 plateau, which solved DeepSea-5). Fits the BLR on 80% of that model's
real replay features and tests two properties on the 20% held-out split:

  TEST 1 (gating) — BLR posterior MEAN accuracy vs the deterministic head.
    If the BLR mean's grid-distance ≈ predict_state's, the feature locus is
    sound (a linear head on the penultimate is expressive enough).

  TEST 2 (the point) — posterior self-widening / calibration.
    Sampled-M̂ disagreement (predictive std) should be HIGHER on scarce states
    than on dense states. This is the data-coupled uncertainty dropout lacked
    (dropout gave ~0 spread everywhere = overconfident, per v1/v2 diagnostics).

Run: python3 validate_neural_linear.py [run_dir] [ckpt]
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from TRL.networks.transition import Network as TN
from TRL.networks.terminal import Network as EN
from TRL.bayes.neural_linear_head import NeuralLinearHead
from TRL.tuning.common import coerce_numeric

RUN = sys.argv[1] if len(sys.argv) > 1 else "logdir/5-determinsitic/BTRL-v4_protected_1M/0"
CKPT = sys.argv[2] if len(sys.argv) > 2 else "200000"
N = 5  # DeepSea-5 grid side


def grid_dist(pred_idx, true_idx):
    return np.sqrt(((pred_idx // N) - (true_idx // N)) ** 2 + ((pred_idx % N) - (true_idx % N)) ** 2)


cfg = coerce_numeric(json.load(open(f"{RUN}/hyper_parameters.txt")))
d = f"{RUN}/checkpoints/{CKPT}"
tn = TN(cfg["transition"], 25, [0, 1], p_seq=cfg["transition"]["p_seq"], p_attn=cfg["transition"]["p_attn"],
        max_steps=cfg["experiment"]["time_limit"], device="cpu")
tn.load_state_dict(torch.load(f"{d}/transition.pt", map_location="cpu")); tn.eval(); tn.unlock_all_dropouts()
en = EN(25, cfg["terminal"], "cpu"); en.load_state_dict(torch.load(f"{d}/terminal.pt", map_location="cpu")); en.eval()
episodes = pickle.load(open(f"{d}/replay.pt", "rb"))
for ep in episodes:  # buffer was saved on GPU; validation runs on CPU
    for k in ep:
        if torch.is_tensor(ep[k]):
            ep[k] = ep[k].cpu()

# 80/20 split
rng = np.random.RandomState(0); idx = rng.permutation(len(episodes))
train = [episodes[i] for i in idx[len(episodes) // 5:]]
test = [episodes[i] for i in idx[: len(episodes) // 5]]

head = NeuralLinearHead({}, tn, en, 25, [0, 1], feat_dim=cfg["transition"]["hidden_dim"], device="cpu")


class DS: pass
ds = DS(); ds.episodes = train; ds.goal_episodes = []
head.update_posteriors(ds)

# visitation count per current-state in the TRAIN split (for dense/scarce split)
train_counts = np.zeros(25)
for ep in train:
    for s in ep["states"][:, -1].argmax(-1).numpy():
        train_counts[int(s)] += 1

# assemble held-out (context, timestep, action, true-next, current-state)
obs, ts, act, true_next, cur = [], [], [], [], []
for ep in test:
    T = ep["states"].shape[0]
    obs.append(ep["states"]); ts.append(ep["timestep"]); act.append(ep["actions"].reshape(T, 1))
    true_next.append(ep["next_states"].argmax(-1)); cur.append(ep["states"][:, -1].argmax(-1))
obs = torch.cat(obs).float(); ts = torch.cat(ts).long(); act = torch.cat(act).int()
true_next = torch.cat(true_next).numpy(); cur = torch.cat(cur).numpy()

with torch.no_grad():
    det_logits, _, _ = tn.forward(ts, obs, act)              # deterministic head
    det_pred = det_logits.argmax(-1).numpy()
    X = head._features(ts, obs, act)                          # (M, D)
    blr_mean_logits = (X @ head.mu)[:, :25]                   # BLR posterior MEAN
    blr_mean_pred = blr_mean_logits.argmax(-1).numpy()
    # K sampled hypotheses for predictive spread
    Kp = 30; samp = []
    for _ in range(Kp):
        head.sample(); samp.append((X @ head.w)[:, :25].argmax(-1).numpy())
    samp = np.stack(samp)                                     # (K, M)

det_gd = np.mean([grid_dist(p, t) for p, t in zip(det_pred, true_next)])
blr_gd = np.mean([grid_dist(p, t) for p, t in zip(blr_mean_pred, true_next)])

from scipy.stats import spearmanr
freq = train_counts[cur]
with torch.no_grad():
    epi_var = torch.einsum("md,de,me->m", X, head.cov, X).numpy()          # xᵀΣx
    xnorm = (X * X).sum(1).numpy()                                          # ||x||²
    epi_norm = epi_var / (xnorm + 1e-9)                                     # norm-corrected
rho_raw, _ = spearmanr(freq, epi_var)
rho_norm, _ = spearmanr(freq, epi_norm)

print("=" * 64)
print(f"checkpoint {RUN.split('/')[-2]}/{CKPT} | held-out transitions: {len(true_next)}")
print(f"\nTEST 1  BLR-mean vs deterministic-head next-state grid distance:")
print(f"   deterministic head : {det_gd:.3f}  |  BLR posterior mean : {blr_gd:.3f}")
print(f"   -> {'PASS' if blr_gd <= det_gd * 1.25 + 0.05 else 'FAIL'}")
print(f"\nTEST 2  self-widening (variance vs data density), Spearman ρ (want < 0):")
print(f"   raw   xᵀΣx        : ρ = {rho_raw:+.3f}   (confounded by feature norm)")
print(f"   norm  xᵀΣx/||x||² : ρ = {rho_norm:+.3f}   (coverage effect, norm-removed)")

# TEST 3 (decisive) — the forgetting/exploration scenario: remove ALL goal
# transitions from train, refit, and ask whether the goal-approach transition
# now carries HIGHER epistemic uncertainty than a common (well-covered) one.
train_nogoal = []
for ep in train:
    keep = (ep["rewards"].flatten() <= 0.5)
    if keep.all():
        train_nogoal.append(ep)
ds2 = DS(); ds2.episodes = train_nogoal; ds2.goal_episodes = []
head.update_posteriors(ds2)

def norm_var(state_idx, prev_idx, action):
    o = torch.zeros(1, cfg["transition"]["context_length"], 25)
    o[0, -1, state_idx] = 1
    if prev_idx is not None and cfg["transition"]["context_length"] >= 2:
        o[0, -2, prev_idx] = 1
    row = state_idx // N
    t = torch.tensor([[max(row - 1, 0)] * (cfg["transition"]["context_length"] - 1) + [row]])
    with torch.no_grad():
        x = head._features(t, o, torch.tensor([[action]]))
        return float(torch.einsum("md,de,me->m", x, head.cov, x) / ((x * x).sum() + 1e-9))

goal_unc = norm_var(18, 12, 1)     # (3,3)-right-> goal region (removed from train)
common_unc = norm_var(0, None, 0)  # (0,0)-left, the most common transition
print(f"\nTEST 3  goal-removed refit — norm epistemic variance:")
print(f"   goal-approach (18→,removed) : {goal_unc:.4g}")
print(f"   common        (0→,frequent) : {common_unc:.4g}")
print(f"   -> {'PASS' if goal_unc > common_unc else 'FAIL'} "
      f"(goal region should be MORE uncertain once its data is gone)")
