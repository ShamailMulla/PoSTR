"""
RCA for the v3 learn-then-forget collapse (seed3: ~95% at 40k -> ~8% at 100k).

For each checkpoint it loads the frozen components and refits the BLR head on
that checkpoint's own replay buffer, then reports four readouts that between
them localize WHERE the degradation lives:

  A) VALUE DISCRIMINATION — V(s) over the 25 canonical grid states. A healthy
     value net has high V near the goal that decays backward with discount;
     a collapsed value net is ~constant (the v1/v2 failure locus). Reported as
     std(V) and V at the goal-approach state vs the start state.

  B) WORLD-MODEL ACCURACY — BLR next-state grid distance on a held-out split of
     the buffer. Tells whether the transformer features / transition rotted.

  C) GOAL-REWARD RECALL — BLR predicted reward on the actual goal transitions in
     the buffer. If the reward signal is lost, value bootstrap has nothing to
     propagate even with a perfect value net.

  D) BUFFER GOAL COMPOSITION — fraction of stored episodes that still contain a
     goal (reward>0.5). protected_fraction=0 in v3, so age eviction can purge
     goal data; if this collapses in lockstep with performance, forgetting is
     data-driven, not representational.

Run:  python3 diagnostic_v3_rca.py <run_dir> <ckpt> [<ckpt> ...]
"""
import json
import pickle
import sys

import numpy as np
import torch

from TRL.networks.transition import Network as TN
from TRL.networks.terminal import Network as EN
from TRL.networks.value import Network as VN
from TRL.bayes.neural_linear_head import NeuralLinearHead
from TRL.tuning.common import coerce_numeric

N = 5  # DeepSea-5 grid side


def grid_dist(p, t):
    return np.sqrt(((p // N) - (t // N)) ** 2 + ((p % N) - (t % N)) ** 2)


def load_ckpt(run, ckpt):
    cfg = coerce_numeric(json.load(open(f"{run}/hyper_parameters.txt")))
    d = f"{run}/checkpoints/{ckpt}"
    ctx = cfg["transition"]["context_length"]
    tn = TN(cfg["transition"], 25, [0, 1], p_seq=cfg["transition"]["p_seq"],
            p_attn=cfg["transition"]["p_attn"], max_steps=cfg["experiment"]["time_limit"], device="cpu")
    tn.load_state_dict(torch.load(f"{d}/transition.pt", map_location="cpu")); tn.eval(); tn.unlock_all_dropouts()
    en = EN(25, cfg["terminal"], "cpu"); en.load_state_dict(torch.load(f"{d}/terminal.pt", map_location="cpu")); en.eval()
    vn = VN(25, cfg["value"], ctx, "cpu"); vn.load_state_dict(torch.load(f"{d}/value.pt", map_location="cpu")); vn.eval()
    eps = pickle.load(open(f"{d}/replay.pt", "rb"))
    for ep in eps:
        for k in ep:
            if torch.is_tensor(ep[k]):
                ep[k] = ep[k].cpu()
    return cfg, ctx, tn, en, vn, eps


def diagnose(run, ckpt):
    cfg, ctx, tn, en, vn, eps = load_ckpt(run, ckpt)
    head = NeuralLinearHead(cfg["algorithm"], tn, en, 25, [0, 1],
                            feat_dim=cfg["transition"]["hidden_dim"], device="cpu")

    class DS: pass
    ds = DS(); ds.episodes = eps; ds.goal_episodes = []

    # ---- D) buffer goal composition + state coverage --------------------------
    n_goal_ep = sum(int((ep["rewards"].flatten() > 0.5).any()) for ep in eps)
    frac_goal = n_goal_ep / max(1, len(eps))
    # current-state visitation over the buffer -> coverage (distinct cells) and
    # entropy (how concentrated on the optimal path). Tests the
    # success-collapses-diversity hypothesis.
    cnt = np.zeros(25)
    for ep in eps:
        for s in ep["states"][:, -1].argmax(-1).numpy():
            cnt[int(s)] += 1
    p = cnt / max(1, cnt.sum())
    n_distinct = int((cnt > 0).sum())
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())

    # ---- refit BLR on the checkpoint's own buffer -----------------------------
    head.update_posteriors(ds)

    # ---- A) value discrimination over 25 canonical states ---------------------
    S = torch.eye(25)                                   # one-hot per grid cell
    with torch.no_grad():
        V = vn.predict(S).squeeze().numpy()             # V of each next-state
    Vgrid = V.reshape(N, N)
    v_std = float(V.std())
    v_goal = float(V[24])                               # (4,4) goal cell
    v_start = float(V[0])                               # (0,0) start cell
    # value of the goal-approach next-states (last row) vs first row
    v_lastrow = float(Vgrid[-1].mean()); v_firstrow = float(Vgrid[0].mean())

    # ---- held-out split for B) --------------------------------------------------
    rng = np.random.RandomState(0); idx = rng.permutation(len(eps))
    test = [eps[i] for i in idx[: max(1, len(eps) // 5)]]
    obs, ts, act, tn_true = [], [], [], []
    for ep in test:
        T = ep["states"].shape[0]
        obs.append(ep["states"]); ts.append(ep["timestep"]); act.append(ep["actions"].reshape(T, 1))
        tn_true.append(ep["next_states"].argmax(-1))
    obs = torch.cat(obs).float(); ts = torch.cat(ts).long(); act = torch.cat(act).int()
    tn_true = torch.cat(tn_true).numpy()
    with torch.no_grad():
        X = head._features(ts, obs, act)
        blr_pred = (X @ head.mu)[:, :25].argmax(-1).numpy()
        blr_r = (X @ head.mu)[:, 25].numpy()            # predicted reward (mean)
    blr_gd = float(np.mean([grid_dist(p, t) for p, t in zip(blr_pred, tn_true)]))

    # ---- C) goal-reward recall on real goal transitions -----------------------
    g_obs, g_ts, g_act, g_true_r = [], [], [], []
    for ep in eps:
        m = (ep["rewards"].flatten() > 0.5)
        if m.any():
            T = ep["states"].shape[0]
            g_obs.append(ep["states"][m]); g_ts.append(ep["timestep"][m])
            g_act.append(ep["actions"].reshape(T, 1)[m]); g_true_r.append(ep["rewards"].flatten()[m])
    if g_obs:
        go = torch.cat(g_obs).float(); gt = torch.cat(g_ts).long(); ga = torch.cat(g_act).int()
        with torch.no_grad():
            gx = head._features(gt, go, ga)
            pred_goal_r = (gx @ head.mu)[:, 25].numpy()
        goal_r_pred = float(pred_goal_r.mean()); n_goal_tr = len(pred_goal_r)
    else:
        goal_r_pred = float("nan"); n_goal_tr = 0

    return dict(ckpt=ckpt, n_ep=len(eps), frac_goal=frac_goal, n_goal_ep=n_goal_ep,
                v_std=v_std, v_goal=v_goal, v_start=v_start,
                v_lastrow=v_lastrow, v_firstrow=v_firstrow,
                blr_gd=blr_gd, blr_r_meanpred=float(blr_r.mean()),
                goal_r_pred=goal_r_pred, n_goal_tr=n_goal_tr,
                n_distinct=n_distinct, ent=ent, Vgrid=Vgrid)


if __name__ == "__main__":
    run = sys.argv[1]
    ckpts = sys.argv[2:]
    print(f"\nRUN: {run}\n" + "=" * 92)
    hdr = f"{'ckpt':>8} {'eps':>6} {'goalEp%':>8} {'cover':>6} {'ent':>5} {'V_std':>7} " \
          f"{'V_start':>8} {'V_goal':>8} {'BLR_gd':>7} {'goalRhat':>9}"
    print(hdr); print("-" * 92)
    grids = []
    for c in ckpts:
        r = diagnose(run, c)
        grids.append((c, r["Vgrid"]))
        print(f"{r['ckpt']:>8} {r['n_ep']:>6} {100*r['frac_goal']:>7.1f}% {r['n_distinct']:>4}/25 "
              f"{r['ent']:>5.2f} {r['v_std']:>7.3f} {r['v_start']:>8.3f} {r['v_goal']:>8.3f} "
              f"{r['blr_gd']:>7.3f} {r['goal_r_pred']:>9.3f}")
    print("=" * 92)
    for c, g in grids:
        print(f"\nV(s) grid @ ckpt {c}  (rows=depth 0..4, cols 0..4, goal=bottom-right):")
        for row in g:
            print("   " + " ".join(f"{x:+6.2f}" for x in row))
