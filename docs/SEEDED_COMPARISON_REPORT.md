# PSDRL vs BTRL — 5-Seed Comparison on DeepSea-5 (deterministic)

**Date:** 2026-07-16
**Runs:** `BRL/src/logdir/deepsea-5/PSDRL-deepsea5_ctrl/{0..4}` (seeds 1–5) and
`BTRL/src/logdir/5-determinsitic/BTRL-postr_det_seed{1..5}/0`
**Budget:** 1,000,000 environment steps per run; all 10 runs completed.
**Plot:** `plots/seeded_comparison/psdrl_vs_btrl_solve_rate.png` (rolling 1000-episode solve rate per seed + mean).

An episode counts as "solved" if its return exceeds 0.9 (treasure reached; optimal return ≈ 0.99 for PSDRL, 0.984 observed for BTRL due to its shorter time limit of 6 vs 25).

## Per-seed results

| Agent | Seed | Episodes | Solve rate (all) | First solve (step) | Solve rate (last 100k steps) | Cumulative reward | Test solve rate |
|-------|------|---------:|-----------------:|-------------------:|-----------------------------:|------------------:|----------------:|
| PSDRL | 1 | 200,000 | 0.0% | 140 | 0.0% | −108 | 0.0% |
| PSDRL | 2 | 200,000 | 0.0% | 220 | 0.0% | −91 | 0.8% |
| PSDRL | 3 | 200,000 | **88.1%** | 5 | **88.1%** | **174,351** | 3.2% |
| PSDRL | 4 | 200,000 | 0.0% | 11,155 | 0.1% | −17 | 0.0% |
| PSDRL | 5 | 200,000 | 0.0% | 7,035 | 0.0% | −92 | 0.0% |
| BTRL | 1 | 240,800 | 84.7% | 15 | 12.7% | 200,596 | 82.1% |
| BTRL | 2 | 209,852 | 23.5% | 10 | 11.4% | 47,747 | 21.9% |
| BTRL | 3 | 205,127 | 12.5% | 75 | 12.1% | 24,446 | 12.3% |
| BTRL | 4 | 205,686 | 13.8% | 5 | 11.2% | 27,178 | 13.4% |
| BTRL | 5 | 209,230 | 22.1% | 10 | 12.3% | 44,706 | 20.6% |

**Aggregates:** mean cumulative reward — BTRL 68,935 (median 44,706) vs PSDRL 34,809 (median **−91**). Mean overall solve rate — BTRL 31.3% vs PSDRL 17.6%. Mean solve rate over the final 100k steps — PSDRL 17.6% vs BTRL 12.0%.

## Findings

1. **BTRL explores reliably; PSDRL does not.** Every BTRL seed found the treasure within its first ~15 episodes (worst seed: step 75) and every seed accumulated substantial positive reward. PSDRL is bimodal: seed 3 locked onto the optimal policy immediately and held an 88% solve rate for the entire million steps, while the other four seeds — despite each stumbling onto the treasure at least once — never learned from it and finished with *negative* cumulative reward. On median performance BTRL dominates (44,706 vs −91); PSDRL's mean is carried entirely by one lucky seed.

2. **BTRL fails to retain the solution.** This is the headline problem in these runs. Seed 1 held a ~99% solve rate for 800k steps and then collapsed to ~12% at around step 813k. Seeds 2 and 5 show early high-performance phases (53% and 45% solve rate in early fifths of training) that decay; seeds 3 and 4 decay almost immediately after their early discovery. By the last 100k steps, *all five* BTRL seeds converge to a suspiciously uniform 11–13% solve-rate floor — well above random (~3% for DeepSea-5) but far below solved. Whatever knocks the policy off the optimum, it acts on every seed and drives them to the same steady state.

3. **Train/test consistency differs sharply.** BTRL's test solve rate tracks its train solve rate almost exactly (e.g., seed 1: 84.7% train vs 82.1% test). PSDRL seed 3 solves 88% of training episodes but only 3.2% of test episodes — worth checking the PSDRL test-episode path (`BRL/src/main.py: run_test_episode`) before quoting its test numbers; the agent likely re-samples a model or applies policy noise (`policy_noise: 0.05`) at test time.

4. **Solve-rate trajectory by fifths of training** (fraction of episodes solved per fifth):
   - PSDRL: seed 3 flat at 0.88 throughout; all other seeds flat at 0.00.
   - BTRL seed 1: 0.97 / 0.99 / 0.99 / 0.99 / **0.31** (late collapse)
   - BTRL seed 2: 0.53 / 0.22 / 0.18 / 0.13 / 0.11 (steady decay)
   - BTRL seed 3: 0.15 / 0.12 / 0.12 / 0.11 / 0.12 (early loss, never recovered)
   - BTRL seed 4: 0.26 / 0.11 / 0.11 / 0.11 / 0.11 (early loss)
   - BTRL seed 5: 0.28 / 0.45 / 0.11 / 0.13 / 0.13 (partial recovery, then loss)

## Interpretation for the thesis

The runs support the deep-exploration claim: BayesFormer posterior sampling finds the sparse reward in ≤75 steps on every seed, where PSDRL's linear Bayesian model needs luck (1/5 seeds). They do **not** yet support a sample-efficiency claim in steady state, because BTRL exhibits a retention failure — the policy drifts off the optimum and settles at an ~12% solve-rate floor. Plausible suspects to investigate next, roughly in order:

- **Replay eviction:** `capacity: 10000` (~2,100 episodes) with `protected_fraction: 0.0` — once the buffer is dominated by post-discovery episodes with degraded returns, successful trajectories can be evicted entirely (PSDRL seed 3 never collapses with the same capacity, but its buffer stays ~88% successful episodes, so eviction is benign there). The seed-1 collapse at ~813k and the shared 11–13% floor are both consistent with a rare-success/eviction equilibrium.
- **Value re-fitting under a drifting sampled model** (`temp: 0.0001`, `explore_temp: 4.0`, neural-linear head): if the exploratory posterior temperature keeps injecting optimism after the task is solved, the greedy policy can be pulled off the optimal column.
- Compare against the earlier `BTRL-v4_protected_seed*` runs (which used a non-zero protected fraction) to isolate the eviction hypothesis.

## Attention analysis (added 2026-07-16)

Figures in `BTRL/plots/attention/`, generated with the existing tooling (`visualize_attention_runs.py` descent rollout, `attention_perlayer_grid.py`, `attention_nmt_style.py`) pointed at the `postr_det` checkpoints:

- `descent_v5_postr_det_seed{1..5}_final.{png,pdf}` — descent attention rollout at the 1M checkpoint, all seeds
- `descent_v5_postr_det_seed1_at800k` / `_at900k` — seed 1 straddling its ~813k collapse
- `perlayer_v5_postr_det_seed1`, `grid_v5_postr_det_seed1`, `nmt_v5_postr_det_seed1` — per-layer maps, grid projection, and full-trajectory (OOD probe) view at 1M

Note: because `postr_det` runs use the neural-linear posterior with dropout at 0, the "dropout OFF" and "locked M̂" panels are identical by construction — attention routing is deterministic in these runs; the posterior lives in the BLR head, not the attention.

**Finding — the collapse coincides with an attention-routing shift away from action tokens.** At 800k steps (rolling solve rate still ≈99%), seed 1's rollout attention is action-dominant and uniform across the descent: each prediction draws ~0.35 from each action token and ~0.15 from each state token, i.e. the model's next-state prediction is strongly conditioned on which action is being evaluated — exactly what discriminating `right` from `left` requires. At 900k (solve rate ≈12%) the routing has inverted: state tokens receive 0.17–0.46 and action tokens drop as low as 0.10–0.14, and the pattern becomes row-dependent rather than uniform. The 1M map is similar (action attention 0.08–0.21). A transition model that under-attends the action token predicts similar next-states for both actions, which flattens the Q-gap between `right` and `left` and lets the policy wander off the diagonal — a mechanistic account of the retention failure consistent with the replay-eviction hypothesis (post-collapse training data contains few successful right-descents, so action identity loses predictive value in the buffer distribution).

Seed 3 (never retained; ~12% throughout) shows the same state-leaning, near-uniform routing (action attention 0.19–0.34) at 1M — the "collapsed" signature, without ever having had the action-dominant phase at a checkpoint boundary.

Per-layer view (seed 1, 1M): layer 0 attends mostly to state tokens (and gives a(t) literally 0.0 attention), layer 2 collapses to a fixed distribution dominated by s(t) (0.67) regardless of query. The action information that survives enters only via layer 1's diffuse routing — the fragile link.

## Sample efficiency (added 2026-07-16)

Figure: `plots/seeded_comparison/psdrl_vs_postr_cumulative_regret.png` (per-episode regret = optimal return − achieved return; optimal 0.99 for PSDRL / 0.984 for PoSTR due to its shorter time limit).

| Agent | Seed | First treasure (step) | Time-to-learn (rolling-500 solve ≥ 50%) | Cum. regret @100k | Cum. regret @1M | Regret/ep, last 200k |
|---|---|---:|---:|---:|---:|---:|
| PSDRL | 1 | 140 | never | 19,838 | 198,108 | 0.990 |
| PSDRL | 2 | 220 | never | 19,822 | 198,091 | 0.990 |
| PSDRL | 3 | 5 | **2,500** | **2,347** | **23,649** | **0.118** |
| PSDRL | 4 | 11,155 | never | 19,802 | 198,017 | 0.990 |
| PSDRL | 5 | 7,035 | never | 19,832 | 198,092 | 0.990 |
| PoSTR | 1 | 15 | 6,123 | **1,172** | 36,351 | 0.791 |
| PoSTR | 2 | 10 | 4,313 | 3,001 | 158,748 | 0.878 |
| PoSTR | 3 | 75 | 8,726 | 16,609 | 177,398 | 0.869 |
| PoSTR | 4 | 5 | 124,564 | 15,965 | 175,217 | 0.876 |
| PoSTR | 5 | 10 | 4,838 | 15,317 | 161,176 | 0.864 |

Three regimes: **discovery** (PoSTR decisively better: median 10 vs 220 steps to first treasure, and PSDRL's discoveries are mostly wasted), **consolidation** (PoSTR converts discovery into a ≥50% policy on 4/5 seeds within 4–9k steps; PSDRL on 1/5), **retention** (PSDRL's one solved seed is near-flat at 0.118 regret/ep; every PoSTR seed drifts back to ~0.85). Mean cumulative regret at 1M: PoSTR 141.8k vs PSDRL 163.2k; median 161.2k vs 198.1k. **Neither achieves sublinear regret** — the PSRL √T-style guarantee is forfeited by PSDRL at the discovery end (sharpened sampling → 4/5 seeds pay maximal regret forever) and by PoSTR at the retention end (instability tax of ~0.85/ep forever).

**Attribution caveat:** the discovery advantage cannot yet be credited to the transformer. `PSDRL-deepsea5-prior1e3` (transition/reward prior widened 10× to 1e-3; 1 seed, 100k steps) solved DeepSea-5 at an 88% solve rate — a legitimate *prior* change (data-coupled shape, wider scale), suggesting PSDRL-ctrl's failure is posterior-scale miscalibration, not the GRU model class. A fair architecture comparison needs PSDRL at prior 1e-3 × 5 seeds × 1M as the baseline.

Note: sample efficiency ≠ compute efficiency — PoSTR trains 2.5× more often (update_freq 100 vs 250) with a much larger model; claims here are per environment step only.

## Final four-way comparison (added 2026-07-17)

Two new 5-seed × 1M-step arms completed 2026-07-17 on TARDIS: **PSDRL prior 1e-3** (`PSDRL-deepsea5_prior1e3_1M`, the corrected baseline — the original ctrl mistakenly ran the instructed 1e-3 prior as 1e-4) and **PoSTR v5 ring-only** (`BTRL-v5_ring_only_seed{1..5}`: BLR true posterior, no `explore_temp`, `protected_fraction 0.1`). Figure: `plots/seeded_comparison/final_fourway_comparison.png`.

| Arm | Seeds retained at 1M | First treasure (median) | Mean cum. regret @1M | Median | Mean final solve rate |
|---|---|---:|---:|---:|---:|
| PSDRL ctrl (1e-4) | 1/5 (88%) | 220 | 163,191 | 198,091 | 17.6% |
| PSDRL prior 1e-3 | 2/5 (87–88%) | 16,680 | **132,060** | 198,072 | 35.1% |
| PoSTR explore_temp 4 | 0/5 (floor 11–13%) | **10** | 141,778 | **161,176** | 12.0% |
| PoSTR ring-only | 0/5 (floor 12–14%) | **10** | 172,387 | 173,700 | 13.4% |

**PSDRL prior 1e-3:** widening the prior 10× doubled the consolidation rate (2/5 vs 1/5) and gives the best mean regret of any arm, but PSDRL remains bimodal — 3/5 seeds pay maximal regret for the full million steps, and its median regret is indistinguishable from ctrl. Discovery stays slow (first treasure at 10.7k–239k steps). **The PoSTR discovery/consolidation advantage survives the corrected baseline** (first treasure ≤75 steps on every seed across both PoSTR arms; time-to-learn 4–9k steps on most seeds vs 23–59k for PSDRL's two successes), so that claim can now be made fairly. Retention remains PSDRL's win: its consolidated seeds run at 0.12 regret/ep vs PoSTR's ~0.85.

**PoSTR ring-only — negative result:** all 5 seeds decayed to the same 12–14% solve-rate floor as `postr_det`, despite the ring being verifiably engaged (251 protected episodes on every seed, sampling-share floor active) and discovery being instant. Removing `explore_temp` neither fixed nor materially harmed anything (its mean regret is worse than postr_det's only because postr_det seed 1 happened to hold its plateau for 800k steps). **Guaranteed goal data + true-posterior BLR is not sufficient for retention.**

**Posterior-alive probe on ring-only seed 1** (temp-1 samples, K=40, main-buffer refit — conservative, since the runtime head additionally saw the ring):

| | goal eps in buffer | distinct next-states by depth 0–4 |
|---|---|---|
| 100k (healthy phase) | 25.3% | 1 / 2 / 1 / 1 / 1 (near-deterministic) |
| 1M (floor phase) | 23.5% | 1 / 6 / 10 / 1 / **18** (alive at depths 1, 2, 4) |

Contrast with postr_det @1M (no ring): fully collapsed at depths 0–3. **The ring partially restored self-widening — and performance still did not recover.** The bottleneck has therefore moved past uncertainty: an alive posterior cannot rescue a policy whose *mean* model and value function are wrong at depth, which localizes the remaining failure in the continually-retrained transformer features / value-training loop. Consistent with this, every PoSTR arm — dropout, BLR, explore_temp on/off, ring on/off — lands on the same ~12–13% floor, suggesting a common attractor of the feature/value loop rather than anything posterior-specific.

**Next experiment (the remaining unrun cell):** freeze or EMA the feature encoder used by the BLR head and value network after consolidation, while the live transformer continues training the mean model. This directly targets the stationary-φ premise that the self-widening theory requires and that every arm so far has violated.

## v6 frozen-φ ablation (added 2026-07-18)

5 seeds × 1M steps, frozen-φ ONLY (no ring, no explore_temp, BLR temp 1) — `v3_neural_linear` is the exact no-freeze control. Figure: `plots/seeded_comparison/v6_frozen_phi_vs_v3.png`. Freeze trigger: held-out transition-loss plateau (>2% rel. improvement resets, patience 20 update cycles) AND ≥1 reward-bearing episode in the buffer.

| Seed | φ frozen at (step) | Val loss at freeze | Solve rate by tenths of training | Final posterior (distinct/modal) |
|---|---:|---:|---|---|
| 1 | 3,404 | 1.64 | 0.31 — flat, ±1% for 1M steps | 6.0 / 0.35 |
| 2 | 2,004 | 1.62 | 0.36 — flat | 6.0 / 0.30 |
| 3 | 2,902 | 1.56 | 0.15 — flat | 4.0 / 0.55 |
| 4 | 4,304 | 1.54 | 0.07 — flat | 4.0 / 0.50 |
| 5 | 10,302 | 0.57 | 0.07 — flat | 2.2 / 0.89 |

**Result 1 — instability is eliminated.** Every seed is a flatline: solve rate constant to ±1% across all ten tenths of a million steps, with zero measured feature drift post-freeze and the posterior alive throughout. No learn-then-forget, no dips, no floor-convergence dynamics. Against the control's chaotic churn (spikes to 90%+ followed by collapse), this is direct confirmation by elimination: **continual retraining of φ is the instability mechanism.** Kill the drift, kill the collapse.

**Result 2 — the freeze fired too early, capping the ceiling.** All seeds froze at 2–10k steps, before value consolidation completed, so each locked in a mediocre policy (7–36%) and held it perfectly forever. The theoretical trade-off (drift risk ↔ fixed approximation error) landed exactly as stated, with the trigger choosing a bad t₀: on the small early buffer, validation loss plateaus trivially. Notably, freeze-time val loss does *not* predict the ceiling (seed 5 froze latest with the best val loss, 0.57, yet performs worst at 7% with the most collapsed posterior) — next-state prediction accuracy on the buffer distribution is not the same thing as goal-relevant discrimination in the features.

**Combined interpretation across arms:** v3 (drifting φ) reaches high performance transiently and always loses it; v6 (frozen φ) keeps whatever it has forever but stops improving. Consolidation *requires* continued representation learning; retention *requires* representation stationarity. A hard freeze can't have both — which is precisely the two-timescale motivation for the EMA variant (v7): the BLR/value system reads a slowly-moving EMA copy of φ (target-network style), so the representation keeps improving on a timescale much slower than the posterior refits. Alternative/complementary: a stricter freeze criterion (plateau measured against post-turnover fresh data, or triggered on value-loss consolidation rather than transition-loss plateau).

## Value-function inertness — the RTG/value probe (added 2026-07-19)

Prompted by the question "can we project the RTG head to show the learned value of all states?": the RTG head cannot show learned values — it is bypassed in the v3+ prediction path (`transformer_rl_agent.py:82` routes rewards through the BLR) *and* untrained (the `transition.py` optimizer predates the prediction heads, so `predict_rtg`/`predict_state` are frozen random projections in every arm). Probing the **value network** instead produced a much bigger finding.

**Finding 1 — the value network is a constant function in every v3+ arm, at every checkpoint, including during healthy plateaus.** std of V over the 25 one-hot states = 0.00000 for all `postr_det` seed-1 checkpoints (including the 800k checkpoint inside its 99% plateau), ≤0.0007 for all v6 and v8 checkpoints. Its output swings globally between checkpoints (tracking a mean-return-like scalar) but never discriminates states. Evaluated on the BLR's predicted state logits (the actual acting-time input), it returns the same constant for every state and both actions (2.3965 at postr_det 800k). Figures: `plots/seeded_comparison/valuegrid_v6_seed1.png`, `valuegrid_v8_seed1.png`.

**Finding 2 — the operative policy is the BLR reward margin, nothing else.** Decomposing the acting score at postr_det 800k: preference for RIGHT comes entirely from r̂(right)=0.188 vs r̂(left)=0.000, identical at every diagonal state; the value term is a constant offset. The full-grid margin map (`blr_reward_margin_800k_vs_1M.png`) shows the consolidated "policy" was a **uniform +0.187 always-right bias** in the reward column — not a state-dependent plan. At 1M (floor phase) the margins have collapsed to near-zero noise (±0.05), which with per-episode Thompson sampling of W yields exactly the observed ~12% stochastic solve floor.

**Implications:**
1. No component of the v3+ agent performs multi-step credit assignment: the RTG head is untrained, the value net is inert, and acting is one-step reward-greedy. Every retention result (ring, freeze, LoRA) has really been modulating the *stability of the BLR reward margin* under feature drift.
2. The value net's flatness is a self-consistent fixed point: flat V ⇒ bootstrap targets ≈ r̂ + γ·const ⇒ targets carry no state signal ⇒ V stays flat. It also trains on one-hot inputs but is queried on continuous logits (train/eval input mismatch). Candidate v9 fixes: train V on the model's predicted-logit distribution, fix the optimizer-heads quirk, and check target symmetry-breaking at init.
3. Benchmark caveat for the thesis: DeepSea's optimal policy (always-right) is expressible as a constant action bias, so DeepSea alone cannot distinguish value-based deep planning from a reward-margin bias. The deep-exploration claim (finding the treasure) stands; claims about learned *value structure* need a task whose optimal policy is state-dependent.

## Config notes

- BTRL (`postr_det`): dropout disabled (`p_seq: 0.0`, `p_attn: 0.0`), neural-linear posterior (`neural_linear: true`, `blr_coefficient: 100`, `explore_temp: 4.0`), 3 encoder layers, hidden 512, time limit 6, `update_freq: 100`.
- PSDRL (`ctrl`): stock config — GRU transition (hidden 256/GRU 128), priors 1e-4, `update_freq: 250`, `policy_noise: 0.05`, time limit 25.
- Both: deterministic DeepSea-5, replay capacity 10,000, 1M steps, checkpoints every 100k.
