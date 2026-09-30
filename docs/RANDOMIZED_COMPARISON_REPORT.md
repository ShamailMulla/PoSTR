# PoSTR vs PSDRL on Randomized DeepSea-5 — 5 seeds × 1M steps

**Dates:** runs 2026-09-28 12:40 → 2026-09-30. **Host:** TARDIS (RTX 3070 Laptop, all 10 runs concurrently).
**MLflow:** experiment "Randomized DeepSea-5: PoSTR vs PSDRL (Sep 2026)" — groups
"PoSTR postr_rand (randomized DeepSea)" and "PSDRL prior 1e-3 (randomized DeepSea)".
**Raw numbers:** `docs/randomized_comparison_summary.json`.

This supersedes every PoSTR-vs-PSDRL comparison in `SEEDED_COMPARISON_REPORT.md`,
which compared the two agents on **different tasks** (see "What changed" below).

## What changed from the July / early-September runs

| | Before | Now (both agents) |
|---|---|---|
| Task | PoSTR: non-randomized DeepSea with an `env_step` override (episode ends with 0.99 on reaching the corner, 4 steps). PSDRL: standard randomized DeepSea | standard bsuite DeepSea-5, `randomize_actions=True`, 5-step episodes, +1 − move costs (optimal return 0.99) |
| Action mapping | PSDRL train and test envs drew **different** unseeded random mappings | one mapping per run seed, shared by train and test |
| PoSTR reward loss | broadcast over all 100×100 prediction/target pairs (`transition.py:82`) | fixed (`view_as`) |

## Configs

- **PoSTR (`postr_rand`)** — `config_v3_neural_linear.yaml` via `run_v3_1M.py`: neural-linear
  (BLR) head, `explore_temp 4`, dropout off, update every 100 steps, replay 10k,
  time limit 6. Identical to the July/Sep `postr_det` arm apart from the fixes above.
  Traced 1 update cycle in 20.
- **PSDRL (`psdrl_rand`)** — `psdrl_deepsea5_prior1e3_1M.yaml` via `baselines/psdrl/src/run_psdrl.py`:
  the corrected baseline (priors 1e-3), GRU transition model, update every 250 steps,
  `policy_noise 0.05`, time limit 25.

## Results

"Solved" = episode return > 0.9. Time-to-learn = first step at which the rolling-500
solve rate reaches 50%. Random policy solves 1/32 ≈ 3.1%.

| Agent | Seed | First treasure | Time-to-learn | Solve (all) | Solve (last 100k) | Test (last 100k) | Best rolling-1k | Regret @1M |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PSDRL | 1 | 45 | 176,665 | 72.6% | 88.4% | 30/39 | 92% | 54,408 |
| PSDRL | 2 | 3,325 | never | 0.0% | 0.0% | 0/39 | 4% | 198,083 |
| PSDRL | 3 | 7,925 | 136,480 | 75.7% | 88.0% | 33/39 | 92% | 48,253 |
| PSDRL | 4 | 6,395 | 201,145 | 70.3% | 87.9% | 34/39 | 92% | 58,907 |
| PSDRL | 5 | 8,005 | 145,500 | 74.8% | 87.9% | 33/39 | 91% | 50,032 |
| PoSTR | 1 | 1,001 | never | 5.5% | 2.0% | 4/100 | 19% | 188,116 |
| PoSTR | 2 | 3,286 | never | 3.1% | 3.1% | 1/100 | 5% | 192,895 |
| PoSTR | 3 | 446 | never | 1.6% | 1.7% | 2/100 | 5% | 195,875 |
| PoSTR | 4 | 1,011 | never | 2.8% | 2.7% | 3/100 | 5% | 193,534 |
| PoSTR | 5 | 176 | never | 3.4% | 6.5% | 4/100 | 9% | 192,202 |

| Aggregate | PSDRL | PoSTR |
|---|---:|---:|
| Seeds that learned (≥50%) | **4/5** | 0/5 |
| Median first treasure (step) | 6,395 | **1,001** |
| Mean solve rate, last 100k | **70.4%** | 3.2% |
| Cumulative regret @1M, mean / median | **81,937 / 54,408** | 192,524 / 192,895 |

Solve rate by fifths of training (0–200k … 800k–1M):
PSDRL 1: 11/88/88/88/88%, 3: 29/86/88/88/88%, 4: 0/87/88/88/88%, 5: 24/86/88/88/88%, 2: 0 throughout.
PoSTR: 0–9% in every fifth for every seed.

## Findings

1. **PSDRL solves randomized DeepSea-5; PoSTR does not.** Four PSDRL seeds learn between
   136k and 201k steps and hold the solution to 1M. 88% is this config's ceiling:
   `policy_noise 0.05` takes a random action 5% of the time, half of which are wrong, so
   the best achievable rate over 5 steps is 0.975⁵ ≈ 88.1%. PoSTR stays at the random
   rate (1.6–6.5% over the last 100k) on every seed; its regret is 97% of the maximum.
2. **PoSTR still discovers faster** (median first treasure 1,001 vs 6,395 steps) but
   never converts a discovery into a policy.
3. **PSDRL's July test anomaly is explained.** Test solve rates now track training
   (77–87% vs ~88%); the July 88%-train / 3.2%-test gap came from the test environment
   using a different random action mapping.
4. **PoSTR's earlier results depended on the non-randomized task.** There, "always the
   same action" is optimal, so a constant bias in the reward model could solve it. With
   per-cell action mappings that route is gone.
5. **Mechanism, from the traces (501 per seed):** the value network is flat across all 25
   states on seeds 1–4 (median spread 0 to 3.5e-5; seed 5 median 0.014), and posterior
   diversity drops from 13–15 distinct predicted next states at the start to 1–4 by the
   end (seed 3: 1). With a flat value network, PoSTR acts one-step greedy on predicted
   reward, which cannot find a 5-step path whose correct action differs per cell — the
   outcome predicted in `wiki/planning-in-sampled-model.md` §10 (cell P1).

## Implications

- PoSTR in its current form (neural-linear head + learned value network) should not be
  presented as competitive with PSDRL on DeepSea.
- The next experiment is the one the theory page proposes: replace the value network with
  exact backward-induction planning in each posterior sample (Option 1, Markov model),
  with fixed-feature epochs, and compare against this PSDRL arm on the same task.
- PSDRL's own weak point is seed 2 (never learned) and 4-of-5 reliability; a
  planning-based PoSTR should be judged on median regret and on how many seeds learn.

## Caveats

- 5 seeds per arm; one action-mapping per seed (mapping seed = run seed), so seeds
  differ in both randomness and task instance.
- The 10 runs shared one laptop GPU; this affects wall-clock, not results.
- PSDRL and PoSTR still differ in time limit (25 vs 6) and update frequency (250 vs 100),
  as in July. The time limit does not bind here (episodes end at 5 steps).
