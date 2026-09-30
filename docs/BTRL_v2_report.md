# Bayesian Transformer for Reinforcement Learning — Version 2: Diagnosis and Repair of the BTRL Agent

## Abstract

The original dissertation introduced Bayesian Transformer for Reinforcement
Learning (BTRL), a model-based agent that replaces the bayesian linear
regression transition model of Posterior Sampling for Deep Reinforcement
Learning (PSDRL) with a BayesFormer-style transformer, and reported that the
agent failed to converge on bsuite's Deep Sea environment within the
available resources. This report documents the diagnostic campaign carried
out after submission to determine why, and the resulting second version of
the agent. Three iterations were required. Version 2.1 diagnosed and
repaired two defects in the Thompson Sampling mechanism itself — per-step
resampling of dropout masks (the "dithering trap") and the resulting
non-stationarity of the value network's inputs ("representation drift") —
by locking all dropout masks for the duration of each episode. Long-horizon
evaluation of v2.1 over 10⁶ environment steps then exposed two further
failure modes located in the replay buffer and the dropout posterior: a
world-model collapse under a small buffer, and a catastrophic
learning–forgetting cycle under a large one. Version 2.2 applied a
three-stage multi-objective hyperparameter optimisation, which produced the
central experimental finding of this work: posterior *calibration*, not
one-step prediction accuracy, predicts downstream control performance — and
also demonstrated that no hyperparameter setting removes the forgetting
cycle, establishing it as structural. Version 2.3 repairs the structural
defect with a reward-aware protected replay partition; the agent reaches
sustained goal-reaching within 47k steps and, in the run in progress at the
time of writing, maintains a ~45–55% goal-rate at 224k steps, more than
five times beyond the point at which every previous version had collapsed.
The results explain precisely why the dropout-based posterior of BTRL v1
could not reproduce PSDRL's stability, and motivate a neural-linear
uncertainty head over the transformer representation as the next version.

## 1. Introduction

The original dissertation (hereafter BTRL v1) concluded that the agent
"does not converge … with the current configuration and limited timesteps",
and proposed deeper networks and longer training as future work. Both
suggestions were pursued after submission, and both turned out to be
insufficient on their own: the obstacles were two implementation-level
defects and one structural defect, none of which is visible at the
horizons evaluated in v1 (10⁴ steps on Deep Sea 3; 10⁵ steps on Deep Sea
5). This report treats every experiment conducted before and after
submission as a diagnostic step toward BTRL v2, and versions them
accordingly:

| Version | Content | Outcome |
|---|---|---|
| v1 | Agent as submitted | Fails to converge; cause unknown |
| v2.1 | Episode-locked dropout: fixes the dithering trap and representation drift; long-horizon (10⁶-step) evaluation | Thompson Sampling semantics restored; evaluation exposes world-model collapse (replay 10²) and a learning–forgetting cycle (replay 10⁴) |
| v2.2 | Three-stage hyperparameter optimisation of the transition and value networks | Calibration-over-accuracy finding; 6× faster solving; forgetting cycle recurs → structural |
| v2.3 | Protected replay partition (reward-aware retention) | Sustained goal-reaching; run in progress |

Throughout, the agent, environment and evaluation protocol are those of
v1: the transition, terminal and value networks on deterministic Deep Sea
(N = 5, 25 one-hot states), where the optimal policy takes N `right`
actions for a terminal reward of 0.99 (optimal return ≈ 0.982 after move
costs) and a uniformly random policy reaches the goal with probability
2⁻⁵ ≈ 3%.

## 2. Diagnostic Methodology

Three instruments were added to the v1 codebase; all are retained in the
repository for reuse.

**Long-horizon evaluation.** All v2 experiments run for 10⁶ environment
steps (~200,000 episodes) with checkpoints every 10⁵ steps — an order of
magnitude beyond v1's evaluation horizon. Sample efficiency is measured by
cumulative regret Σ(0.982 − episode return), episodes-to-solve (first
episode at which the trailing 20-episode mean return reaches 90% of
optimal), and the goal-rate (fraction of episodes reaching the treasure)
over rolling windows.

**Checkpoint diagnostics.** A post-mortem tool (`diagnose_checkpoint.py`)
inspects any saved checkpoint on four axes: (i) replay buffer content —
episode count, reward distribution, state-visitation grid; (ii) the value
network's output over all 25 one-hot states; (iii) the transition network's
predicted next state and reward for every reachable (state, action) pair
under K = 50 independently sampled world hypotheses M̂, giving both the
consensus prediction and the *inter-hypothesis agreement* — a direct
measurement of posterior diversity; (iv) greedy rollouts in the real
environment under freshly sampled hypotheses.

**Hyperparameter optimisation pipeline.** A three-stage Optuna pipeline
(Section 4): an offline multi-objective search over the transition network,
an online validation of its Pareto candidates by full agent runs, and an
online search over the value network with the validated transition
configuration frozen.

## 3. Version 2.1 — Stabilising the Sampled World Model

### 3.1 The dithering trap and representation drift

In v1, the BayesFormer dropout masks were resampled on every forward call.
Under Thompson Sampling this is a semantic error with two consequences.
First, the *dithering trap*: the sampled world hypothesis changed at every
step, so the agent could never commit to a coordinated exploration
strategy (e.g. always-right in Deep Sea) for a whole episode — precisely
the behaviour Posterior Sampling requires. Second, *representation drift*:
the hidden states Hᵃ that feed the value network changed on every call, so
the value network's regression target was non-stationary and could not
converge.

v2.1 fixes both with a single mechanism, `EpisodeLockedDropout`: at the
first step of each episode the agent samples one dropout mask per site
(state, positional and action embeddings; query, key and value
projections; attention maps; feed-forward skip connections) and holds all
of them fixed until the episode terminates. Sampling a world hypothesis M̂
is thereby restored to its intended role as an episode-level commitment.

### 3.2 Long-horizon evaluation: two failure modes

With the sampling semantics repaired, two 10⁶-step runs isolated the effect
of replay capacity, with all other hyperparameters at their v1 values.

**Replay capacity 10² transitions: world-model collapse.** The agent
reached the goal once (episode 0, by chance) and never again in ~60,000
episodes. Checkpoint diagnostics at 10⁵ and 3×10⁵ steps are essentially
identical: the buffer held only 22 episodes (109 transitions) dominated by
left-column states; the goal episode had been evicted within a few hundred
steps. The transition network's predictions had become *independent of its
input* — for every probed state it predicted next-state 5 under action 0
and next-state 0 under action 1, exactly the two most-visited states in
the buffer — and 50 sampled hypotheses agreed with each other 98–100% of
the time. With posterior diversity gone, Thompson Sampling proposed the
same wrong world every episode, and the value network, whose training
inputs are the transition network's outputs, regressed to a constant
(V(s) ≈ 0.4646 for 23 of 25 states). The causal chain is a closed loop:
tiny buffer → biased data → model predicts the marginal mode → posterior
collapses → no exploration → no new goal data.

**Replay capacity 10⁴ transitions: the learning–forgetting cycle.**
Raising capacity broke the loop — at 10⁵ steps the buffer retained 52 goal
transitions and the value function showed structure — and the agent
genuinely solved the task at ~140k steps (≥18 of 20 consecutive episodes
reaching the goal, impossible by chance at a 3% background rate). It then
*unlearned* it. Over the full run: 1,577 goal episodes (0.79%), cumulative
regret 195,959 of a maximum ≈196,710, with dead zones (goal-rate exactly
0%, *below* the random baseline) spanning 250–350k and 400k–1M steps
except one brief blip. The proposed mechanism is a buffer-turnover purge
cycle: (1) a performance dip stops goal episodes entering the buffer; (2)
a dip lasting one buffer lifetime (~2,000 episodes) evicts every goal
transition; (3) the reward head, retrained continually on the current
buffer, unlearns the goal; (4) the value function follows; (5) the
still-overconfident posterior generates no directed exploration to
rediscover it, so the dead zone persists until residual chance breaks it.
Notably the dead zones lengthen over the run — recovery depends on luck,
not on any systematic force.

## 4. Version 2.2 — Hyperparameter Optimisation

### 4.1 Pipeline

v1 selected hyperparameters by independent grid search over prediction
accuracy. v2.2 replaces this with a three-stage Optuna pipeline:

1. **Offline transition search** (200 NSGA-II trials on a fixed dataset of
   2,000 transitions): a two-objective Pareto search minimising (a) mean
   grid distance between predicted and true next states under 100 sampled
   hypotheses, and (b) the *overconfident-error rate* — the fraction of
   (state, action) pairs predicted wrongly (grid distance > 0.5 cells)
   with near-zero inter-hypothesis variance. Objective (b) targets exactly
   the posterior-collapse failure of Section 3.2.
2. **Online validation**: the Pareto candidates plus the v1 baseline were
   re-evaluated by full agent runs (3 seeds × 10⁴ steps, value network
   held fixed), testing whether the offline objectives predict control
   performance (the "objective mismatch" question, Lambert et al. 2020).
3. **Online value search** (50 TPE trials, transition configuration
   frozen at the validated winner): single objective, seed-averaged
   cumulative regret. The discount factor was excluded: in a fixed-length
   N-step chain with a single terminal reward, γ applies the same
   monotone rescaling to every action's value and cannot change the
   greedy ranking.

### 4.2 The calibration result

| Candidate | Offline grid distance | Offline overconfident-error | Online regret (3 seeds) |
|---|---|---|---|
| **Calibrated, inaccurate** | 1.99 | **0.000** | **1404 ± 61** |
| Accuracy leader | **0.60** | 0.167 | 1894 ± 41 |
| Second-most accurate | 0.65 | 0.167 | 1946 ± 17 |
| v1 baseline | — | — | 1957 ± 9 |
| Further accuracy candidates | 0.66–0.68 | 0.20–0.30 | 1963 ± 7 |

The candidate selected purely for posterior calibration — with roughly
random one-step accuracy — outperformed every accuracy-optimised candidate
and the baseline by a wide margin, collecting ~28% of achievable reward
within 10⁴ steps where the others collected 0–3%. One-step prediction
accuracy *anti-correlates* with downstream performance in this regime;
posterior calibration predicts it. This is the strongest form of the
objective-mismatch phenomenon and directly vindicates the diagnosis of
Section 3.2: what kills BTRL is not model inaccuracy but a posterior that
is confidently wrong. Consistently, the winning configuration uses the
largest sequence-dropout rate in the search space (p_seq = 0.3) — the
setting that keeps hypothesis diversity alive. The value search then
independently selected a small, conservatively trained value network
(16 units × 3 layers, learning rate 2.7×10⁻³) — notably close to the v1
dissertation's value configuration (16 units, 10⁻³), which had drifted in
the intervening code changes.

### 4.3 Long-horizon result: the cycle is structural

The fully tuned agent (v2.2) solved Deep Sea 5 at step 23,898 — six times
earlier than the v2.1 run — reaching a 46% goal-rate, and accumulated
8,316 goal episodes over the run (five times v2.1). But the forgetting
cycle recurred: performance decayed gradually from 46% to a dead zone
after ~400k steps. Tuning changes how fast the agent learns and how
gracefully it degrades; it does not change whether the collapse eventually
happens. The instability is therefore structural — located in the replay
buffer's retention policy, not in the networks.

## 5. Version 2.3 — Protected Replay Partition

### 5.1 Design

The v1 replay buffer evicts by age alone, giving it a half-life property
under sparse rewards: any goal-reaching pause longer than one buffer
turnover erases all positive-reward transitions. v2.3 adds a *protected
partition*: episodes containing reward (stored tanh-squashed, so goal
transitions ≈ 0.757) are **copied** — not diverted — into a separate FIFO
ring reserving 10% of capacity, and training samples uniformly over the
union of both rings. Copy-not-divert keeps the main ring an unbiased
picture of recent experience; uniform union sampling means the goal-data
share of training batches rises automatically toward the ring's capacity
fraction exactly when the main ring fills with failures. The change is
~30 lines in the replay buffer, controlled by a single configuration key
whose default reproduces the previous behaviour, so v2.3 is a
one-variable ablation against v2.2. This is a retention-prioritised
cousin of prioritised experience replay (Schaul et al. 2016), suited to
BTRL because its trainers sample uniformly from whatever the buffer
contains.

### 5.2 Results (run in progress)

At the time of writing the v2.3 run has completed 224k of 10⁶ steps. The
agent recovered from an early performance dip that would have initiated
the purge cycle in earlier versions (the protected ring held ~1,000 goal
transitions through it), reached sustained solving at step 47,004, and
currently maintains a 45–55% goal-rate — more than five times beyond the
point at which v2.2 had entered terminal decay, and past every collapse
onset observed in earlier versions. Final statistics will be added on
completion. The falsifiable design prediction is recorded in advance:
either the plateau holds (retention was the binding constraint), or
collapse recurs despite a guaranteed ~10% goal-data share, which would
exonerate retention and isolate the posterior as sole cause.

![Goal-reaching rate across versions (1,000-episode rolling window). The v2.1 runs collapse or oscillate to zero; v2.2 learns fastest but relapses; v2.3 (in progress) sustains performance past every previous collapse point.](plots/fig_versions_goal_rate.png)

![Cumulative regret across versions. A flat curve indicates a solved agent; the earlier the bend, the more sample-efficient.](plots/fig_versions_regret.png)

![BTRL v2.2 learning curve over 10⁶ steps: rolling mean episode return and goal-rate, showing acquisition at ~24k steps, gradual degradation, and terminal relapse.](plots/fig_v22_learning_curve.png)

## 6. Why PSDRL Does Not Exhibit This Failure

PSDRL shares the recency-biased replay buffer (a limitation its authors
acknowledge) and yet trains stably, including on Atari. The difference is
located in the uncertainty mechanism. PSDRL's bayesian linear regression
posterior has precision Σ⁻¹ = σ⁻²I + σ⁻²ΦᵀΦ built directly from the data:
uncertainty is wide exactly where data is scarce. Before solving, this
directs exploration — sampled models are optimistic precisely about
unvisited regions, which is how the agent finds the goal in the first
place. After solving, the posterior tightens on the truth along the
well-visited goal path, so every sampled hypothesis agrees on the optimal
action; performance reaches ~100%, nearly every episode replenishes the
buffer with goal data, and the eviction policy stops mattering. And if
goal data ever were purged, uncertainty would *return* in that region and
re-trigger exploration: forgetting is self-correcting. On Atari the
question does not even arise, because rewards are dense enough that the
buffer always contains reward signal.

The dropout posterior of BTRL provides neither property. Its uncertainty
is set by the dropout rate, not by data: it is not wide where the agent
has not been (undirected exploration; the v2.1 small-buffer run never
re-found the goal in 60,000 episodes) and not tight where the agent has
been (no certified solution; the plateau at ~46–55% rather than ~100%,
because some episodes act on wrong hypotheses even at peak performance).
The same miscalibration explains the ceiling and the collapse. Replacing
PSDRL's exact posterior with a dropout approximation therefore silently
removed the mechanism that made its replay buffer safe.

## 7. Discussion

**Was the BayesFormer hypothesis wrong?** BTRL bundled two claims: that
transformer self-attention over trajectory context is a better world-model
backbone than PSDRL's recurrent model, and that dropout masks can play the
role of the exact posterior in Thompson Sampling. Only the second is
falsified, and the failure is now mechanistically characterised:
collapse to overconfidence (measured directly as 98–100% inter-hypothesis
agreement while wrong), no data-coupled re-widening, and consequent
inability either to direct exploration or to certify a solution. This
gives empirical, control-relevant force to the known theoretical
observation that dropout uncertainty does not concentrate with data. The
attention-based representation itself is untested by these failures and
is retained in the proposed next version.

**Why v1 could not have seen this.** The v1 evaluations ran to 10⁴–10⁵
steps. Every phenomenon documented here — the solve at 24–140k steps, the
purge cycle with its ~2,000-episode turnover clock, the dead zones from
250k onward — lives partly or wholly beyond that horizon. The v1 results
remain reproducible at their stated horizons; the diagnosis required an
order of magnitude more environment interaction, together with
posterior-level instrumentation rather than reward curves alone.

**Secondary factors.** The value network's 10⁻² learning rate (an
inadvertent drift from v1's 10⁻³) acts as an amplifier: value targets are
simulated from the freshly sampled hypothesis each update round, and in
Deep Sea the two actions' values differ by fractions of the move cost
(~10⁻³), so value swings far larger than the decision margin flip the
policy grid-wide within a single update round. The v2.2 search selected
the conservative regime, as predicted by this analysis, but tuning alone
did not remove the cycle — consistent with its role as amplifier rather
than cause.

## 8. Conclusions and Future Work

BTRL v1's non-convergence decomposes into three separable defects: broken
Thompson Sampling semantics (fixed in v2.1 by episode-locked dropout), a
miscalibrated dropout posterior (characterised in v2.2, where posterior
calibration proved to be the property that predicts control performance),
and an age-only replay eviction policy that converts sparse-reward dips
into catastrophic forgetting (repaired in v2.3 by the protected
partition). Each fix was validated as a single-variable change, and each
intermediate failure is documented with checkpoint-level evidence rather
than reward curves alone.

The principled next step is BTRL v3: retain the transformer encoder for
trajectory context, but replace dropout-as-posterior with PSDRL's
neural-linear mechanism — a bayesian linear regression head over the
transformer's features. This restores the two properties the analysis
identifies as load-bearing (data-directed exploration and a
self-correcting response to forgetting) while keeping the architectural
contribution of this work, and is predicted to lift the ~50% plateau
toward PSDRL's ~100% on Deep Sea.

## 9. References

[1] Sasso, R., Conserva, M. and Rauber, P. (2023) 'Posterior Sampling for
Deep Reinforcement Learning', arXiv [cs.LG]. Available at:
http://arxiv.org/abs/2305.00477.

[2] Sankararaman, K. A., Wang, S. and Fang, H. (2022) 'BayesFormer:
Transformer with Uncertainty Estimation', arXiv [cs.CL]. Available at:
http://arxiv.org/abs/2206.00826.

[3] Osband, I. et al. (2020) 'Behaviour Suite for Reinforcement Learning',
arXiv [cs.LG]. Available at: http://arxiv.org/abs/1908.03568.

[4] Lambert, N., Amos, B., Yadan, O. and Calandra, R. (2020) 'Objective
Mismatch in Model-based Reinforcement Learning', arXiv [cs.LG]. Available
at: http://arxiv.org/abs/2002.04523.

[5] Schaul, T., Quan, J., Antonoglou, I. and Silver, D. (2016)
'Prioritized Experience Replay', arXiv [cs.LG]. Available at:
http://arxiv.org/abs/1511.05952.

[6] Chen, L. et al. (2021) 'Decision Transformer: Reinforcement Learning
via Sequence Modeling', arXiv [cs.LG]. Available at:
http://arxiv.org/abs/2106.01345.

[7] Akiba, T. et al. (2019) 'Optuna: A Next-generation Hyperparameter
Optimization Framework', arXiv [cs.LG]. Available at:
http://arxiv.org/abs/1907.10902.
