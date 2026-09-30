# Planning in the Sampled Model — PSRL without a Value Network

**Status:** theory proposal, 2026-09-26. Not implemented. Everything marked
*(verify)* is a citation or claim to check against the source before it goes in
the dissertation.

**One-line idea:** drop the [[value-network|value network]]. Each episode, draw one
model from the BLR posterior ([[neural-linear-head]]), compute that model's optimal
policy **exactly** by backward induction over the transformer's predicted next
states and rewards, and act on it. This is textbook PSRL
([[thompson-sampling]]); the value network in [[psdrl|PSDRL]] was only ever an
approximation of this planning step.

---

## 1. Why — the evidence

1. **The value network is inert.** `value_onehot_std` (spread of V over all 25 grid
   states) is ≈0 in every PoSTR arm and checkpoint (report, 2026-07-19), and the
   Sep 2026 rerun traces show it stays ≈0 (1e-6–0) **during** seed 3's 99% phases
   (46–77k, 523–529k). The agent solves DeepSea with a constant V, so the operative
   policy is one-step greedy on the BLR reward margin.
2. **One-step greedy cannot do deep exploration.** The treasure only enters the
   return after H steps; nothing in the current agent propagates a sampled model's
   optimism about it back to the first action.
3. **Collapses follow feature (φ) jumps, not value changes.** Rerun seed 3 at 529→531k:
   feature drift 0.198 (typical ~1e-4), posterior distinct next-states 4→1, then the
   collapse; V only becomes non-constant *after*. Seed 2 at ~31–33k shows the same
   pattern. Consistent with the v6 result that freezing φ removes the instability.
4. **The current sampling schedule is a hybrid** (see [[dithering-trap]] § Scope):
   the value net is trained under one posterior sample and used to act under another.

---

## 2. Setting

Finite-horizon episodic MDP `M = (S, A, H, P, R)`: finite states `S`, actions `A`,
horizon `H`, deterministic or stochastic transitions `P(s'|s,a,h)`, bounded mean
rewards `R(s,a,h)` (rescaled to `[0,1]` for the bounds; DeepSea's move cost is negative). DeepSea-N: `|S| = N²`, `|A| = 2`, `H = N`, deterministic.

Bayesian setting: the true MDP `M*` is drawn from a prior `f`. At the start of
episode `k` the agent has history `H_k` (all transitions so far) and picks a policy
`π_k`. With `V^π_{M,h}(s)` the value of `π` in `M` from step `h`,

```
Regret(T)       = Σ_{k=1..K}  V*_{M*,1}(s_1) − V^{π_k}_{M*,1}(s_1),     K = T / H
BayesRegret(T)  = E[ Regret(T) ]      (over M* ~ f, the data and the agent's sampling)
```

---

## 3. The agent (Option 1: Markov model)

**Model class.** The transformer supplies features `φ(s, a, h) ∈ ℝ^D` (plus a bias
entry) from the **current state, action and step only** — Option 1, no history
context. A weight matrix `W = [W_s | w_r]` defines a deterministic model `M_W`:

```
r̂_W(s,a,h)  = φ(s,a,h)ᵀ w_r
ŝ'_W(s,a,h) = argmax_j  φ(s,a,h)ᵀ W_s[:, j]          (one-hot next state)
```

**Loop.**

```
for episode k = 1, 2, ...
    W_k ~ N(μ_k, Σ_k)                          # one Thompson sample, held all episode
    Q_{H+1} ≡ 0
    for h = H, ..., 1:                        # backward induction in M_{W_k}
        Q_h(s,a) = r̂_{W_k}(s,a,h) + max_{a'} Q_{h+1}(ŝ'_{W_k}(s,a,h), a')
    act a_h = argmax_a Q_h(s_h, a) for h = 1..H
    update (Λ, B) with the episode's transitions  # §5, exact and incremental
    (optionally) update φ on the representation schedule   # §8
```

Terminal predictions: treat predicted termination as an absorbing zero-reward state
(the existing terminal network can supply it; DeepSea terminates at `h = H` anyway).

---

## 4. Proposition 1 — exact planning

> For a fixed `W`, backward induction in §3 returns `π*_{M_W}`, the optimal policy of
> the sampled model, using `|S|·|A|·H` model queries.

*Proof sketch.* `M_W` is a finite-horizon MDP with deterministic transitions, so
the Bellman optimality recursion `Q_h(s,a) = R(s,a,h) + max_{a'} Q_{h+1}(s',a')` has
no expectation and is solved exactly by backward induction (standard DP). The model
inputs are `(s, a, h)` only, so `Q_h` is a function of the state and step; paths
that reach the same `(s, h)` share their value, which is what lets the search
collapse from a tree to a table. ∎

**Why Option 1 matters.** A history-conditioned model predicts from a context window,
so two paths reaching the same cell with different histories can get different
predictions; nodes cannot be merged and exact planning needs the full
`|A|^(H−h)` action tree. The standard PSRL analysis also assumes a Markov model.
DeepSea is Markov, so nothing is lost by conditioning on `(s, a, h)`.

**Cost.** DeepSea-5: 25·2·5 = 250 queries; DeepSea-10: 2,000. All `(s,a,h)` features
can be computed in one batched forward pass. With φ frozen (§8) the feature table is
computed once, and planning each episode is just matrix products with `W_k`.

**Deterministic predictions.** Given `W`, `ŝ'_W` is a deterministic function
(argmax of fixed logits), so the one-hot projection keeps the sampled model
deterministic. Randomness enters only through the draw of `W`, once per episode.

---

## 5. Proposition 2 — the posterior is exact, incremental and policy-independent

Likelihood for each transition target `y = [onehot(s'), r]`:
`y = Wᵀφ + ε`, `ε ~ N(0, β⁻¹ I)`. Prior: each column of `W` ~ `N(0, α⁻¹ I)`.
For **fixed φ**, the posterior over each column is Gaussian with a shared precision:

```
Λ_k = α I + β Σ_{t<k} φ_t φ_tᵀ          B_k = Σ_{t<k} φ_t y_tᵀ
Σ_k = Λ_k⁻¹                              μ_k = β Σ_k B_k
```

(In the code `noise_variance` plays the role of the precision `β` and
`blr_coefficient` is `α`.)

> **(a) Sufficient statistics.** The posterior depends on the data only through
> `(Λ_k, B_k)`, which update in O(D²) per transition. The posterior over the
> **entire** history can be kept with no replay buffer at all.
>
> **(b) Policy-independence.** The likelihood of the data factorises over
> transitions; the agent's action choices depend only on past data and its own
> sampling randomness, so they drop out of Bayes' rule. The posterior is correct
> whatever policy collected the data.

*Consequence for the current agent.* With a drifting φ, statistics built in old
feature spaces cannot be added to new ones, so the head is refit from the replay
buffer. The buffer holds 10,000 transitions with FIFO eviction, which depends on
recent behaviour, so the "posterior" then conditions on a policy-dependent subset
of the data and is not a posterior of anything. This is a formal version of the
replay-eviction hypothesis in `SEEDED_COMPARISON_REPORT.md`.

**On- vs off-policy.** Neither label fits: no policy or value function is learned.
Data collection is on-policy (the agent acts with `π*_{M̂_k}`); the only learned
object is the model, and by (b) its posterior is valid under any behaviour policy —
no importance weights, no off-policy correction.

**Misspecification note.** The Gaussian likelihood on one-hot targets is a surrogate
(true transitions are categorical). Everything above is exact *for the surrogate
model class*; §6 deals with the gap to the true environment.

---

## 6. Regret

**Theorem (PSRL; Osband, Russo & Van Roy 2013 *(verify)*).** If `M* ~ f`, the agent
samples `M̂_k ~ f(· | H_k)` at the start of each episode and follows `π*_{M̂_k}`
for the whole episode, then

```
BayesRegret(T) = Õ( H · S · √(A · T) )
```

Key step: given `H_k`, `M*` and `M̂_k` have the same distribution, so
`E[V*_{M*}] = E[V*_{M̂_k}]` and the regret becomes
`E Σ_k (V^{π_k}_{M̂_k} − V^{π_k}_{M*})` — the gap between the sampled and true model
*on the agent's own policy*, which concentration bounds control.

**What PoSTR-P must satisfy to inherit it:**

| Requirement | How the design meets it | Status |
|---|---|---|
| R1. Sample from the exact posterior of the true prior | exact within the linear-Gaussian class (Prop. 2) | holds only within the class — see below |
| R2. Plan exactly in the sample | backward induction (Prop. 1) | holds (Option 1) |
| R3. Policy fixed within an episode | one draw per episode, no mid-episode redraw | requires removing the current mid-episode redraw |
| R4. Markov, finite horizon | `(s,a,h)` inputs; DeepSea | holds |
| R5. Untempered sample | `transition_temp = reward_temp = explore_temp = 1` | current configs use `explore_temp: 4` |

**Model class with fixed φ.** With frozen features the agent is PSRL over a model
class that is linear in known features. Osband & Van Roy 2014 *(verify)* bound PSRL
regret for a general model class via its Kolmogorov and eluder dimensions,
`Õ(√(d_K d_E T))`; for a class linear in `D` features both are `Õ(D)`, giving a bound
that depends on `D`, not on `|S|`. Related analyses for linear structure: linear
mixture MDPs / value-targeted regression (Ayoub et al. 2020) and randomized
least-squares value iteration (Zanette et al. 2020) *(verify all)*.

**Misspecification (simulation lemma, Kearns & Singh 2002 *(verify)*).** If some
`M_W*` in the class has `|R − r̂| ≤ ε_r` and `‖P(·|s,a) − P̂(·|s,a)‖₁ ≤ ε_P`, then for
every policy `|V^π_{M*} − V^π_{M_{W*}}| ≤ H ε_r + H² ε_P`. Over `K = T/H` episodes this
adds a term linear in `T`, roughly `T (ε_r + H ε_P)`. Two cautions:
- the posterior concentrates on the KL-closest model in the class, not the
  `L∞`-closest, so combining this with the Bayesian bound needs care — an open
  item (§11);
- the linear term is unavoidable under misspecification; it is exactly why the
  quality of φ matters, and it should be measured (prediction error of the frozen
  class on held-out transitions).

**Tempering.** Sampling with inflated variance (`explore_temp > 1`) is not posterior
sampling, so the Bayesian bound does not apply. Frequentist Thompson-sampling
analyses do inflate the variance (Agrawal & Goyal 2013 for linear bandits; Zanette
et al. 2020 for RLSVI *(verify)*), so it is a legitimate alternative route with a
different guarantee — but pick one framing. For the clean claim, use temperature 1.

---

## 7. Sampling schedule

| Schedule | Theory | Use |
|---|---|---|
| Per step (e.g. dropout mask per forward call) | breaks R3; causes the [[dithering-trap]] | never |
| Per episode | PSRL's schedule; the bound above | default for DeepSea |
| Lazy: resample when data doubles, or on a growing schedule | preserves regret with fewer switches for continuing tasks (TSDE, Ouyang et al. 2017 *(verify)*) | ablation |
| Every fixed `m` steps (PSDRL) | heuristic; PSDRL uses it because refitting its value net is expensive and Atari episodes are long | not needed once planning is cheap |

---

## 8. Where φ comes from — the open core

Prop. 2 and the regret statement need φ **fixed while it is used**. Options:

1. **Freeze after warm-up (v6).** Exact PSRL after the freeze. v6 showed the plateau
   trigger fires far too early (2–10k steps) and locks in poor features; the
   trigger needs a criterion tied to the goal-relevant structure (e.g. plateau on
   post-turnover data, or on reward-prediction error), not the transition loss on a
   small early buffer.
2. **Epochs of fixed features.** Keep φ fixed within an epoch, accumulate
   `(Λ, B)`; at the end of an epoch retrain φ, recompute the statistics over stored
   data, start the next epoch. Each epoch is exact PSRL in that epoch's class;
   doubling epoch lengths bound the number of switches by `O(log T)`. The regret
   of the whole run should decompose into per-epoch PSRL regret plus a switching
   cost — a candidate theorem (§11).
3. **EMA features (v7).** The head reads a slowly moving copy `φ̄`. Posterior error
   then depends on how far `φ̄` moves between refits; bounding it needs a
   Lipschitz argument on `μ, Σ` in the features — open.

Option 2 has the cleanest theory and reuses every result above unchanged, so it is
the recommended starting point.

---

## 9. Benchmark requirement

**Task mismatch found 2026-09-26.** PoSTR (BTRL) and the PSDRL baseline (BRL) have not
been trained on the same task:

| | PSDRL baseline (`baselines/psdrl/src/PSDRL/common/deepsea_wrapper.py`) | PoSTR (`src/TRL/common/utils.py`) |
|---|---|---|
| `randomize_actions` | `True` (bsuite default) | `False` |
| Episode | standard: 5 steps, +1 (minus move costs) for moving right *from* the bottom-right cell | `env_step` override: reaching the bottom-right cell ends the episode after 4 steps with reward 0.99 |

The override has been in `env_step` since at least 2026-06-25, so every July PoSTR run
used it. Any PoSTR-vs-PSDRL comparison (the four-way table in
`SEEDED_COMPARISON_REPORT.md`) compares different tasks, and needs rerunning on a
shared environment before it goes in the dissertation.

**Fixed 2026-09-26.** Both codebases now train on standard randomized DeepSea:
- BTRL: `randomize_actions` (default `true`), `mapping_seed` (default: the run seed,
  shared by train and test envs) and `legacy_goal` (default `false`; `true` reproduces
  the old task) are `experiment:` config keys; the 0.99 override moved out of
  `env_step` into the wrapper behind `legacy_goal`.
- BRL (PSDRL): train and test envs now share `mapping_seed`. Before, bsuite drew an
  unseeded random action mapping per environment, so **PSDRL's test episodes ran on a
  different task from its training** — the likely cause of PSDRL seed 3's 88% train vs
  3.2% test solve rate in the July report.

Use DeepSea with `randomize_actions=True` (bsuite's standard setting; every current
log prints "Only randomized_actions=True is the DeepSea environment"). With
`randomize_actions=False`, "always right" is optimal in every cell, so a constant
action bias solves the task — which is how a flat value network plus a reward bias
reached 99%. With randomized actions the correct action differs per cell, so only a
state-dependent plan can solve it, and planning vs no planning becomes measurable.

---

## 10. Predictions and the experiment

2×2 on randomized DeepSea-5 and -10, 5 seeds each:

| | drifting φ | fixed φ (epochs) |
|---|---|---|
| **value net** (current) | P1: floor ≈ random baseline; no retention | P2: no collapses, low ceiling (as v6) |
| **exact planning** | P3: discovers, still collapses at φ jumps | P4: discovers **and** retains |

**P1 observed (2026-09-30, randomized DeepSea-5, `RANDOMIZED_COMPARISON_REPORT.md`):**
current PoSTR stays at the random rate on 5/5 seeds while PSDRL learns on 4/5.

Falsifiable claims: P4 beats all other cells on regret at 1M; P3's collapses coincide
with feature-drift spikes (trace every cycle around them); P4's regret curve is
sublinear, as the theory predicts. If P4 does not retain, the misspecification term
(§6) is the first suspect — measure the frozen class's prediction error.

---

## 11. What still has to be proven or checked

1. **Misspecified PSRL bound** for the linear-Gaussian surrogate on categorical
   transitions (KL-projection vs `L∞` gap).
2. **Epoch theorem** (§8, option 2): per-epoch PSRL regret + switching cost, with
   doubling epochs.
3. **EMA bound** (§8, option 3), if that route is kept.
4. ~~**Empirical Markov check**~~ — **done 2026-09-26** (`src/markov_check.py`,
   results in `runlogs/markov_check.json`). The model's only history is the previous
   state (the candidate action is broadcast over the context). On the 18 checkpoints
   with working features, the predicted next state changes with the previous state in
   25–50% of (cell, action) pairs, and the one-step greedy action flips in 0–67% of
   cells (6 cells in DeepSea-5 have two predecessors). True DeepSea dynamics are Markov,
   so this history dependence is **spurious — it is prediction error, not information**.
   Option 1 loses nothing and removes an error source. The other 7 checkpoints had
   collapsed features (next item).
5. **Feature collapse (found during the Markov check).** Seed 1 at every checkpoint and
   seed 3 up to 500k have constant features (spread ~1e-7 against magnitude ~0.2, over
   the whole replay buffer); the BLR mean then predicts the next state *worse* than the
   majority class (10–11% vs 17–19%). Seed 3's features recovered between 500k and
   700k, matching its re-solve at ~520k. Checkpoints with healthy features (77–85%
   next-state accuracy) are *also* at the ~12% floor, with right−left reward margins of
   ~0.001 at non-goal cells.
6. ~~**Reward-signal check**~~ — **done 2026-09-26** (`src/reward_check.py`,
   `runlogs/reward_check.json`). On all 18 healthy checkpoints the BLR reward model is
   essentially exact: reward R² 0.99–1.00; at the goal transition r̂ = 0.73–0.76
   (target tanh(0.99) = 0.757) with a right−left margin of +0.57 to +0.77, kept above
   0.5 by 55–100% of posterior samples; at non-goal cells r̂(right) ≈ −0.002 and
   r̂(left) ≈ 0, i.e. it has learned the move cost. On the 7 collapsed checkpoints r̂ is
   a constant 0.01–0.05 everywhere (R² 0). **So the floor on healthy seeds is not a
   reward-model failure — it is a propagation failure.** The treasure signal exists
   only at the goal transition; with an inert value network, one-step greedy acting
   sees the move cost and prefers *left* at every other cell, so it reaches the goal
   only when sampling noise flips all four choices (consistent with, not yet proven to
   produce, the ~12% floor). This is exactly what exact planning (§4) supplies, which
   strengthens the proposal. It also suggests why early 99% phases happened: the report
   (2026-07-19) found a uniform +0.19 "right" bias in the reward column during a 99%
   phase, i.e. an *inaccurate* reward model; an accurate one is worse under one-step
   greedy acting.
7. **Citations:** every *(verify)* above.

---

## References *(verify all)*

- Osband, Russo & Van Roy (2013). (More) Efficient Reinforcement Learning via Posterior Sampling. NeurIPS.
- Osband & Van Roy (2014). Model-based Reinforcement Learning and the Eluder Dimension. NeurIPS.
- Osband & Van Roy (2017). Why is Posterior Sampling Better than Optimism for Reinforcement Learning? ICML.
- Ouyang, Gagrani, Nayyar & Jain (2017). Learning Unknown Markov Decision Processes: A Thompson Sampling Approach. NeurIPS.
- Kearns & Singh (2002). Near-Optimal Reinforcement Learning in Polynomial Time. Machine Learning.
- Agrawal & Goyal (2013). Thompson Sampling for Contextual Bandits with Linear Payoffs. ICML.
- Ayoub, Jia, Szepesvári, Wang & Yang (2020). Model-Based Reinforcement Learning with Value-Targeted Regression. ICML.
- Zanette, Brandfonbrener, Brunskill, Pirotta & Lazaric (2020). Frequentist Regret Bounds for Randomized Least-Squares Value Iteration. AISTATS.
- Sasso, Conserva & Rauber (2023). Posterior Sampling for Deep Reinforcement Learning. ICML.

## Links

- [[thompson-sampling]] — the PSRL loop this page makes exact
- [[neural-linear-head]] — the BLR posterior (Prop. 2)
- [[value-network]] — what this proposal removes
- [[dithering-trap]] — sampling-schedule failure modes (§7)
- [[representation-drift]] — why φ must be stationary (§8)
- [[deepsea]] — benchmark (§9)
