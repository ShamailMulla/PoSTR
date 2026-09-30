# BTRL Hyperparameter Optimisation — Design Report

This report documents the three-stage Optuna pipeline for tuning BTRL's two
learned networks — the Transformer **transition network** (world model) and the
MLP **value network** — on deterministic DeepSea-5: which parameters are tuned,
what each represents, why it is in (or out of) the search space, what the
objectives measure, and how to read the Pareto fronts.

Pipeline (one notebook per stage, all in `src/`):

| Stage | Notebook | What runs | Objective(s) |
|---|---|---|---|
| 1 | `hpo1_transition_search.ipynb` | offline: train/evaluate the transition network on a fixed dataset | 2-objective Pareto: mean grid distance ↓ × overconfident-error rate ↓ |
| 2 | `hpo2_validation.ipynb` | online: full agent runs of the Stage-1 Pareto candidates | validation only — correlates offline metrics with cumulative regret |
| 3 | `hpo3_value_search.ipynb` | online: full agent runs with the winning transition config frozen | single objective: seed-averaged cumulative regret ↓ |

All studies persist to one SQLite database (`src/hpo_artifacts/optuna_btrl.db`)
and pass results between notebooks through files in `src/hpo_artifacts/`, so
each notebook is independently re-runnable and resumable after interruption.
Shared logic lives in `src/TRL/tuning/` (search spaces, trial training, the
full-agent evaluation harness) so the validation notebook and the value search
measure downstream performance with *identical* code.

---

## 1. Transition network (Stage 1)

### Why offline, and why these two objectives

Each trial trains on the **same fixed dataset** of 2,000 random-policy
transitions and is evaluated on a held-out per-state test set. No environment
interaction per trial means (a) trials are cheap (~seconds–minutes, so NSGA-II
can afford 200 of them), (b) the score is low-noise, and (c) differences
between trials are attributable to the transition hyperparameters alone —
whereas a full-agent score would confound the world model with the (untuned)
value network and seed luck.

**Objective 1 — `mean_grid_dist` (minimise).** DeepSea states are flat indices
into an N×N grid (`state s → row s//N, col s%N`). For each (state, action)
test pair, 100 world hypotheses M̂ are sampled by locking the BayesFormer
dropout masks (`lock_all_dropouts`, exactly the Thompson-Sampling mechanism the
agent uses), and the metric is the mean Euclidean distance in (row, col) space
between each hypothesis's predicted next state and the true one. Grid distance
is used instead of flat-index error because flat indices are misleading at row
boundaries (indices 4 and 5 differ by 1 but sit at opposite ends of adjacent
rows). This measures **world-model accuracy**.

**Objective 2 — `overconfident_error_rate` (minimise).** The fraction of
(state, action) pairs where the posterior is simultaneously *wrong*
(mean grid distance > 0.5 cells) and *near-certain* (variance of the sampled
predictions < 0.05). This measures **posterior calibration**, which accuracy
alone cannot see — and it is the failure mode that specifically breaks
Thompson Sampling: if every sampled M̂ agrees on the same wrong transition,
no episode will ever act under a hypothesis that explores the correction. A
model that is slightly less accurate but *knows where it is wrong* is more
valuable to BTRL than a confidently wrong one.

### Parameters tuned

| Parameter | Range / choices | What it represents | Why tune it |
|---|---|---|---|
| `hidden_dim` × `heads` | 20 valid pairs: dim ∈ {32, 64, 128, 256, 512}, heads ∈ {1, 2, 4, 8}, dim % heads = 0 | embedding width and number of attention heads — the model's capacity and how many distinct relations attention can track | capacity governs the accuracy/overfitting trade-off on only 2,000 transitions; sampled as a joint index so `head_dim = dim/heads` is always integral |
| `learning_rate` | log-uniform [1e-4, 1e-2] | Adam step size | interacts strongly with width/depth; log scale because its effect is multiplicative |
| `num_encoder_layers` | 1–4 | Transformer depth (stacked encoder blocks) | DeepSea dynamics are simple; deeper may only add variance — worth testing rather than assuming |
| `forward_expansion` | {2, 4, 8} | FFN width multiplier inside each block | second capacity axis, partially interchangeable with `hidden_dim` |
| `p_seq` | {0.05, 0.1, 0.2, 0.3} | dropout on state/action/positional embeddings | **not a regulariser here** — in the BayesFormer construction the dropout masks *are* the posterior over world models; `p_seq` shapes the observation-noise component |
| `p_attn` | {0.02, 0.05, 0.1, 0.15} | dropout on Q/K/V projections, attention maps, and FFN skip connections | shapes the epistemic-uncertainty component of the posterior — directly controls how diverse the sampled M̂ are, i.e. the breadth of Thompson-Sampling exploration |

`p_seq`/`p_attn` are the reason Objective 2 exists: they trade prediction
sharpness against posterior diversity, and the Pareto front makes that
trade-off visible instead of collapsing it into one number.

**Fixed (not tuned):** `context_length = 2` (changes the replay-buffer and
agent tensor shapes — a structural choice, kept fixed for comparability),
`obs_type = "grid"`, and the training-loop settings of the *online* agent
(`training_iterations`, `window_length`), which belong to the agent's update
schedule rather than the network.

### Reading the transition Pareto front

With two competing objectives there is generally no single best trial: making
predictions sharper (lower grid distance) typically means lower dropout and a
tighter posterior, which raises the risk of confident errors — and vice versa.
A trial is **Pareto-optimal** if no other trial is at least as good on both
objectives and strictly better on one. The set of such trials is the Pareto
front (starred in the notebook's scatter plot: x = grid distance,
y = overconfident-error rate, colour = exact-match accuracy).

- **Bottom-left corner** is ideal: accurate *and* calibrated.
- Points on the front but far along one axis are specialists: e.g. very
  accurate but overconfident (dangerous for exploration), or well-calibrated
  but sloppy (safe but slow to plan with).
- Dominated points (inside the cloud) are strictly worse than some front
  point and are discarded.

NSGA-II is used because it maintains a diverse population along the front
rather than converging to one scalarised optimum. The front (not a single
winner) is exported to `transition_pareto.json` — choosing among front members
is exactly what Stage 2's online validation is for.

---

## 2. Validation stage (Stage 2) — the objective-mismatch check

Model-based RL has a documented failure mode (**objective mismatch**, Lambert
et al. 2020): prediction accuracy does not always predict control performance.
Stage 2 therefore takes the top Pareto candidates, plus the hand-set baseline
config, and runs each through the **full BTRL agent** (3 seeds × 10,000 steps,
value network fixed at the base config for all candidates so the comparison
isolates the transition model). It reports:

- Spearman rank correlations (with scatter plots) between each offline metric
  and online cumulative regret. Strong positive correlation validates the
  offline objectives as proxies — a defensible thesis figure. Weak or negative
  correlation is itself a finding: transition tuning would then need online
  signal.
- The winner by mean cumulative regret, frozen to
  `best_transition_config.yaml` for Stage 3.

**Why goal-reaching is *not* a Stage-1 objective.** Whether the agent reaches
the end state depends on the whole system (world model + value network +
Thompson-Sampling dynamics + seed), so inside the transition study it would be
a high-variance, confounded signal — most trials would score zero regardless
of model quality, flattening the search landscape. It enters here instead,
where it can be measured properly (multi-seed full runs) and attributed
(everything but the transition params is held fixed).

**On "steps required to reach the end state".** In DeepSea an episode always
lasts exactly N steps — the agent descends one row per step — so
within-episode path length is a constant and cannot be minimised. The quantity
that actually varies is how much *training experience* is needed before the
agent reliably reaches the goal. That is sample efficiency, and it is exactly
what the online metrics measure:

- `cumulative_regret` = Σ (optimal return − episode return) over the training
  budget. Its slope is the current performance gap; the earlier the curve
  flattens, the more sample-efficient the agent. This is the primary online
  metric (the PSDRL literature reports regret for the same reason).
- `first_goal_episode` — first episode in which the goal was reached at all
  (exploration success).
- `episodes_to_solve` — first episode where the trailing 20-episode mean
  return ≥ 90% of optimal (reliable solving, censored at the budget if never).
- `solved_rate_last_100` — end-of-training stability.

Regret already integrates "ability to reach the end state" and "speed of
getting there", so a separate steps-to-goal objective would be redundant.

---

## 3. Value network (Stage 3)

### Why online, and why a single objective rather than a Pareto front

The value network's training targets are generated by simulating the *sampled*
world model on replayed states, and its outputs feed straight back into action
selection — its quality is only meaningful inside the closed loop. So each
trial runs the full agent (3 seeds × 10,000 steps) with the transition network
frozen at the validated winner.

Unlike Stage 1, there is no second objective in genuine tension with the
first: cumulative regret already subsumes goal-reaching ability, learning
speed, and stability (candidates like `episodes_to_solve` are near-monotone
functions of regret in DeepSea, and a supervised proxy like held-out TD loss
would reintroduce the objective-mismatch problem one level down). A Pareto
front over strongly correlated objectives is a thin diagonal band that adds
trials without adding information. Single-objective TPE is therefore used —
which also unlocks Optuna's hyperparameter-importance analysis (fANOVA),
plotted in the notebook. `episodes_to_solve`, `first_goal_episode`, and
`solved_fraction` are still recorded per trial as user attributes for
inspection.

(In short: the transition network gets a Pareto front because accuracy and
calibration genuinely compete; the value network does not, because its
candidate objectives all point the same way.)

### Parameters tuned

| Parameter | Choices | What it represents | Why tune it |
|---|---|---|---|
| `training_iterations` (κ) | {5, 10, 20, 50} | value-net gradient passes per agent update call | controls how hard the value function chases each newly sampled M̂ — too low and values lag the sampled model, too high and they overfit one hypothesis (representation-drift sensitivity) |
| `hidden_layers` | {1, 2, 3, 4} | MLP depth | the 25-dim one-hot input is simple, but the bootstrap target is non-stationary; depth trades expressiveness against optimisation stability. (Note: `value.py` previously ignored this key and always built 2 hidden layers — it is now respected, and existing configs were updated to `2` to preserve the old architecture.) |
| `hidden_dim` | {8, 16, 32, 64, 128, 256} | MLP width | main capacity axis |
| `learning_rate` | log-uniform [1e-4, 1e-2] | Adam step size | interacts with κ and `target_update_freq` |
| `target_update_freq` | {5, 10, 20, 50} | steps between target-network snapshots | bootstrap-target stability: frequent updates track the sampled model faster but risk divergence; infrequent updates are stable but slow |

### Excluded: `discount` (γ)

Deliberately not tuned. DeepSea is a fixed-length N-step chain with a single
terminal reward: every bootstrapped value is scaled by γ^(steps-to-go), a
factor that is *the same monotone transformation for all actions compared at a
given depth*, so changing γ cannot reorder the greedy argmax in
`select_action` — it only rescales the value function the network must fit.
Tuning it would spend trials on a parameter that cannot change behaviour
(while slightly changing the regression problem's conditioning, a nuisance
effect). It stays at the base-config value of 0.95.

### Final held-out comparison

The notebook ends by running the tuned config and the base config on **fresh
seeds** (10–12, never used during search) and plotting cumulative-regret
curves — the summary figure for the thesis: flatter curve = solved, earlier
bend = more sample-efficient.

---

## 4. Reproducibility and budgets

- **Storage**: `sqlite:///src/hpo_artifacts/optuna_btrl.db`, studies
  `btrl_transition_hpo` (2 directions) and `btrl_value_hpo` (1 direction),
  both created with `load_if_exists=True` — interrupt and re-run freely; the
  notebooks only run the *remaining* trials up to `N_TRIALS`.
- **Seeds**: search data and samplers are seeded (42); every trial in the
  online stages averages over the same seed set {0, 1, 2}; the final
  comparison uses held-out seeds {10, 11, 12}.
- **No pruner**: multi-objective studies do not support pruning, and in the
  online stage multi-seed regret is only comparable at full budget.
- **Measured cost** (GPU, torch2_rl env): ~1.4 min per 10k-step seed →
  Stage 2 ≈ 25 min total; Stage 3 ≈ 3.5 h for 50 trials. Stage 1 trials are
  seconds–minutes each. Budget knobs are the constants at the top of each
  notebook (`N_TRIALS`, `SEEDS`, `TOTAL_STEPS`, `DATA_STEPS`).
- Notebooks are regenerable from `src/generate_hpo_notebooks.py`.
