# Optuna Search Objectives

Two objectives for the [[transition-network|transition model]] hyperparameter search, both minimised via NSGA-II Pareto-front optimisation.

---

## Objective 1 — `mean_grid_dist` (World Model Accuracy)

**What it measures**: Mean Euclidean distance in (row, col) grid space between the transition model's predicted next state and the true next state, averaged across all reachable test (state, action) pairs and across N_HYPOTHESES=100 sampled world hypotheses M̂.

**Why it aligns with the thesis**: BTRL's exploration mechanism is [[thompson-sampling|Thompson Sampling]] — at episode start, one M̂ is locked and the agent acts greedily under it. For this to drive correct exploration (e.g., always-right in DeepSea), the locked world model must accurately predict the consequences of actions. `mean_grid_dist` directly measures this: a model that gets the spatial transitions right will produce coherent locked hypotheses that plan good trajectories.

**What it does NOT capture**: Whether the model's *uncertainty* (spread across M̂) is appropriate. A model can have low `mean_grid_dist` but near-zero inter-hypothesis variance (all M̂ predict the same thing), which would make Thompson Sampling degenerate to a greedy policy. `mean_grid_variance` in the results table tracks this separately.

**Evaluated with**: locked masks only — `test_model_locked()` with `n_hypotheses=100`.

---

## Objective 2 — `env_steps_to_accuracy` (Sample Efficiency)

**What it measures**: The minimum number of environment transitions (collected steps) the model needs to achieve `mean_grid_dist < TARGET_GRID_DIST` (default 0.5 cells). Implemented by training fresh model instances on progressively larger subsets of the collected transitions (`PROBE_SIZES = [250, 500]`), then falling back to `TIME_LIMIT = 1000` if neither probe hits the threshold.

**Why it matters for the thesis**: The central claim of BTRL over PSDRL is *sample efficiency* — the agent should need fewer real environment interactions to learn an accurate world model and start exploring effectively. A hyperparameter configuration that achieves the same final accuracy on fewer transitions is strictly better for this claim.

**What it is NOT**: Total RL episode steps to first reward (which would require running the full agent-environment loop). That metric — the true end-to-end sample efficiency — belongs in the main evaluation of `run_experiments.py`, not the transition model search. `env_steps_to_accuracy` is a proxy: transition model sample efficiency is a necessary (though not sufficient) condition for overall RL sample efficiency.

**Threshold**: `TARGET_GRID_DIST = 0.5` cells — within half a grid cell on average. In a 5×5 DeepSea grid this means the model is essentially pointing at the correct next state for most transitions.

---

## Why These Two Together

| Property | Captured by |
|---|---|
| World model accuracy | Obj 1: `mean_grid_dist` ↓ |
| Sample efficiency | Obj 2: `env_steps_to_accuracy` ↓ |
| Posterior diversity | `mean_grid_variance` (user attr, not objective) |
| Convergence speed | `epochs_to_converge` (user attr, not objective) |

The Pareto front reveals the trade-off: small models converge quickly on less data but may plateau at higher error; large models reach lower error but need more transitions. The thesis argument is that BayesFormer hyperparameters on the Pareto front should sit closer to the accuracy-and-efficiency corner than equivalent GRU configurations.

---

## Links

- [[transition-network]] — the model being optimised
- [[thompson-sampling]] — why world model accuracy enables exploration
- [[deepsea]] — the evaluation environment, state encoding, reachable states
- [[loss-functions]] — `compute_state_loss` used during training
- [[learnings]] — session log
