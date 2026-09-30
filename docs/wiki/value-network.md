# Value Network

The MLP that estimates state value `V(s)` over the currently sampled world model.
It is the **critic half** of BTRL's model-based control loop (the
[[transition-network]] is the world model, not an actor — see [[btrl]]).

> **Status (2026-09):** inert in every v3+ arm — its output is constant across all
> states, including while the agent solves DeepSea at 99% (report 2026-07-19; Sep 2026
> rerun traces). The operative policy is the BLR reward margin. See
> [[planning-in-sampled-model]] for the proposal to replace it with exact planning.

## Architecture (`TRL/networks/value.py`)

`Flatten → Linear(input_dim, hidden_dim) → Tanh → [Linear(hidden_dim, hidden_dim) → Tanh] × hidden_layers → Linear(hidden_dim, 1)`

- `input_dim = env_dim = 25` for DeepSea-5 (the state vector; no autoencoder — see
  [[psdrl-autoencoder-analysis]]).
- HPO-selected for DeepSea (see [[optuna-objectives]]): `hidden_dim = 16`,
  `hidden_layers = 3`, `learning_rate = 2.7e-3`, target refreshed every 50 steps.

## Role in the loop

- **Action selection** (`btrl.select_action`): for each action the model predicts
  the next state `ŝ'`; the agent picks `argmax_a [ r̂_a + γ · V(ŝ'_a) · (not terminal) ]`.
  So the value net is queried on **predicted next states**.
- **Training** (`TRL/training/policy_btrl.py`): value targets are **simulated from
  the sampled world model** (not from stored transitions). The Bellman backup is
  `V(s) ← max_a[r + γ V_target(s'_a)]` — the value-net input is the **current**
  state, matching PSDRL's `PSDRL/training/policy.py`; a periodically-refreshed
  target network stabilises the bootstrap.

## Links

- [[planning-in-sampled-model]] — proposed replacement

- [[transition-network]] — the world model whose samples feed the value targets
- [[psdrl]] — reference implementation of the value trainer
- [[optuna-objectives]] — how the value HPs were searched
