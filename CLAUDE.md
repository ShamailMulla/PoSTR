# BTRL — Bayesian Transformer Reinforcement Learning

## Research Context

BTRL is a novel model-based RL agent that extends **PSDRL** (Posterior Sampling for Deep RL) by replacing its linear Bayesian transition model with a **BayesFormer-style Transformer**. The core thesis: Transformers' self-attention over trajectory context should enable better world-model learning in sparse-reward, deep-exploration tasks (e.g., DeepSea from bsuite).

The exploration mechanism is **Thompson Sampling**: at the start of each episode, the agent samples one concrete hypothesis of the world (M̂) and acts greedily under it for the entire episode. This episodic commitment is what enables deep, coordinated exploration strategies.

---

## Execution Flow

```
run_experiments.py          # loops N seeds, calls main.main(config)
└─ main.py:main()
   ├─ init_env()             # creates bsuite DeepSea environment
   ├─ BTRL(config, ...)      # builds all networks and trainers
   └─ run_experiment()
      └─ episode loop
         ├─ agent.select_action(obs, step)   # step==1: sample_model()
         ├─ env_step()
         ├─ agent.update(...)                # every update_freq steps: train
         │   ├─ transition_trainer.train_()
         │   └─ policy_trainer.train_(model, dataset)
         └─ repeat
```

---

## Module Map

```
src/
├─ main.py                      # top-level training loop
├─ run_experiments.py           # multi-seed runner
├─ configs/                     # YAML experiment configs
└─ TRL/
   ├─ agent/btrl.py             # BTRL — top-level agent, orchestrates everything
   ├─ bayes/transformer_rl_agent.py  # BayesianTransformer — owns episode state
   ├─ networks/
   │   ├─ transition.py         # Transformer transition model (main architecture)
   │   ├─ value.py              # MLP value network
   │   └─ terminal.py          # MLP terminal state predictor
   ├─ training/
   │   ├─ transition.py         # TransitionModelTrainer
   │   └─ policy_btrl.py       # PolicyTrainer (value network)
   └─ common/
       ├─ replay.py             # Dataset / replay buffer (+ protected reward-bearing ring, see below)
       ├─ utils.py              # env wrappers, preprocessing
       ├─ logger.py             # TensorBoard logging
       ├─ data_manager.py       # checkpointing
       └─ settings.py           # global loss/optim constants
```

---

## Architecture

### Transition Network (`networks/transition.py`)

Encoder-only Transformer. Input: sequence of (timestep, state, action) tuples of length `context_length`. Output: predicted next state and return-to-go.

**Key classes:**
- `EpisodeLockedDropout` — custom dropout that can be frozen for an episode (see below)
- `MultiHeadAttention` — Q/K/V projections with per-element dropout
- `TransformerBlock` — attention + LayerNorm + FFN + skip connections
- `Encoder` — embeds states/actions/timesteps, stacks them, runs through TransformerBlocks
- `Network` — wraps Encoder with prediction heads; exposes `lock_all_dropouts` / `unlock_all_dropouts`

### BayesianTransformer (`bayes/transformer_rl_agent.py`)

Owns episode-level state: `self.trajectory` (context window of states) and `self.time_transition` (timestep context). Calls `transition_network.forward()` to predict next states for all actions in parallel.

**Key method:**
- `sample_model()` — locks all dropout masks in the transition network. This IS the Thompson Sampling "sample M̂" step. Called by BTRL at episode start (step == 1).

### BTRL (`agent/btrl.py`)

Top-level orchestrator. Instantiates all networks and trainers. Calls `model.sample_model()` at the start of each episode to lock the world hypothesis.

---

## Critical Invariant: Episode-Locked Dropout

**The most important design constraint in this codebase.**

The BayesFormer Transformer uses dropout masks as approximate Bayesian inference — each unique mask configuration represents one sample from the posterior over world models. For Thompson Sampling to work correctly:

> The dropout mask must be sampled ONCE at episode start (step == 1) and held FIXED for the entire episode.

Violating this causes two cascading failures:
1. **Dithering Trap** — the agent's imagined world flips every step; it cannot commit to a consistent exploration strategy (e.g., always-right in DeepSea).
2. **Representation Drift** — the hidden states H^a feeding the value network change on every call; the value network's training target is non-stationary and it cannot converge.

### `EpisodeLockedDropout`

All 7 dropout sites in `transition.py` use this class instead of `nn.Dropout`:

| Site | Instance name | Class |
|---|---|---|
| State embedding dropout | `enc.state_dropout` | `Encoder` |
| Positional encoding dropout | `enc.pos_dropout` | `Encoder` |
| Action embedding dropout | `enc.action_dropout` | `Encoder` |
| Q projection dropout | `attn.q_dropout` | `MultiHeadAttention` |
| K projection dropout | `attn.k_dropout` | `MultiHeadAttention` |
| V projection dropout | `attn.v_dropout` | `MultiHeadAttention` |
| Attention map dropout | `attn.attn_map_dropout` | `MultiHeadAttention` |
| FFN skip-connection dropout | `block.dropout` | `TransformerBlock` |

**Lock shapes** (N = num_actions, L = context_length):
- Encoder embedding masks: `(N, L, embed_size)`
- Q/K/V projection masks: `(N, 2L, heads, head_dim)` — `2L` because states+actions are interleaved
- Attention map mask: `(N, heads, 2L, 2L)`
- FFN skip mask: `(N, 2L, embed_size)`

**Stochastic fallback**: `EpisodeLockedDropout` checks `mask.shape == x.shape` before applying the locked mask. Training calls use different batch dimensions (N=100, L=10 from config) than rollout (N=2, L=2), so they always fall through to standard stochastic dropout. No explicit unlock needed for training.

---

## Protected Replay Partition (v4 fix)

`replay.protected_fraction` (default 0.0 = original behaviour) reserves that fraction of `replay.capacity` as a separate FIFO ring holding **copies** of reward-bearing episodes (any stored tanh reward > 0.5). Rationale: age-only eviction can purge goal transitions when a performance dip outlasts one buffer turnover. The ring guarantees a presence floor *and* a sampling-share floor for goal data; `sample_sequences` draws uniformly over the union of both rings. Copy-not-divert keeps the main ring an unbiased picture of recent experience. Logged as `Data/Protected Episodes` / `Data/Protected Transitions`. The ring is a correct improvement that keeps the buffer honest; version-history context is in `../wiki/failure-modes.md`.

---

## v3 — Neural-Linear Head (posterior over world models)

`algorithm.neural_linear: true` replaces BayesFormer dropout-as-posterior with a
Bayesian linear regression head (`TRL/bayes/neural_linear_head.py`) on the
transformer's penultimate features (`transition.features()` = `predict_state[:-1]`,
dim `hidden_dim`). Motivation: 5/6 v2.3 seeds collapsed even with goal data
guaranteed — the dropout posterior can't self-widen, so retention wasn't enough
(see `../wiki/failure-modes.md`, `../wiki/neural-linear-head.md`). The BLR precision
`Λ = βΦᵀΦ + αI` is data-built, so uncertainty is high where data is scarce and
shrinks as it accumulates (verified: sampling spread shrinks with replay size).

Flow: `sample_model()` → `head.sample()` (draw W per episode, true-posterior
Thompson sampling with `transition_temp=reward_temp=1`); `predict()` →
`head.predict()` (`W·Φ`); `btrl.update()` calls `head.update_posteriors(dataset)`
after the gradient stage and before value training. Dropout is demoted to `p=0`.
Config: `configs/config_v3_neural_linear.yaml` (note `protected_fraction: 0` — v3
tests whether the BLR self-corrects *without* the buffer fix). `neural_linear`
absent/false → unchanged v1/v2 dropout behaviour.

---

## Hyperparameter Tuning (Optuna)

Three-stage pipeline, one notebook per stage (design rationale in `HPO_REPORT.md`):

1. `src/hpo1_transition_search.ipynb` — offline 2-objective NSGA-II search over the transition network (mean grid distance × overconfident-error rate).
2. `src/hpo2_validation.ipynb` — full-agent runs of the Pareto candidates; correlates offline metrics with cumulative regret and freezes the winner.
3. `src/hpo3_value_search.ipynb` — online single-objective TPE search over the value network (seed-averaged cumulative regret; `discount` deliberately excluded).

Shared logic: `src/TRL/tuning/` (search spaces, trial training, full-agent evaluation harness). Studies persist to `src/hpo_artifacts/optuna_btrl.db` (resumable, `load_if_exists=True`); stage outputs pass through files in `src/hpo_artifacts/`. Run in order; notebooks target the `torch2_rl` kernel. Regenerate notebooks with `python3 src/generate_hpo_notebooks.py`. Note: `TRL/networks/value.py` now respects `value.hidden_layers` (previously ignored, always 2).

---

## How to Run

```bash
cd BTRL/src

# Single run
python main.py --config configs/config_vector-setting1.yaml

# Multi-seed experiment
python run_experiments.py

# Monitor training
tensorboard --logdir logdir
```

**Config files:**
- `config_vector_env3-setting1.yaml` — DeepSea depth=3, deterministic, setting 1
- `config_vector_env3-setting2.yaml` — DeepSea depth=3, deterministic, setting 2
- `config_btrl_vector_env5.yaml` — DeepSea depth=5
- `config_vector_test-deterministic.yaml` / `config_vector_test-stochastic.yaml` — test configs

---

## Known Issues

See `../integration_problem.txt` for full problem descriptions.

**Problem #1 — Dithering Trap** (FIXED by EpisodeLockedDropout): Step-wise mask resampling during rollout destroyed Thompson Sampling consistency.

**Problem #2 — Representation Drift** (FIXED by EpisodeLockedDropout): nn.Dropout's per-call resampling caused non-stationary H^a features, making the value network's training target a moving target.

Both problems are resolved by the same `EpisodeLockedDropout` mechanism. The fix was applied to all dropout sites in `transition.py`, with `BTRL.select_action()` triggering `model.sample_model()` at step == 1 of each episode.