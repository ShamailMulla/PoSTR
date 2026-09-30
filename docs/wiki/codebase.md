# Codebase Structure

Code for the [[btrl|BTRL]] thesis project. Root: `BTRL/src/`

---

## Module Map

```
BTRL/src/
├── main.py                          # top-level: main(config) → run_experiment()
├── run_experiments.py               # loops N seeds → main.main(config)
├── notebook_utils.py                # shared training/eval for notebooks
├── generate_optuna_notebook.py      # script to regenerate Optuna notebook
│
├── configs/
│   ├── config_vector-setting1.yaml  # DeepSea-5, main config
│   ├── config_vector-setting2.yaml  # DeepSea-5, smaller model
│   ├── config_btrl_vector_env5.yaml # DeepSea-5, larger model
│   ├── config_psdrl_vector1.yaml    # PSDRL baseline config
│   ├── config_psdrl_vector2.yaml    # PSDRL baseline config (large)
│   ├── config_vector_test-deterministic.yaml
│   └── config_vector_test-stochastic.yaml
│
└── TRL/
    ├── agent/
    │   └── btrl.py                  # BTRL — top-level agent orchestrator
    ├── bayes/
    │   └── transformer_rl_agent.py  # BayesianTransformer — episode state + trajectory
    ├── networks/
    │   ├── transition.py            # BayesFormer Transformer (main architecture)
    │   ├── value.py                 # MLP value network
    │   └── terminal.py             # MLP terminal state predictor
    ├── training/
    │   ├── transition.py            # TransitionModelTrainer (windowed training)
    │   └── policy_btrl.py           # PolicyTrainer (value network)
    └── common/
        ├── replay.py                # Dataset + replay buffer
        ├── utils.py                 # env wrappers, compute_state_loss()
        ├── settings.py              # global constants (TM_REWARD_LOSS_F, etc.)
        ├── logger.py                # TensorBoard logging
        └── data_manager.py          # checkpointing
```

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `test1_transition_model-deterministic-lock_dropouts.ipynb` | Manual ablation: hidden_dim sweep, locked vs unlocked dropout |
| `optuna_transition_hparam_search.ipynb` | Optuna-based hyperparameter optimisation (NSGA-II Pareto) |
| `Untitled.ipynb` | Scratch notebook |

---

## Execution Flow

```
main.py: main(config)
  └── init_env()                          → env, possible_actions, obs_space
  └── BTRL(config, env, obs_space)
        ├── TransitionNetwork(config['transition'], obs_space, ...)
        ├── TransitionModelTrainer(config['transition'], transition_net, ...)
        ├── BayesianTransformer(transition_net, obs_space, ...)
        ├── TerminalNetwork(...)
        └── ValueNetwork(...)
  └── run_experiment(agent, env, config)
        └── episode loop:
              t=0: agent.model.sample_model()     # → lock_all_dropouts()
              for each step:
                a = agent.select_action(obs, step)
                obs, r, done = env_step(env, a)
                agent.update(obs, a, r, done, step)  # every update_freq steps
```

---

## Key Functions

| Function | File | Purpose |
|---|---|---|
| `compute_state_loss()` | `common/utils.py` | Environment-aware state loss dispatch |
| `lock_all_dropouts()` | `networks/transition.py` | Sample and freeze all 8 dropout masks |
| `unlock_all_dropouts()` | `networks/transition.py` | Return to stochastic training mode |
| `sample_model()` | `bayes/transformer_rl_agent.py` | Thompson Sampling: lock masks at episode start |
| `early_stopping_train()` | `notebook_utils.py` | Training loop with patience for notebooks |
| `generate_transition_data()` | `notebook_utils.py` | Data generation from bsuite env |
| `test_model_locked()` | `notebook_utils.py` | Evaluate with locked world hypotheses |
| `test_model_unlocked()` | `notebook_utils.py` | Evaluate with stochastic masks (broken mode) |

---

## Config Schema

Key fields under `transition:`:

```yaml
transition:
  obs_type: "grid"           # NEW: loss function dispatch
  hidden_dim: 32             # Transformer embedding dimension
  context_length: 2          # number of (s, a) pairs in context window
  num_encoder_layers: 2      # number of TransformerBlocks
  heads: 2                   # number of attention heads (must divide hidden_dim)
  forward_expansion: 4       # FFN expansion ratio
  learning_rate: 1e-2
  training_iterations: 20    # gradient steps per update call
  window_length: 3           # TBPTT window size
```

---

## How to Run

```bash
cd BTRL/src

# Single seed
python main.py --config configs/config_vector-setting1.yaml

# Multi-seed experiment
python run_experiments.py

# Monitor
tensorboard --logdir logdir
```

---

## Links

- [[transition-network]] — detailed architecture docs
- [[episode-locked-dropout]] — the custom dropout class
- [[loss-functions]] — the state loss implementation
- [[btrl]] — agent-level architecture
