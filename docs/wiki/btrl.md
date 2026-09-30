# BTRL — Bayesian Transformer Reinforcement Learning

MSc Thesis project at Queen Mary University of London.
**Core claim:** replacing PSDRL's GRU transition model with a BayesFormer Transformer enables richer world-model learning in deep-exploration environments.

---

## Thesis Argument

[[psdrl|PSDRL]] uses a GRU recurrent network as its forward model, which:
- Processes transitions sequentially with a fixed hidden state
- Uses neural-linear Bayesian regression for uncertainty (last-layer only)

BTRL hypothesis: a [[bayesformer|BayesFormer]] Transformer over a context window should:
1. Better capture long-range dependencies in trajectories
2. Provide richer, multi-layer Bayesian uncertainty via dropout
3. Enable more effective [[thompson-sampling|Thompson Sampling]] in hard exploration tasks

---

## Architecture

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

### Networks

| Network | Type | Purpose |
|---|---|---|
| [[transition-network\|TransitionNetwork]] | BayesFormer Transformer | Predicts next state + reward from context |
| ValueNetwork | MLP | Q-function approximation under sampled M̂ |
| TerminalNetwork | MLP | Predicts episode termination |

### Module Map

```
src/TRL/
├─ agent/btrl.py                     # top-level orchestrator
├─ bayes/transformer_rl_agent.py     # owns episode state + trajectory
├─ networks/transition.py            # BayesFormer architecture
├─ networks/value.py                 # MLP value network
├─ networks/terminal.py              # MLP terminal predictor
├─ training/transition.py            # TransitionModelTrainer
├─ training/policy_btrl.py           # PolicyTrainer
└─ common/
    ├─ replay.py                     # Dataset / replay buffer
    ├─ utils.py                      # env wrappers, loss functions
    ├─ logger.py                     # TensorBoard logging
    └─ settings.py                   # global loss/optim constants
```

---

## Thompson Sampling Loop

```python
# Episode start (step == 1)
agent.model.sample_model()           # locks all dropout masks → samples M̂

# Every timestep
for action in possible_actions:
    next_state, reward = transition_network.predict(context, action)

policy = argmax_action(value_network(next_state))
obs, reward = env.step(policy)
trajectory.append((obs, policy))     # update context window
```

The locked dropout mask remains constant for the entire episode. This is what makes it [[thompson-sampling|Thompson Sampling]] rather than step-wise randomised execution.

---

## Critical Invariant: [[episode-locked-dropout|Episode-Locked Dropout]]

> The dropout mask must be sampled ONCE at episode start (step == 1) and held FIXED for the entire episode.

Two failure modes arise without this:
- [[dithering-trap]] — the imagined world flips every step; no coordinated exploration
- [[representation-drift]] — non-stationary H^a features prevent value network convergence

---

## Experiment: DeepSea-5

Environment: [[deepsea|DeepSea]] depth=5 (25-state grid, 0.99 reward only at bottom-right).
Config: `config_vector-setting1.yaml`

Key hyperparameters:
```yaml
transition:
  hidden_dim: 32-512    # swept via Optuna
  context_length: 2
  num_encoder_layers: 2
  heads: 2
  forward_expansion: 4
  dropout: 0.2
  obs_type: "grid"
```

Evaluation metric: `mean_grid_dist` — Euclidean distance in (row, col) space between predicted and true next state (handles row-boundary artefacts that flat-index distance misses).

---

## Links

- [[psdrl]] — baseline being extended
- [[bayesformer]] — source of the BayesFormer architecture
- [[transition-network]] — the main network
- [[episode-locked-dropout]] — the critical design constraint
- [[dithering-trap]] — integration failure #1
- [[representation-drift]] — integration failure #2
- [[loss-functions]] — environment-aware training loss
- [[deepsea]] — test environment
- [[dissertation-feedback]] — examiner comments
