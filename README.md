# PoSTR — Posterior Sampling with Transformers for RL

MSc thesis project (Queen Mary University of London). PoSTR (called BTRL in the code:
`TRL/`, `BTRL` agent class) extends [PSDRL](https://github.com/remosasso/PSDRL) —
Posterior Sampling for Deep RL — by replacing its GRU transition model with a
Transformer world model, with Thompson sampling over world models for deep
exploration. Evaluated on bsuite DeepSea.

This repository is a fork of PSDRL. Upstream PSDRL, with the thesis's changes for
DeepSea, lives in [`baselines/psdrl/`](baselines/psdrl/) and is the baseline.

> **Branches.** `main` holds the code as of **30 July 2026** (thesis state). This is the
> **`development`** branch: the fixes listed under
> [Issues fixed on this branch](#issues-fixed-on-this-branch), standard randomized
> DeepSea for both agents, MLflow tracking, and the current-setup hyperparameter search.

## Layout

| Path | What it is |
|---|---|
| `src/main.py`, `src/run_*.py` | PoSTR entry points (`run_v3_1M.py` runs one seed of the neural-linear agent for 1M steps) |
| `src/TRL/` | the PoSTR package: `agent/btrl.py` (agent), `networks/transition.py` (Transformer world model, BayesFormer dropout), `bayes/neural_linear_head.py` (Bayesian linear regression posterior over transformer features, v3+), `training/` (transition and value training), `common/` (replay, env wrappers, utils), `tuning/` (hyperparameter-search code) |
| `src/configs/` | experiment configs (`config_v3_neural_linear.yaml` is the main PoSTR config; v5/v6 are the ring-buffer and frozen-feature ablations) |
| `src/hpo1_transition_search_blr.ipynb` | **current** stage-1 search: transition network + BLR posterior on randomized DeepSea, every trial logged to MLflow |
| `src/hpo1_transition_search.ipynb` | the July 2026 stage-1 search (locked-dropout posterior, non-randomized DeepSea); `generate_hpo_notebooks.py` regenerates it and stages 2–3 |
| `src/run_comparison.sh`, `src/mlflow_sync.py` | 5-seed PoSTR vs PSDRL runs on randomized DeepSea, streamed into MLflow (parent run per arm, one child per seed) |
| `src/TRL/common/tracing.py` | sampled MLflow tracing of update cycles (1 in `TRACE_EVERY`) |
| `src/markov_check.py`, `src/reward_check.py` | checkpoint analyses: history dependence of predictions; reward-model signal at the goal |
| `src/optuna_to_mlflow.py` | copies an Optuna study into MLflow, or logs trials live (`MLflowTrialLogger`) |
| `src/optuna_*_hparam_search.ipynb` | earlier single-notebook Optuna searches (transition, value) |
| `src/*.ipynb`, `src/TRL/tuning/test1_*.ipynb` | transition-model experiments from the original thesis submission |
| `baselines/psdrl/` | PSDRL baseline: upstream code plus a bsuite DeepSea / MemoryChain wrapper, DeepSea configs and Optuna notebooks (`param_tuning/`) |
| `docs/` | `RANDOMIZED_COMPARISON_REPORT.md` (5-seed PoSTR vs PSDRL on randomized DeepSea, Sep 2026 — supersedes the July comparison), `HPO_REPORT.md`, `BTRL_v2_report.md`, `SEEDED_COMPARISON_REPORT.md` (July 2026, different tasks per agent) and the research wiki (`docs/wiki/home.md`; the planning proposal is `docs/wiki/planning-in-sampled-model.md`) |
| `CLAUDE.md` | architecture notes and execution flow |

## Setup

Python 3.10, CUDA GPU recommended.

```bash
pip install -r requirements.txt
```

## Running

PoSTR, one seed of the main configuration (neural-linear posterior, randomized DeepSea-5):

```bash
cd src
python run_v3_1M.py --seed 1 --name postr_rand --explore_temp 4
```

The task is set by `experiment:` keys in the config: `randomize_actions` (default
`true`), `mapping_seed` (default: the run seed, shared by the train and test envs) and
`legacy_goal` (default `false`; `true` reproduces the July task).

Both agents, 5 seeds each, tracked in a local MLflow server (started if needed;
`MLFLOW_STORE` sets its backend folder, `PY` the Python interpreter):

```bash
cd src
./run_comparison.sh          # or: ./run_comparison.sh resume
```

PoSTR writes a full-state resume checkpoint every 20k steps
(`logdir/.../checkpoints/latest/resume.pt`; `run_v3_1M.py --resume_from`).

PSDRL baseline, one seed:

```bash
cd baselines/psdrl/src
python main.py --config configs/psdrl_deepsea5_prior1e3_1M.yaml --seed 1
```

Logs and checkpoints go to `logdir/` under the working directory (TensorBoard events
plus `metrics.jsonl`).

Hyperparameter search for the transition network and its posterior: run
`src/hpo1_transition_search_blr.ipynb` (200 NSGA-II trials on a fixed offline dataset;
study in `src/hpo_artifacts/optuna_postr_current.db`, resumable; trials appear in the
MLflow experiment "HPO — current setup" as they finish).

## Provenance of this snapshot

The original working copy was lost in August 2026. This snapshot was rebuilt from the
editor's saved file versions and session logs, so a few files are reconstructions:

- `src/TRL/common/logger.py`, `src/TRL/common/data_manager.py`,
  `src/TRL/networks/terminal.py` — restored from full-file reads in the session logs.
- `src/TRL/agent/__init__.py` (`from .btrl import BTRL as Agent`) and the other package
  `__init__.py` files — recreated; the originals were not captured.
- `src/hpo1_transition_search.ipynb` — regenerated with `generate_hpo_notebooks.py`
  (no outputs; the July study results were lost).
- `src/posterior_alive.py`, `src/*_gate.sh`, `src/rca_watch.sh`,
  `src/v3_stoch_launcher.sh`, `src/configs/config_btrl_vector_env5.yaml`,
  `baselines/psdrl/src/configs/psdrl_deepsea5_ctrl.yaml` — recovered from the session logs.

Some July configs and analysis scripts could not be recovered (e.g. the v8 LoRA and v9
configs, `best_*_config.yaml` tuning outputs, plotting scripts). All run outputs
(`logdir/`, checkpoints, Optuna databases) were lost; the results survive only in `docs/`.

## Issues fixed on this branch

Found in September 2026; they affect every result produced with the `main` (July) code:

1. **PoSTR does not run standard DeepSea.** `TRL/common/utils.py` builds DeepSea with
   `randomize_actions=False` (bsuite's debug mode, where one action is "right" in every
   cell) and `env_step` ends the episode with reward 0.99 on *reaching* the bottom-right
   cell. The PSDRL baseline runs standard randomized DeepSea, so PoSTR-vs-PSDRL numbers
   compare different tasks.
2. **PSDRL test environment is a different task.** `baselines/psdrl/.../deepsea_wrapper.py`
   passes no `mapping_seed`, so bsuite draws a fresh random action mapping for the
   training and the test environment.
3. **Reward-loss shape bug.** `TRL/training/transition.py` (and the freeze-check loss in
   `agent/btrl.py`) compare a `[B,1]` prediction with a `[B,1,1]` target, which
   broadcasts the loss over all B×B prediction/target pairs.

Also found (not yet changed): PoSTR's value network is constant across states in every
run, so the agent acts one-step greedy on predicted reward; on randomized DeepSea it
stays at the random-policy solve rate while PSDRL learns on 4/5 seeds
(`docs/RANDOMIZED_COMPARISON_REPORT.md`). The proposed fix — exact planning in each
posterior sample — is written up in `docs/wiki/planning-in-sampled-model.md`.

## Citation

PoSTR builds on PSDRL:

```
@inproceedings{sasso2023posterior,
  title = {Posterior Sampling for Deep Reinforcement Learning},
  author = {Sasso, Remo and Conserva, Michelangelo and Rauber, Paulo},
  booktitle={International Conference on Machine Learning},
  year = {2023}
}
```
