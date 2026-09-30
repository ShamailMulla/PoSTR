# PoSTR — Posterior Sampling with Transformers for RL

MSc thesis project (Queen Mary University of London). PoSTR (called BTRL in the code:
`TRL/`, `BTRL` agent class) extends [PSDRL](https://github.com/remosasso/PSDRL) —
Posterior Sampling for Deep RL — by replacing its GRU transition model with a
Transformer world model, with Thompson sampling over world models for deep
exploration. Evaluated on bsuite DeepSea.

This repository is a fork of PSDRL. Upstream PSDRL, with the thesis's changes for
DeepSea, lives in [`baselines/psdrl/`](baselines/psdrl/) and is the baseline.

> **Branches.** `main` holds the code as of **30 July 2026** (thesis state). Ongoing
> development — bug fixes, standard randomized DeepSea, MLflow tracking, planning in
> the sampled model — happens on the development branch; see
> [Known issues in this snapshot](#known-issues-in-this-snapshot).

## Layout

| Path | What it is |
|---|---|
| `src/main.py`, `src/run_*.py` | PoSTR entry points (`run_v3_1M.py` runs one seed of the neural-linear agent for 1M steps) |
| `src/TRL/` | the PoSTR package: `agent/btrl.py` (agent), `networks/transition.py` (Transformer world model, BayesFormer dropout), `bayes/neural_linear_head.py` (Bayesian linear regression posterior over transformer features, v3+), `training/` (transition and value training), `common/` (replay, env wrappers, utils), `tuning/` (hyperparameter-search code) |
| `src/configs/` | experiment configs (`config_v3_neural_linear.yaml` is the main PoSTR config; v5/v6 are the ring-buffer and frozen-feature ablations) |
| `src/hpo1_transition_search.ipynb` | Optuna search for the transition network (stage 1 of the pipeline in `docs/HPO_REPORT.md`); `generate_hpo_notebooks.py` regenerates it and stages 2–3 |
| `src/optuna_*_hparam_search.ipynb` | earlier single-notebook Optuna searches (transition, value) |
| `src/*.ipynb`, `src/TRL/tuning/test1_*.ipynb` | transition-model experiments from the original thesis submission |
| `baselines/psdrl/` | PSDRL baseline: upstream code plus a bsuite DeepSea / MemoryChain wrapper, DeepSea configs and Optuna notebooks (`param_tuning/`) |
| `docs/` | `HPO_REPORT.md`, `BTRL_v2_report.md`, `SEEDED_COMPARISON_REPORT.md` (5-seed PoSTR vs PSDRL, July 2026) and the research wiki (`docs/wiki/home.md`) |
| `CLAUDE.md` | architecture notes and execution flow |

## Setup

Python 3.10, CUDA GPU recommended.

```bash
pip install -r requirements.txt
```

## Running

PoSTR, one seed of the main configuration (neural-linear posterior, deterministic DeepSea-5):

```bash
cd src
python run_v3_1M.py --seed 1 --name postr_det --explore_temp 4
```

PSDRL baseline, one seed:

```bash
cd baselines/psdrl/src
python main.py --config configs/psdrl_deepsea5_prior1e3_1M.yaml --seed 1
```

Logs and checkpoints go to `logdir/` under the working directory (TensorBoard events
plus `metrics.jsonl`).

Hyperparameter search for the transition network: open `src/hpo1_transition_search.ipynb`
and run all cells (200 NSGA-II trials on a fixed offline dataset; the Optuna study is
stored in `src/hpo_artifacts/optuna_btrl.db` and is resumable).

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

## Known issues in this snapshot

Found in September 2026 and fixed on the development branch — they affect every
result produced with this code:

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
