"""
Hyperparameter tuning for BTRL (Optuna).

Three-stage pipeline, one notebook per stage (see src/hpo1..hpo3 notebooks):
  1. transition_search — offline multi-objective search over the Transformer
     world model (mean grid distance x posterior miscalibration).
  2. agent_eval        — full-agent evaluation harness used to validate the
     Pareto candidates on downstream performance (cumulative regret).
  3. value_search      — online single-objective search over the value network
     with the winning transition config frozen.

All studies share one SQLite database (hpo_artifacts/optuna_btrl.db) so the
notebooks can be run independently and resumed after interruption.
"""
