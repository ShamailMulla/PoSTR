"""Copy a finished (or running) Optuna study into MLflow: one parent run per study,
one child run per completed trial.

    python optuna_to_mlflow.py --storage sqlite:///hpo_artifacts/optuna_btrl.db \
        --study btrl_transition_hpo --experiment "July 30" --run-name "Stage 1 transition search" \
        [--notebook hpo1_transition_search_july30_run.ipynb] [--watch]

Each child run gets the trial's hyperparameters (params), its objective values
(`objective_0`, `objective_1`, ... plus the names given by --objective-names) and every
numeric user attribute as metrics, and a `pareto` tag. Re-running only adds trials
that are not in MLflow yet, so --watch can follow a study while it runs.
"""
import argparse
import os
import time

import mlflow
import optuna

URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def sync(client, exp_id, parent_id, study, objective_names):
    done = {r.data.tags.get("optuna_trial") for r in
            client.search_runs([exp_id], f"tags.mlflow.parentRunId = '{parent_id}'", max_results=5000)}
    pareto = {t.number for t in study.best_trials} if study.directions and len(study.directions) > 1 \
        else ({study.best_trial.number} if any(t.state.is_finished() for t in study.trials) else set())
    added = 0
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE or str(t.number) in done:
            continue
        run = client.create_run(exp_id, run_name=f"trial {t.number:03d}",
                                tags={"mlflow.parentRunId": parent_id, "optuna_trial": str(t.number),
                                      "pareto": str(t.number in pareto).lower()})
        rid = run.info.run_id
        for k, v in t.params.items():
            client.log_param(rid, k, v)
        for i, v in enumerate(t.values):
            client.log_metric(rid, objective_names[i] if i < len(objective_names) else f"objective_{i}", v)
        for k, v in t.user_attrs.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in t.params:
                client.log_metric(rid, k, float(v))
        client.log_metric(rid, "duration_s", (t.datetime_complete - t.datetime_start).total_seconds())
        client.set_terminated(rid, "FINISHED", end_time=int(t.datetime_complete.timestamp() * 1000))
        added += 1
    # refresh Pareto tags (the front changes as trials arrive)
    for r in client.search_runs([exp_id], f"tags.mlflow.parentRunId = '{parent_id}'", max_results=5000):
        tag = str(int(r.data.tags.get("optuna_trial", -1)) in pareto).lower()
        if r.data.tags.get("pareto") != tag:
            client.set_tag(r.info.run_id, "pareto", tag)
    complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    client.log_metric(parent_id, "completed_trials", complete)
    client.log_metric(parent_id, "pareto_size", len(pareto))
    return added, complete


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage", required=True)
    ap.add_argument("--study", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--objective-names", nargs="*", default=["mean_grid_dist", "overconfident_error_rate"])
    ap.add_argument("--notebook", help="executed notebook to attach to the parent run")
    ap.add_argument("--watch", action="store_true", help="keep syncing until the study stops growing")
    ap.add_argument("--target", type=int, default=0, help="with --watch: stop once this many trials complete")
    a = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    mlflow.set_tracking_uri(URI)
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(a.experiment)
    exp_id = exp.experiment_id if exp else client.create_experiment(a.experiment)
    parents = client.search_runs([exp_id], f"attributes.run_name = '{a.run_name}'")
    parent_id = parents[0].info.run_id if parents else client.create_run(
        exp_id, run_name=a.run_name, tags={"optuna_study": a.study, "optuna_storage": a.storage}).info.run_id
    if not parents:
        study = optuna.load_study(study_name=a.study, storage=a.storage)
        for k, v in {"directions": ",".join(d.name for d in study.directions),
                     "sampler": type(study.sampler).__name__}.items():
            client.log_param(parent_id, k, v)
    while True:
        study = optuna.load_study(study_name=a.study, storage=a.storage)
        added, complete = sync(client, exp_id, parent_id, study, a.objective_names)
        print(f"{time.strftime('%H:%M:%S')} +{added} trials ({complete} complete)", flush=True)
        if not a.watch or (a.target and complete >= a.target):
            break
        time.sleep(60)
    if a.notebook and os.path.exists(a.notebook):
        client.log_artifact(parent_id, a.notebook)
    client.set_terminated(parent_id, "FINISHED")


if __name__ == "__main__":
    main()
