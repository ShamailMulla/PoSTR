"""One-off: load the original Jul 2026 results (SEEDED_COMPARISON_REPORT.md) into MLflow.

    python mlflow_import_original.py

The raw logs of these runs were lost, so each run holds only the summary numbers the
report recorded. Safe to re-run: runs are matched by name and not duplicated.
"""
import os

import mlflow
from mlflow.entities import Metric

URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
ROOT = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.normpath(os.path.join(ROOT, "../docs/SEEDED_COMPARISON_REPORT.md"))
TAGS = {"source": "original_2026-07", "raw_logs": "lost (summary numbers from report only)"}
N = None  # "never"

# seed: episodes, overall, first, last100k, cum_reward, test, ttl, regret@100k, regret@1M, regret/ep last200k, fifths
POSTR = {
    1: (240800, .847, 15, .127, 200596, .821, 6123, 1172, 36351, .791, [.97, .99, .99, .99, .31]),
    2: (209852, .235, 10, .114, 47747, .219, 4313, 3001, 158748, .878, [.53, .22, .18, .13, .11]),
    3: (205127, .125, 75, .121, 24446, .123, 8726, 16609, 177398, .869, [.15, .12, .12, .11, .12]),
    4: (205686, .138, 5, .112, 27178, .134, 124564, 15965, 175217, .876, [.26, .11, .11, .11, .11]),
    5: (209230, .221, 10, .123, 44706, .206, 4838, 15317, 161176, .864, [.28, .45, .11, .13, .13]),
}
PSDRL = {
    1: (200000, 0., 140, 0., -108, 0., N, 19838, 198108, .990, [0.] * 5),
    2: (200000, 0., 220, 0., -91, .008, N, 19822, 198091, .990, [0.] * 5),
    3: (200000, .881, 5, .881, 174351, .032, 2500, 2347, 23649, .118, [.88] * 5),
    4: (200000, 0., 11155, .001, -17, 0., N, 19802, 198017, .990, [0.] * 5),
    5: (200000, 0., 7035, 0., -92, 0., N, 19832, 198092, .990, [0.] * 5),
}
SEED_KEYS = ("episodes", "solve_rate_overall", "first_treasure_step", "solve_rate_last100k",
             "cum_reward", "test_solve_rate", "time_to_learn_step", "cum_regret_100k",
             "cum_regret_1M", "regret_per_ep_last200k")
# arm: seeds retained at 1M, median first treasure, mean regret@1M, median regret@1M, mean final solve rate
FOURWAY = {
    "PSDRL ctrl (prior 1e-4)": (1, 220, 163191, 198091, .176),
    "PSDRL prior 1e-3": (2, 16680, 132060, 198072, .351),
    "PoSTR explore_temp 4 (postr_det)": (0, 10, 141778, 161176, .120),
    "PoSTR ring-only (v5)": (0, 10, 172387, 173700, .134),
}
# seed: phi frozen at step, val loss at freeze, flat solve rate, posterior distinct, modal
V6 = {1: (3404, 1.64, .31, 6.0, .35), 2: (2004, 1.62, .36, 6.0, .30), 3: (2902, 1.56, .15, 4.0, .55),
      4: (4304, 1.54, .07, 4.0, .50), 5: (10302, .57, .07, 2.2, .89)}


def ensure(client, exp_name, run_name, tags, metrics, params=None, artifact=None, parent=None):
    """Create the run once; on re-runs only (re)attach it to its parent group."""
    exp = client.get_experiment_by_name(exp_name)
    exp_id = exp.experiment_id if exp else client.create_experiment(exp_name)
    found = client.search_runs([exp_id], f"attributes.run_name = '{run_name}'")
    if found:
        if parent:
            client.set_tag(found[0].info.run_id, "mlflow.parentRunId", parent)
        return found[0].info.run_id
    if parent:
        tags = {**tags, "mlflow.parentRunId": parent}
    run = client.create_run(exp_id, run_name=run_name, tags={**TAGS, **tags})
    ms = []
    for k, v in metrics.items():
        if isinstance(v, list):   # stepped series [(step, value)]
            ms += [Metric(k, float(x), 0, s) for s, x in v]
        elif v is not None:
            ms.append(Metric(k, float(v), 0, 0))
    client.log_batch(run.info.run_id, metrics=ms)
    for k, v in (params or {}).items():
        client.log_param(run.info.run_id, k, v)
    if artifact and os.path.exists(artifact):
        client.log_artifact(run.info.run_id, artifact)
    client.set_terminated(run.info.run_id, "FINISHED")
    return run.info.run_id


def group(client, exp_name, name, tags, rows):
    """Parent run: mean of each per-seed metric (and of the fifths series)."""
    metrics = {}
    for k in {k for r in rows for k, v in r.items() if not isinstance(v, list)}:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            metrics[f"mean_{k}"] = sum(vals) / len(vals)
    if all("solve_rate_fifth" in r for r in rows):
        metrics["mean_solve_rate_fifth"] = [
            (s, sum(r["solve_rate_fifth"][i][1] for r in rows) / len(rows))
            for i, (s, _) in enumerate(rows[0]["solve_rate_fifth"])]
    return ensure(client, exp_name, name, {**tags, "level": "group"}, metrics)


def seed_metrics(row):
    m = dict(zip(SEED_KEYS, row[:10]))
    fifths = [((i + 1) * 200_000, v) for i, v in enumerate(row[10])]
    m["solve_rate_fifth"] = fifths
    for i, (s, v) in enumerate(fifths):
        m[f"solve_rate_fifth{i + 1}"] = [(s, v)]
    return m


def main():
    mlflow.set_tracking_uri(URI)
    c = mlflow.MlflowClient()
    cmp_exp, all_exp = "PoSTR postr_det - rerun vs original", "Thesis results - original Jul 2026"
    postr = {s: seed_metrics(row) for s, row in POSTR.items()}
    psdrl = {s: seed_metrics(row) for s, row in PSDRL.items()}
    v6 = {s: {"phi_frozen_step": fz, "val_loss_at_freeze": vl, "solve_rate_flat": sr,
              "posterior_distinct_final": dist, "posterior_modal_final": modal}
          for s, (fz, vl, sr, dist, modal) in V6.items()}
    postr_tags = {"agent": "PoSTR", "arm": "postr_det"}

    g = group(c, cmp_exp, "postr_det original (Jul 2026)", postr_tags, list(postr.values()))
    for s, m in postr.items():
        ensure(c, cmp_exp, f"original seed{s}", {**postr_tags, "seed": str(s)}, m, parent=g)

    g = group(c, all_exp, "PoSTR postr_det (5 seeds)", postr_tags, list(postr.values()))
    for s, m in postr.items():
        ensure(c, all_exp, f"PoSTR postr_det seed{s}", {**postr_tags, "seed": str(s)}, m, parent=g)

    tags = {"agent": "PSDRL", "arm": "ctrl"}
    g = group(c, all_exp, "PSDRL ctrl (5 seeds)", tags, list(psdrl.values()))
    for s, m in psdrl.items():
        ensure(c, all_exp, f"PSDRL ctrl seed{s}", {**tags, "seed": str(s)}, m,
               {"prior": "1e-4", "update_freq": "250", "time_limit": "25"}, parent=g)

    tags = {"agent": "PoSTR", "arm": "v6_frozen_phi"}
    g = group(c, all_exp, "PoSTR v6 frozen-phi (5 seeds)", tags, list(v6.values()))
    for s, m in v6.items():
        ensure(c, all_exp, f"PoSTR v6 frozen-phi seed{s}", {**tags, "seed": str(s)}, m, parent=g)

    g = ensure(c, all_exp, "Four-way comparison (arm aggregates)", {"level": "group"}, {})
    for arm, (ret, first, mean_r, med_r, final) in FOURWAY.items():
        ensure(c, all_exp, f"four-way: {arm}", {"agent": arm.split()[0], "arm": arm, "level": "aggregate (5 seeds)"},
               {"seeds_retained_at_1M": ret, "first_treasure_median": first, "cum_regret_1M_mean": mean_r,
                "cum_regret_1M_median": med_r, "solve_rate_final_mean": final}, parent=g)
    ensure(c, all_exp, "SEEDED_COMPARISON_REPORT", {"level": "report"}, {}, artifact=REPORT)


if __name__ == "__main__":
    main()
