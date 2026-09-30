"""Stream the PoSTR (postr_det) rerun seeds into the local MLflow server.

    python mlflow_sync.py --init 1   # create parent + seed runs, print seed 1's run id
    python mlflow_sync.py            # loop until every seed has finished
    python mlflow_sync.py --once     # single pass
    python mlflow_sync.py --reopen 1 --from_step 60000   # seed 1 resumed from a checkpoint

Every command takes --arm (default postr_det, the Sep 2026 rerun); see ARMS below for the
randomized-DeepSea comparison arms postr_rand and psdrl_rand.

Seeds are child runs of one parent run, which also carries the mean across seeds.

Reads logdir/*/BTRL-postr_det_seed*/*/metrics.jsonl (written by the running agents, so
training is untouched) and logs one point per 1,000 env steps, plus derived metrics that
match SEEDED_COMPARISON_REPORT.md: rolling solve rate, cumulative reward/regret,
first-treasure step, time-to-learn and solve rate by fifths of training.
"""
import argparse
import collections
import glob
import json
import os
import subprocess
import time

import mlflow
import numpy as np
from mlflow.entities import Metric, Param, RunStatus

URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = "PoSTR postr_det - rerun vs original"
ROOT = os.path.dirname(os.path.abspath(__file__))
BRL = os.path.normpath(os.path.join(ROOT, "../baselines/psdrl/src"))
SOLVED = 0.9
BUCKET, STEPS = 1000, 1_000_000
RAND_EXP = "Randomized DeepSea-5: PoSTR vs PSDRL (Sep 2026)"
ARMS = {
    # legacy task (non-randomized, 0.99 on reaching the corner), pre-reward-loss-fix
    "postr_det": dict(experiment="PoSTR postr_det - rerun vs original",
                      group="postr_det rerun (traced, Sep 2026)", agent="PoSTR", source="rerun_2026-09",
                      state="mlflow_sync_state_traced.json", optimal=0.984,
                      metrics=ROOT + "/logdir/*/BTRL-postr_det_seed{seed}/*/metrics.jsonl",
                      proc="run_v3_1M.py --seed {seed} --name postr_det",
                      log=ROOT + "/runlogs/postr_det_s{seed}.log"),
    # standard randomized DeepSea, reward-loss fix, shared train/test action mapping
    "postr_rand": dict(experiment=RAND_EXP, group="PoSTR postr_rand (randomized DeepSea)", agent="PoSTR",
                       source="randomized_2026-09", state="mlflow_sync_state_postr_rand.json", optimal=0.99,
                       metrics=ROOT + "/logdir/*/BTRL-postr_rand_seed{seed}/*/metrics.jsonl",
                       proc="run_v3_1M.py --seed {seed} --name postr_rand",
                       log=ROOT + "/runlogs/postr_rand_s{seed}.log"),
    "psdrl_rand": dict(experiment=RAND_EXP, group="PSDRL prior 1e-3 (randomized DeepSea)", agent="PSDRL",
                       source="randomized_2026-09", state="mlflow_sync_state_psdrl_rand.json", optimal=0.99,
                       metrics=BRL + "/logdir/*/PSDRL-psdrl_rand_seed{seed}/*/metrics.jsonl",
                       proc="run_psdrl.py --seed {seed} --name psdrl_rand",
                       log=ROOT + "/runlogs/psdrl_rand_s{seed}.log"),
}
ARM_NAME = "postr_det"
ARM = ARMS[ARM_NAME]


def configure(name):
    """Point every function below at one arm's experiment, files and processes."""
    global ARM_NAME, ARM, EXPERIMENT, GROUP, STATE, OPTIMAL, TAGS
    ARM_NAME, ARM = name, ARMS[name]
    EXPERIMENT, GROUP, OPTIMAL = ARM["experiment"], ARM["group"], ARM["optimal"]
    STATE = os.path.join(ROOT, "runlogs", ARM["state"])
    TAGS = {"source": ARM["source"], "agent": ARM["agent"], "arm": name, "group": GROUP,
            "host": os.uname().nodename}


def flatten(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", str(v)[:500]


def read_rows(path):
    rows = []
    for line in open(path):
        try:
            rows.append(json.loads(line.replace("NaN", "null")))
        except json.JSONDecodeError:
            pass  # line still being written
    return rows


def alive(seed):
    return subprocess.run(["pgrep", "-f", ARM["proc"].format(seed=seed)], capture_output=True).returncode == 0


configure(ARM_NAME)


def get_parent(client, exp_id, state):
    if "parent" not in state:
        run = client.create_run(exp_id, run_name=GROUP, tags={**TAGS, "level": "group (5 seeds)"})
        state["parent"] = {"run_id": run.info.run_id, "synced": 0}
    return state["parent"]["run_id"]


def get_run(client, exp_id, seed, state):
    key = f"seed{seed}"
    if key in state:
        return state[key]["run_id"]
    run = client.create_run(exp_id, run_name=f"{'rerun' if ARM_NAME == 'postr_det' else ARM_NAME} seed{seed}",
                            tags={**TAGS, "seed": str(seed),
                                  "mlflow.parentRunId": get_parent(client, exp_id, state)})
    state[key] = {"run_id": run.info.run_id, "synced": 0, "fifths": 0, "params": False}
    return run.info.run_id


def stitched_rows(files):
    """A resumed run logs to a new numbered dir starting at the resume step; keep each
    earlier dir's rows only up to where the next one takes over."""
    rows = []
    for f in files:
        new = read_rows(f)
        if new:
            rows = [r for r in rows if r["Timestep"] < new[0]["Timestep"]] + new
    return rows


def sync_seed(client, exp_id, seed, state):
    g = sorted(glob.glob(ARM["metrics"].format(seed=seed)),
               key=lambda f: int(os.path.basename(os.path.dirname(f))))
    if not g:
        return None
    run_dir = os.path.dirname(g[0])
    run_id = get_run(client, exp_id, seed, state)
    st = state[f"seed{seed}"]

    if not st["params"] and os.path.exists(f"{run_dir}/hyper_parameters.txt"):
        params = dict(flatten(json.load(open(f"{run_dir}/hyper_parameters.txt"))))
        items = [Param(k, v) for k, v in params.items()]
        for i in range(0, len(items), 100):
            client.log_batch(run_id, params=items[i:i + 100])
        client.set_tag(run_id, "logdir", run_dir)
        st["params"] = True

    rows = stitched_rows(g)
    if not rows:
        return None
    last_step = rows[-1]["Timestep"]
    finished = last_step >= STEPS - 10
    done_bucket = STEPS if finished else (last_step // BUCKET) * BUCKET

    # Walk every episode (derived metrics need full history), emit points past `synced`.
    now = int(time.time() * 1000)
    metrics, win, win500 = [], collections.deque(maxlen=1000), collections.deque(maxlen=500)
    cum_r = cum_regret = 0.0
    n_eps = n_solved = 0
    first = ttl = None
    fifth = [0, 0]
    fifth_vals = []
    next_emit = BUCKET
    prev = None

    series = {}

    def emit(step, row):
        series[step] = (sum(win) / max(len(win), 1), cum_r, cum_regret, n_solved / max(n_eps, 1))
        if step <= st["synced"] or step > done_bucket:
            return
        for k, v in row.items():
            if k != "Timestep" and isinstance(v, (int, float)) and v == v:
                metrics.append(Metric(k, float(v), now, step))
        for k, v in (("solve_rate_rolling1k", sum(win) / max(len(win), 1)),
                     ("cum_reward", cum_r), ("cum_regret", cum_regret), ("episodes", n_eps),
                     ("solve_rate_overall", n_solved / max(n_eps, 1))):
            metrics.append(Metric(k, float(v), now, step))

    for row in rows:
        t = row["Timestep"]
        while prev is not None and t >= next_emit:   # bucket boundary crossed
            emit(next_emit, prev)
            next_emit += BUCKET
        r = row.get("Reward/Train_Reward")
        if r is not None:
            n_eps += 1
            solved = r > SOLVED
            n_solved += solved
            cum_r += r
            cum_regret += OPTIMAL - r
            win.append(solved)
            win500.append(solved)
            if first is None and solved:
                first = t
            if ttl is None and len(win500) == 500 and sum(win500) >= 250:
                ttl = t
            f = min(int(t // (STEPS // 5)), 4)
            if f > len(fifth_vals):
                fifth_vals.append(fifth[1] / max(fifth[0], 1))
                fifth = [0, 0]
            fifth[0] += 1
            fifth[1] += solved
        prev = row
    if finished:
        emit(STEPS, prev)
        fifth_vals.append(fifth[1] / max(fifth[0], 1))

    for i, v in enumerate(fifth_vals):
        if i >= st["fifths"]:
            metrics.append(Metric("solve_rate_fifth", v, now, (i + 1) * STEPS // 5))
            metrics.append(Metric(f"solve_rate_fifth{i + 1}", v, now, (i + 1) * STEPS // 5))
    st["fifths"] = len(fifth_vals)
    for k, v in (("first_treasure_step", first), ("time_to_learn_step", ttl)):
        if v is not None:
            metrics.append(Metric(k, float(v), now, 0))
    metrics.append(Metric("progress_steps", float(last_step), now, 0))

    for i in range(0, len(metrics), 1000):
        client.log_batch(run_id, metrics=metrics[i:i + 1000])
    st["synced"] = max(st["synced"], done_bucket)

    if finished or not alive(seed):
        log = ARM["log"].format(seed=seed)
        if os.path.exists(log):
            client.log_artifact(run_id, log)
        status = RunStatus.FINISHED if finished else RunStatus.FAILED
        client.set_terminated(run_id, RunStatus.to_string(status))
        st["closed"] = True
    return {"series": series, "done": done_bucket, "first": first, "ttl": ttl, "fifths": fifth_vals}


def sync_parent(client, exp_id, state, results):
    """Mean over seeds, for buckets every seed has reached."""
    pid = get_parent(client, exp_id, state)
    ps = state["parent"]
    if len(results) < 5:
        return
    upto = min(r["done"] for r in results.values())
    now = int(time.time() * 1000)
    names = ("solve_rate_rolling1k", "cum_reward", "cum_regret", "solve_rate_overall")
    ms = []
    for step in sorted(results[1]["series"]):
        if ps["synced"] < step <= upto and all(step in r["series"] for r in results.values()):
            for i, k in enumerate(names):
                vals = [r["series"][step][i] for r in results.values()]
                ms.append(Metric(f"mean_{k}", float(np.mean(vals)), now, step))
                ms.append(Metric(f"min_{k}", float(np.min(vals)), now, step))
                ms.append(Metric(f"max_{k}", float(np.max(vals)), now, step))
            ps["synced"] = step
    for k in ("first", "ttl"):
        vals = [r[k] for r in results.values() if r[k] is not None]
        if vals:
            ms.append(Metric(f"median_{'first_treasure_step' if k == 'first' else 'time_to_learn_step'}",
                             float(np.median(vals)), now, 0))
    n_f = min(len(r["fifths"]) for r in results.values())
    for i in range(n_f):
        ms.append(Metric("mean_solve_rate_fifth", float(np.mean([r["fifths"][i] for r in results.values()])),
                         now, (i + 1) * STEPS // 5))
    for i in range(0, len(ms), 1000):
        client.log_batch(pid, metrics=ms[i:i + 1000])
    if all(state.get(f"seed{s}", {}).get("closed") for s in range(1, 6)) and not ps.get("closed"):
        client.set_terminated(pid, "FINISHED")
        ps["closed"] = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="postr_det", choices=sorted(ARMS))
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--init", type=int, help="create runs and print this seed's run id")
    ap.add_argument("--reopen", type=int, help="mark this seed's run RUNNING again after a resume")
    ap.add_argument("--from_step", type=int, default=0, help="resume step (re-sync from here)")
    a = ap.parse_args()
    configure(a.arm)
    mlflow.set_tracking_uri(URI)
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    exp_id = exp.experiment_id if exp else client.create_experiment(EXPERIMENT)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    if a.reopen:
        st, rid = state[f"seed{a.reopen}"], state[f"seed{a.reopen}"]["run_id"]
        client.update_run(rid, status="RUNNING")
        client.set_tag(rid, f"resumed_from_step_{a.from_step}", time.strftime("%Y-%m-%d %H:%M"))
        st["closed"] = False
        st["synced"] = min(st["synced"], a.from_step // BUCKET * BUCKET)
        st["fifths"] = min(st["fifths"], a.from_step // (STEPS // 5))
        state["parent"]["synced"] = min(state["parent"]["synced"], st["synced"])
        state["parent"]["closed"] = False
        client.update_run(state["parent"]["run_id"], status="RUNNING")
        json.dump(state, open(STATE, "w"), indent=1)
        print(rid)
        return
    if a.init:
        rid = get_run(client, exp_id, a.init, state)
        json.dump(state, open(STATE, "w"), indent=1)
        print(rid)
        return
    results = {}
    while True:
        for s in range(1, 6):
            if not state.get(f"seed{s}", {}).get("closed") or s not in results:
                r = sync_seed(client, exp_id, s, state) if f"seed{s}" in state else None
                if r:
                    results[s] = r
        sync_parent(client, exp_id, state, results)
        json.dump(state, open(STATE, "w"), indent=1)
        if a.once or all(state.get(f"seed{s}", {}).get("closed") for s in range(1, 6)):
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
