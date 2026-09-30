#!/usr/bin/env bash
# PoSTR vs PSDRL on standard randomized DeepSea-5, 5 seeds x 1M steps each, tracked in MLflow
# (experiment "Randomized DeepSea-5: PoSTR vs PSDRL (Sep 2026)").
#
#   ./run_comparison.sh           # fresh runs (starts the MLflow server if it is down)
#   ./run_comparison.sh resume    # continue PoSTR seeds from their latest resume.pt;
#                                 # PSDRL has no resume support - dead PSDRL seeds are reported
#
# PoSTR: run_v3_1M.py, config_v3_neural_linear.yaml, explore_temp 4 (the postr_det settings),
#        now with randomize_actions + the reward-loss fix; traced 1 cycle in TRACE_EVERY.
# PSDRL: baselines/psdrl/src/run_psdrl.py, psdrl_deepsea5_prior1e3_1M.yaml (the corrected baseline).
# Each run holds a sleep/idle inhibitor; LID_INHIBIT=1 also blocks lid-close suspend.
cd "$(dirname "$0")"
MODE=${1:-fresh}
SRC=$(pwd)
BRL=$(cd ../baselines/psdrl/src && pwd)
PY=${PY:-/home/shamail/torch2_cuda12_env/bin/python3}
M=${MLFLOW_STORE:-/mnt/crucial_ssd/recovered_thesis/mlflow}   # MLflow backend store
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MLFLOW_TRACKING_URI=http://127.0.0.1:5000 TRACE_EVERY=${TRACE_EVERY:-20}
WHAT=sleep:idle
[ "${LID_INHIBIT:-0}" = 1 ] && WHAT=sleep:idle:handle-lid-switch
mkdir -p runlogs

if ! $PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "CUDA is not available - reboot (or reset the GPU) before launching"; exit 1
fi
if ! curl -sf http://127.0.0.1:5000/health > /dev/null; then
  setsid nohup $PY -m mlflow server --backend-store-uri sqlite:///$M/mlflow.db --artifacts-destination $M/artifacts \
    --host 127.0.0.1 --port 5000 >> $M/server.log 2>&1 < /dev/null &
  for i in $(seq 30); do curl -sf http://127.0.0.1:5000/health > /dev/null && break; sleep 2; done
  echo "MLflow server started: http://127.0.0.1:5000"
fi

launch() {  # arm seed workdir script extra-args...
  local arm=$1 s=$2 dir=$3 script=$4; shift 4
  local rid
  if [ "$MODE" = resume ] && [ "$arm" = postr_rand ]; then
    local ckpt step
    ckpt=$(ls -t logdir/*/BTRL-postr_rand_seed$s/*/checkpoints/latest/resume.pt 2>/dev/null | head -1)
    [ -z "$ckpt" ] && { echo "$arm seed$s: no checkpoint - skipped"; return; }
    step=$($PY -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['experiment_step'])" "$ckpt" 2>/dev/null | tail -1)
    rid=$($PY mlflow_sync.py --arm $arm --reopen $s --from_step $step 2>/dev/null | tail -1)
    set -- "$@" --resume_from "$SRC/$ckpt"
    echo "$arm seed$s resuming from step $step"
  else
    rid=$($PY mlflow_sync.py --arm $arm --init $s 2>/dev/null | tail -1)
  fi
  (cd "$dir" && MLFLOW_RUN_ID=$rid setsid nohup systemd-inhibit --what=$WHAT --who="$arm seed$s" \
     --why="1M-step training run" --mode=block $PY -u $script --seed $s --name $arm "$@" \
     >> "$SRC/runlogs/${arm}_s$s.log" 2>&1 < /dev/null &)
  echo "$arm seed$s launched (run $rid)"
  sleep 4
}

for s in 1 2 3 4 5; do
  if pgrep -f "run_v3_1M.py --seed $s --name postr_rand" > /dev/null; then
    echo "postr_rand seed$s already running"
  else
    launch postr_rand $s "$SRC" run_v3_1M.py --steps 1000000 --save_freq 100000 --explore_temp 4
  fi
  if pgrep -f "run_psdrl.py --seed $s --name psdrl_rand" > /dev/null; then
    echo "psdrl_rand seed$s already running"
  elif [ "$MODE" = resume ]; then
    echo "psdrl_rand seed$s is not running and PSDRL cannot resume - relaunch it fresh with ./run_comparison.sh"
  else
    launch psdrl_rand $s "$BRL" run_psdrl.py --steps 1000000
  fi
done

for arm in postr_rand psdrl_rand; do
  pgrep -f "mlflow_sync.py --arm $arm$" > /dev/null || {
    setsid nohup $PY mlflow_sync.py --arm $arm >> runlogs/mlflow_sync_$arm.log 2>&1 < /dev/null &
    echo "sync $arm started"; }
done
