#!/usr/bin/env bash
# Launch the 5 PoSTR postr_det seeds with MLflow tracing + system metrics, then the sync loop.
#
#   ./launch_postr_traced.sh           # fresh runs from step 0
#   ./launch_postr_traced.sh resume    # continue each seed from its newest resume.pt
#
# Each seed is a child run of "postr_det rerun (traced, Sep 2026)" (see mlflow_sync.py).
# Runs hold a systemd sleep/idle inhibitor, so idle timeouts or the power menu won't
# suspend the machine mid-run (a suspend invalidates the CUDA contexts and kills them).
# Closing the lid still suspends unless LID_INHIBIT=1 - only use that if the laptop stays
# somewhere ventilated with the lid shut.
cd "$(dirname "$0")"
MODE=${1:-fresh}
PY=${PY:-/home/shamail/torch2_cuda12_env/bin/python3}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MLFLOW_TRACKING_URI=http://127.0.0.1:5000 TRACE_EVERY=${TRACE_EVERY:-20}
WHAT=sleep:idle
[ "${LID_INHIBIT:-0}" = 1 ] && WHAT=sleep:idle:handle-lid-switch
mkdir -p runlogs

if pgrep -f "run_v3_1M.py --seed [0-9] --name postr_det" > /dev/null; then
  echo "postr_det seeds are already running - stop them first"; exit 1
fi
# btrl.py silently falls back to CPU (~1000x slower) - e.g. after a GPU fault on resume
# from suspend leaves the driver unusable until reboot. Refuse rather than crawl.
if ! $PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "CUDA is not available - reboot (or reset the GPU) before launching"; exit 1
fi
curl -sf http://127.0.0.1:5000/health > /dev/null || { echo "MLflow server is not running - use start_postr.sh"; exit 1; }

for s in 1 2 3 4 5; do
  EXTRA=""
  if [ "$MODE" = resume ]; then
    CKPT=$(ls -t logdir/*/BTRL-postr_det_seed$s/*/checkpoints/latest/resume.pt 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then echo "seed$s: no checkpoint to resume from - skipped"; continue; fi
    STEP=$($PY -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['experiment_step'])" "$CKPT" 2>/dev/null | tail -1)
    RID=$($PY mlflow_sync.py --reopen $s --from_step $STEP 2>/dev/null | tail -1)
    EXTRA="--resume_from $CKPT"
    echo "seed$s resuming from step $STEP ($CKPT)"
  else
    RID=$($PY mlflow_sync.py --init $s 2>/dev/null | tail -1)
  fi
  MLFLOW_RUN_ID=$RID setsid nohup systemd-inhibit --what=$WHAT --who="PoSTR seed$s" \
    --why="1M-step training run" --mode=block \
    $PY -u run_v3_1M.py --seed $s --name postr_det --steps 1000000 \
    --save_freq 100000 --explore_temp 4 $EXTRA >> runlogs/postr_det_s$s.log 2>&1 < /dev/null &
  echo "seed$s pid $! run $RID"
  sleep 4
done

pgrep -f "mlflow_sync.py$" > /dev/null || { setsid nohup $PY mlflow_sync.py >> runlogs/mlflow_sync.log 2>&1 < /dev/null & echo "sync pid $!"; }
