#!/usr/bin/env bash
# One command after a reboot: start the MLflow server (if needed), then the PoSTR runs.
#
#   ./start_postr.sh           # fresh 5-seed run
#   ./start_postr.sh resume    # continue from the latest checkpoints
cd "$(dirname "$0")"
M=${MLFLOW_STORE:-/mnt/crucial_ssd/recovered_thesis/mlflow}
if ! curl -sf http://127.0.0.1:5000/health > /dev/null; then
  setsid nohup ${PY:-/home/shamail/torch2_cuda12_env/bin/python3} -m mlflow server \
    --backend-store-uri sqlite:///$M/mlflow.db --artifacts-destination $M/artifacts \
    --host 127.0.0.1 --port 5000 >> $M/server.log 2>&1 < /dev/null &
  for i in $(seq 30); do curl -sf http://127.0.0.1:5000/health > /dev/null && break; sleep 2; done
  echo "MLflow server started: http://127.0.0.1:5000"
fi
exec ./launch_postr_traced.sh "$@"
