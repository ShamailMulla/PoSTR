#!/bin/bash
# Sequenced launch queue (2026-07-19):
#   1. wait for the in-flight v8 LoRA batch to finish
#   2. v9_fixed DeepSea ablation, 5 seeds x 1M
#   3. when done: memory-env batch — BTRL v9_memory 5 seeds + PSDRL memory5 5 seeds
PY=/home/shamail/torch2_cuda12_env/bin/python3
BTRL=/media/shamail/CRUCIAL/MS_QMUL/Thesis/BTRL/src
BRL=/media/shamail/CRUCIAL/MS_QMUL/Thesis/BRL/src

echo "$(date) waiting for v8 batch to finish..."
while pgrep -f config_v8_lora > /dev/null; do sleep 300; done
echo "$(date) v8 done — launching v9_fixed (5 seeds)"

cd "$BTRL"
pids=()
for s in 1 2 3 4 5; do
  nohup $PY run_v3_1M.py --seed $s --config configs/config_v9_fixed.yaml --name v9_fixed > run_v9_fixed_seed$s.log 2>&1 &
  pids+=($!)
  sleep 20
done
wait "${pids[@]}"
echo "$(date) v9_fixed done — launching memory batch (BTRL v9_memory + PSDRL memory5, 5 seeds each)"

pids=()
for s in 1 2 3 4 5; do
  nohup $PY run_v3_1M.py --seed $s --config configs/config_v9_memory.yaml --name v9_memory > run_v9_memory_seed$s.log 2>&1 &
  pids+=($!)
  sleep 20
  ( cd "$BRL" && nohup $PY main.py --config configs/psdrl_memory5_1M.yaml --seed $s > run_memory5_seed$s.log 2>&1 ) &
  pids+=($!)
  sleep 20
done
wait "${pids[@]}"
echo "$(date) all queued runs complete"
