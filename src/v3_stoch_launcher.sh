cd /media/shamail/CRUCIAL/MS_QMUL/Thesis/BTRL/src
# wait until all 5 deterministic v3 seeds have exited (frees ~4GB GPU)
for p in 205877 205918 205960 206003 206050; do
  while kill -0 $p 2>/dev/null; do sleep 120; done
done
sleep 20
# launch 5 stochastic v3 seeds
for s in 1 2 3 4 5; do
  nohup /home/shamail/torch2_cuda12_env/bin/python3 -u run_v3_1M.py \
    --config configs/config_v3_neural_linear_stochastic.yaml --name v3_neural_linear_stoch --seed $s \
    > run_v3_stoch_seed${s}.log 2>&1 &
  echo "v3-stoch seed $s PID $!"
  sleep 4
done
echo "V3 STOCHASTIC x5 LAUNCHED at $(date +%H:%M)"
