cd /media/shamail/CRUCIAL/MS_QMUL/Thesis/BTRL/src
RUN=logdir/5-determinsitic/BTRL-v3_rca_seed3/0
PY=/home/shamail/torch2_cuda12_env/bin/python3
while [ ! -d "$RUN/checkpoints/100000" ]; do sleep 60; done
sleep 30
echo "=== full peak->collapse sweep ===" > /tmp/rca_sweep.log
$PY diagnostic_v3_rca.py $RUN 20000 40000 60000 80000 100000 >> /tmp/rca_sweep.log 2>&1
echo "SWEEP_DONE" >> /tmp/rca_sweep.log
