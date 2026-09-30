cd /media/shamail/CRUCIAL/MS_QMUL/Thesis/BTRL/src
minstep() {
python3 - <<'PY'
import json,glob,os
ms=1e9
for t in [2,4]:
  for s in [2,3]:
    g=glob.glob(f'logdir/5-determinsitic/BTRL-v3_widen_t{t}_seed{s}/*/metrics.jsonl')
    st=0
    if g:
        f=sorted(g,key=os.path.getmtime)[-1]
        for l in open(f):
            try:
                r=json.loads(l.replace('NaN','null'))
                if r.get('Timestep'): st=r['Timestep']
            except: pass
    ms=min(ms,st)
print(int(ms))
PY
}
start=$(date +%s)
while true; do
  sleep 180
  n=$(pgrep -f 'run_v3_1M.py --seed .* --name v3_widen' | wc -l)
  ms=$(minstep)
  el=$(( $(date +%s) - start ))
  if [ "$n" -lt 1 ] || [ "$ms" -ge 250000 ]; then
    echo "WIDEN_GATE FIRED: elapsed=${el}s alive=${n} min_step=${ms}"
    break
  fi
done
