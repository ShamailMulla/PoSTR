import json, glob, os, time
GOAL = 0.5
PIDS = {'003': 278427, '010': 278475, '030': 278555, '100': 278603}
TEMP = {'003': 0.03, '010': 0.1, '030': 0.3, '100': 1.0}
last = 0.0
flagged = set()

def rows(lbl):
    g = glob.glob('logdir/5-determinsitic/BTRL-v3_temp' + lbl + '_seed1/*/metrics.jsonl')
    if not g:
        return []
    out = []
    for l in open(sorted(g, key=os.path.getmtime)[-1]):
        try:
            out.append(json.loads(l.replace('NaN', 'null')))
        except Exception:
            pass
    return out

def alive(p):
    try:
        os.kill(p, 0); return True
    except OSError:
        return False

def gr500(tr):
    seg = tr[-500:]
    return 100 * sum(1 for r in seg if r > GOAL) / max(len(seg), 1)

while True:
    any_alive = False
    for lbl, pid in PIDS.items():
        if alive(pid):
            any_alive = True
        rs = rows(lbl)
        tr = [r['Reward/Train_Reward'] for r in rs if r.get('Reward/Train_Reward') is not None]
        step = rs[-1]['Timestep'] if rs else 0
        gr = gr500(tr)
        if lbl not in flagged and step > 120000 and gr > 40:
            print('*** temp=%s STABILIZED: %d%% at step %d (past danger zone) ***' % (TEMP[lbl], gr, step))
            flagged.add(lbl)
    now = time.time()
    if now - last > 1200:
        parts = []
        for lbl in PIDS:
            rs = rows(lbl)
            tr = [r['Reward/Train_Reward'] for r in rs if r.get('Reward/Train_Reward') is not None]
            st = rs[-1]['Timestep'] if rs else 0
            parts.append('T%s:%dk/%d%%' % (TEMP[lbl], st // 1000, gr500(tr)))
        print('temp-sweep | ' + ' | '.join(parts))
        last = now
    if not any_alive:
        print('TEMP SWEEP FINISHED — final trajectories:')
        for lbl in PIDS:
            rs = rows(lbl)
            tr = [r['Reward/Train_Reward'] for r in rs if r.get('Reward/Train_Reward') is not None]
            n = len(tr); B = max(1, n // 5)
            traj = [int(100 * sum(1 for r in tr[b*B:(b+1)*B] if r > GOAL) / max(1, len(tr[b*B:(b+1)*B]))) for b in range(5)]
            print('  temp=%s: %s' % (TEMP[lbl], traj))
        break
    time.sleep(120)
