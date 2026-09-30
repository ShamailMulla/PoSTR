"""Progress of the rerun PoSTR (postr_det) seeds vs the original Jul 15-16 runs.

    python track_postr.py

Reference numbers are from SEEDED_COMPARISON_REPORT.md (original runs, 1M steps).
"""
import collections
import glob
import json
import os

SOLVED = 0.9
REF = {  # seed: (first treasure step, time-to-learn, solve-rate by fifths, overall solve rate)
    1: (15, 6123, [0.97, 0.99, 0.99, 0.99, 0.31], 0.847),
    2: (10, 4313, [0.53, 0.22, 0.18, 0.13, 0.11], 0.235),
    3: (75, 8726, [0.15, 0.12, 0.12, 0.11, 0.12], 0.125),
    4: (5, 124564, [0.26, 0.11, 0.11, 0.11, 0.11], 0.138),
    5: (10, 4838, [0.28, 0.45, 0.11, 0.13, 0.13], 0.221),
}


def episodes(path):
    """(timestep, episode return) per training episode."""
    out = []
    for line in open(path):
        try:
            r = json.loads(line.replace("NaN", "null"))
        except json.JSONDecodeError:
            continue  # partially written last line
        if r.get("Reward/Train_Reward") is not None:
            out.append((r["Timestep"], r["Reward/Train_Reward"]))
    return out


def stats(eps):
    first = next((t for t, r in eps if r > SOLVED), None)
    ttl, win = None, collections.deque(maxlen=500)
    for t, r in eps:
        win.append(r > SOLVED)
        if ttl is None and len(win) == 500 and sum(win) >= 250:
            ttl = t
    fifths = [[] for _ in range(5)]
    for t, r in eps:
        fifths[min(int(t // 200_000), 4)].append(r > SOLVED)
    last = [r > SOLVED for _, r in eps[-1000:]]
    return dict(step=eps[-1][0] if eps else 0, n=len(eps), first=first, ttl=ttl,
                overall=sum(r > SOLVED for _, r in eps) / max(len(eps), 1),
                recent=sum(last) / max(len(last), 1),
                fifths=[sum(f) / len(f) if f else None for f in fifths],
                cum=sum(r for _, r in eps))


def fmt(x, pct=False):
    if x is None:
        return "-"
    return f"{x:.0%}" if pct else f"{x:,}"


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    print(f"{'seed':>4} {'step':>9} {'eps':>8} {'1st treasure':>13} {'time-to-learn':>14} "
          f"{'solve all':>9} {'last 1k':>8} {'cum reward':>11}   fifths (new | original)")
    for s in range(1, 6):
        g = sorted(glob.glob(f"{root}/logdir/*/BTRL-postr_det_seed{s}/*/metrics.jsonl"),
                   key=lambda f: int(os.path.basename(os.path.dirname(f))))
        if not g:
            print(f"{s:>4}  no metrics yet")
            continue
        eps = []
        for f in g:   # a resumed run continues in the next numbered dir
            new = episodes(f)
            if new:
                eps = [e for e in eps if e[0] < new[0][0]] + new
        st = stats(eps)
        f0, t0, fifths0, all0 = REF[s]
        new = " ".join(fmt(x, True) for x in st["fifths"])
        old = " ".join(f"{x:.0%}" for x in fifths0)
        print(f"{s:>4} {st['step']:>9,} {st['n']:>8,} {fmt(st['first']):>6} ({f0:>4}) "
              f"{fmt(st['ttl']):>7} ({t0:>6,}) {st['overall']:>5.0%} ({all0:.0%}) {st['recent']:>6.0%} "
              f"{st['cum']:>11,.0f}   {new} | {old}")
    print("(original values in parentheses)")


if __name__ == "__main__":
    main()
