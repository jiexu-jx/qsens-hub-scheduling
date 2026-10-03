"""ICC paper: memory sizing from the Lagrangian bound, per-K capture of LIP, and run times."""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from qsens import fastsim as F
from qsens.lagbound import lag_lower_bound
from qsens.lip import build_lip, run_lip, lip_setup
CF = ROOT / 'results' / 'frontier_cache.json'
cache = json.load(open(CF))
out = {}
# ---- memory sizing on the regime hub
rg = {round(x['p'], 2): x for x in json.load(open(ROOT / 'results' / 'icc_regime.json'))}
out['memsize'] = {}
for p in (0.3, 0.6, 0.8):
    x = rg[p]; tasks = [F.TaskCfg(**t) for t in x['tasks']]
    B0 = x['results']['B0']['wmse']; rows = []
    for Q in range(2, 14):
        t0 = time.time(); lb = lag_lower_bound(tasks, Q, cache=cache)['lb']; dt = time.time() - t0
        st = lip_setup(tasks, Q, cache=cache)
        rows.append(dict(Q=Q, lb=lb, head=100 * (1 - lb / B0), mustar=st['mustar'], sec=dt))
    out['memsize'][p] = rows
    print(p, [(r['Q'], round(r['head'], 1)) for r in rows], flush=True)
json.dump(cache, open(CF, 'w'))
# ---- per-K capture
d = json.load(open(ROOT / 'results' / 'icc_random.json'))
out['perK'] = {}
for K in sorted(set(x['K'] for x in d)):
    row = {}
    for pol in ('LIP', 'PA', 'TPW', 'B2T', 'B3'):
        c = [100 * (x['results']['B0']['wmse'] - x['results'][pol]['wmse']) / (x['results']['B0']['wmse'] - x['lb_lag'])
             for x in d if x['K'] == K]
        row[pol] = float(np.median(c))
    out['perK'][K] = row
print(out['perK'])
# ---- run times on the largest instances (fresh, no cache)
big = [x for x in d if x['K'] == 6][:5]
tb, tt, ts = [], [], []
for x in big:
    tasks = [F.TaskCfg(**t) for t in x['tasks']]; scn = F.Scenario(tasks, x['Q'], x['id'])
    t0 = time.time(); st = lip_setup(tasks, x['Q'], cache={}); tb.append(time.time() - t0)
    t0 = time.time(); lip = build_lip(scn, cache={}, setup=st); tt.append(time.time() - t0)
    run_lip(scn, lip, T=2000, burn=0, seed=1, dyn=True)  # jit warm-up
    t0 = time.time(); run_lip(scn, lip, T=200_000, burn=0, seed=1, dyn=True); ts.append((time.time() - t0) / 2e5)
out['runtime'] = dict(bound_s=float(np.median(tb)), bound_max_s=float(max(tb)), tables_s=float(np.median(tt)),
                      tables_max_s=float(max(tt)), per_slot_us=float(np.median(ts) * 1e6),
                      sumN=[sum(t['N'] for t in x['tasks']) for x in big], Q=[x['Q'] for x in big])
print(out['runtime'])
json.dump(out, open(ROOT / 'results' / 'icc_memsize.json', 'w'), indent=1)
