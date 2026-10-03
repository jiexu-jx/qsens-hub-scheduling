"""ICC paper, Sec. IV example: one LIP slot for the Fig. 1 state (regime hub, p=0.8, Q=8).
Cluster 3 holds four age-1 pairs, cluster 2 one age-2 pair; every cluster at its target J."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from qsens import fastsim as F
from qsens.lip import lip_setup, cluster_tables
from qsens.lagbound import _encode_desc
cache = json.load(open(ROOT / 'results' / 'frontier_cache.json'))
x = {round(r['p'], 2): r for r in json.load(open(ROOT / 'results' / 'icc_regime.json'))}[0.8]
tasks = [F.TaskCfg(**t) for t in x['tasks']]; Q = 8
st = lip_setup(tasks, Q, cache=cache)
tabs = [cluster_tables(t.N, t.p, t.w0, t.rho, Q, mu, t.v, 10) for t, mu in zip(tasks, st['muk'])]
inv = [[], [2], [1, 1, 1, 1]]
s = [tb['lut'][_encode_desc(np.array(a + [0] * 8, dtype=np.int64), len(a), tb['B'])] for tb, a in zip(tabs, inv)]
free = Q - sum(map(len, inv)); xk = [0, 0, 0]; steps = []
while free > 0:
    v = [st['lam'][k] * (tabs[k]['Qx'][s[k], xk[k] + 1] - tabs[k]['Qx'][s[k], xk[k]] + st['muk'][k])
         if len(inv[k]) + xk[k] + 1 <= min(tasks[k].N, Q) else -np.inf for k in range(3)]
    k = int(np.argmax(v)); steps.append(dict(vals=[float(a) for a in v], pick=k))
    if v[k] <= 0: break
    xk[k] += 1; free -= 1
dec = {k: [(sc, bin(int(tabs[k]['Dmask'][s[k], sc])).count('1'), int(tabs[k]['Dkeep'][s[k], sc])) for sc in range(xk[k] + 1)]
       for k in range(3)}
v0 = steps[0]['vals']
out = dict(alloc=xk, steps=steps, decisions=dec, r32=v0[2] / v0[1], r31=v0[2] / v0[0],
           r33=steps[1]['vals'][2] / v0[2])
print(json.dumps(out, indent=1))
json.dump(out, open(ROOT / 'results' / 'icc_walkthrough.json', 'w'), indent=1)
