"""Add LIP and B2T performance to the memory-sizing curves (results/icc_memsize.json)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from qsens import fastsim as F
from qsens.lip import build_lip, run_lip
cache = json.load(open(ROOT / 'results' / 'frontier_cache.json'))
o = json.load(open(ROOT / 'results' / 'icc_memsize.json'))
rg = {round(x['p'], 2): x for x in json.load(open(ROOT / 'results' / 'icc_regime.json'))}
SEEDS = (31, 32, 33)
for p, rows in o['memsize'].items():
    x = rg[float(p)]; tasks = [F.TaskCfg(**t) for t in x['tasks']]; B0 = x['results']['B0']['wmse']
    for r in rows:
        scn = F.Scenario(tasks, r['Q'])
        lip = build_lip(scn, cache=cache)
        w = np.mean([run_lip(scn, lip, T=150_000, burn=15_000, seed=s, dyn=True)['wmse'] for s in SEEDS])
        b = np.mean([F.run(scn, F.PolicyCfg('B2T', F.PRIO_THROUGHPUT, F.CONS_FULL_THRESH, cutoff=(None,) * 3),
                           T=150_000, burn=15_000, seed=s)['wmse'] for s in SEEDS])
        r['LIP'] = 100 * (1 - w / B0); r['B2T'] = 100 * (1 - b / B0)
        assert w >= r['lb'] - 1e-12
    print(p, [(r['Q'], round(r['head'], 1), round(r['LIP'], 1), round(r['B2T'], 1)) for r in rows], flush=True)
json.dump(o, open(ROOT / 'results' / 'icc_memsize.json', 'w'), indent=1)
