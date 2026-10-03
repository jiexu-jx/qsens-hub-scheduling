"""ICC paper: evaluate the Lagrangian index policy (LIP) on the saved toy, regime and
random instances (adds results['LIP'] = staleness-aware LIP and results['LIPs'] = static LIP), using
the cached per-cluster frontiers."""
import json, math, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from qsens import fastsim as F
from qsens.lip import build_lip, run_lip
T_EVAL, BURN, SEEDS = 150_000, 15_000, (11, 12, 13)
CF = ROOT / 'results' / 'frontier_cache.json'


def work(x):
    cache = json.load(open(CF))
    t0 = time.time()
    tasks = [F.TaskCfg(**t) for t in x['tasks']]
    scn = F.Scenario(tasks, x['Q'], x['id'])
    lip = build_lip(scn, cache=cache)
    out = {}
    for name, dy in (('LIP', True), ('LIPs', False)):
        rs = [run_lip(scn, lip, T=T_EVAL, burn=BURN, seed=s, price_free=True, dyn=dy) for s in SEEDS]
        out[name] = dict(wmse=float(np.mean([r['wmse'] for r in rs])),
                         se=float(np.std([r['wmse'] for r in rs], ddof=1) / math.sqrt(len(rs))),
                         deliv=np.mean([r['deliv'] for r in rs], axis=0).tolist())
    out['setup'] = dict(mustar=lip['setup']['mustar'], lam=lip['setup']['lam'], muk=lip['setup']['muk'],
                        lb=float(lip['setup']['lb']), M=[float(m) for m in lip['setup']['M']])
    print(f"{x['id']}: LIP={out['LIP']['wmse']:.5f} PA={x['results']['PA']['wmse']:.5f} LB={x['lb_lag']:.5f} ({time.time()-t0:.0f}s)", flush=True)
    return out


if __name__ == '__main__':
    for fn in sys.argv[1:]:
        d = json.load(open(fn))
        single = isinstance(d, dict)
        items = [d] if single else d
        with Pool(2) as pool:
            outs = pool.map(work, items, chunksize=1)
        for x, o in zip(items, outs):
            x['results'].pop('LIPp', None)
            x['results']['LIP'] = o['LIP']; x['results']['LIPs'] = o['LIPs']; x['lip_setup'] = o['setup']
            assert x['lb_lag'] <= o['LIP']['wmse'] + 1e-12
        json.dump(d, open(fn, 'w'), indent=1)
        print('done', fn, flush=True)
