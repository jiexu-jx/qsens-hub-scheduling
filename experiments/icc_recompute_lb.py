"""Recompute both lower bounds (weak-duality solver, exact dual frontier) for saved
instance files; frontiers are cached on disk in results/frontier_cache.json."""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from qsens import fastsim as F
from qsens.lagbound import lag_lower_bound
from qsens.bound import lower_bound
CF = ROOT / 'results' / 'frontier_cache.json'
cache = json.load(open(CF)) if CF.exists() else {}
for fn in sys.argv[1:]:
    d = json.load(open(fn))
    single = isinstance(d, dict)
    items = [d] if single else d
    for x in items:
        tasks = [F.TaskCfg(**t) for t in x['tasks']]
        r = lag_lower_bound(tasks, x['Q'], A=10, cache=cache)
        s = lower_bound(tasks, x['Q'])
        x['lb_lag'], x['lag_M'], x['lb_simple'] = r['lb'], r['M'], s['lb']
        best = min(v['wmse'] for v in x['results'].values())
        assert r['lb'] <= best + 1e-12 and s['lb'] <= r['lb'] + 1e-9, (x['id'], s['lb'], r['lb'], best)
        json.dump(cache, open(CF, 'w'))
    json.dump(d, open(fn, 'w'), indent=1)
    print('done', fn, len(items), flush=True)
