"""ICC paper: scaling study for asymptotic optimality. Base instances (K=3 from the
random set) are replicated r times (r*K clusters, hub memory r*Q). The Lagrangian
bound per copy is unchanged; we report each policy's per-copy weighted MSE gap to it.
An asymptotically optimal policy's per-copy gap converges to the Jensen gap only."""
import json, math, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from qsens import fastsim as F
from qsens.lip import build_lip, run_lip, lip_setup
import icc_random as R

RS = (1, 2, 4, 8)
T, BURN, SEEDS = 150_000, 15_000, (21, 22, 23)
CF = ROOT / 'results' / 'frontier_cache.json'


def per_copy(scn, pol, r):
    return float(np.mean([F.run(scn, pol, T=T, burn=BURN, seed=s)['wmse'] for s in SEEDS])) / r


def work(x):
    cache = json.load(open(CF))
    base = [F.TaskCfg(**t) for t in x['tasks']]
    Q = x['Q']
    setup = lip_setup(base, Q, cache=cache)
    profs = [R.profile(tc, Q) for tc in base]
    out = dict(id=x['id'], lb=x['lb_lag'], B0=x['results']['B0']['wmse'], rows=[])
    for r in RS:
        tasks = base * r
        scn = F.Scenario(tasks, Q * r)
        sr = {**setup, 'lam': setup['lam'] * r, 'muk': setup['muk'] * r, 'Jtar': setup['Jtar'] * r,
              'M': setup['M'] * r}
        lip = build_lip(scn, cache=cache, setup=sr)
        row = dict(r=r)
        for nm, dy in (('LIP', True), ('LIPs', False)):
            row[nm] = float(np.mean([run_lip(scn, lip, T=T, burn=BURN, seed=s, dyn=dy)['wmse'] for s in SEEDS])) / r
        cut = tuple(p[0] for p in profs) * r
        IEw = tuple(p[1]['IEbar'] for p in profs) * r
        row['PA'] = per_copy(scn, F.PolicyCfg('PA', F.PRIO_MARGINAL, F.CONS_WAIT_CAP, cutoff=cut, IEbar=IEw), r)
        row['B2T'] = per_copy(scn, F.PolicyCfg('B2T', F.PRIO_THROUGHPUT, F.CONS_FULL_THRESH,
                                               cutoff=(None,) * len(tasks)), r)
        out['rows'].append(row)
        print(x['id'], r, {k: round(v, 5) for k, v in row.items() if k != 'r'}, 'LB', round(x['lb_lag'], 5), flush=True)
    return out


if __name__ == '__main__':
    d = json.load(open(ROOT / 'results' / 'icc_random.json'))
    base = [x for x in d if x['K'] == 3][:12]
    with Pool(2) as pool:
        out = pool.map(work, base, chunksize=1)
    json.dump(out, open(ROOT / 'results' / 'icc_scaling.json', 'w'), indent=1)
