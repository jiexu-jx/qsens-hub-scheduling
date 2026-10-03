"""ICC paper, main experiment: random heterogeneous instances of K sensor clusters
sharing one quantum hub. For each instance we report the weighted MSE of the
policies, the closed-form (Prop. 2) and Lagrangian (Prop. 3) lower bounds, and the
classical baseline B0.

Policies
  B0   classical only
  B2   throughput priority (shortest expected completion), full GHZ, no threshold
  B2T  B2 + threshold rule (12)
  B3   Fidelity-Age index priority + threshold rule
  B4   FIFO (oldest outstanding request) + threshold rule
  TPW  throughput priority + our consumption rule (wait-and-subset, capped, cutoffs)
  PA   ours: marginal-precision priority + the same consumption rule
  PAM  marginal-precision priority + myopic subset consumption
"""
import json, math, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qsens import fastsim as F, index as X
from qsens.bound import lower_bound, envelope
from qsens.lagbound import lag_lower_bound
from qsens.physics import J_star, info_classical

T_EVAL, BURN, SEEDS = 150_000, 15_000, (11, 12, 13)
GRID = dict(N=(2, 3, 4, 5, 6), p=(0.1, 0.2, 0.3, 0.45, 0.6), w0=(0.95, 0.97, 0.99),
            rho=(0.99, 0.995), qI=(1e-4, 1e-3, 1e-2), beta=(1.0, 2.0, 5.0, 10.0))
KS = (2, 3, 4, 6)
PER_K = 30


def make_instances(seed=2026):
    rng = np.random.default_rng(seed)
    out = []
    for K in KS:
        for i in range(PER_K):
            tasks = []
            for k in range(K):
                N = int(rng.choice(GRID['N']))
                tasks.append(dict(N=N, p=float(rng.choice(GRID['p'])), w0=float(rng.choice(GRID['w0'])),
                                  rho=float(rng.choice(GRID['rho'])), q=float(rng.choice(GRID['qI'])) / N,
                                  v=1.0, beta=float(rng.choice(GRID['beta']))))
            Nmax = max(t['N'] for t in tasks)
            Qhi = max(Nmax, math.ceil(sum(t['N'] for t in tasks) / 2))
            Q = int(rng.integers(Nmax, Qhi + 1))
            out.append(dict(id=f'K{K}-{i:02d}', K=K, Q=Q, tasks=tasks))
    return out


_prof_cache, _lag_cache = {}, {}


def profile(tc, Q):
    """Cutoff maximising the single-cluster information rate under the capped
    wait-and-subset rule (q-independent, so it is cached), plus s and IEbar."""
    key = (tc.N, tc.p, tc.w0, tc.rho, min(tc.N, Q))
    if key not in _prof_cache:
        amax = int(min(200, 6 / (1 - tc.rho)))
        grid = sorted(set(np.unique(np.round(np.geomspace(1, amax, 20)).astype(int))))
        best = None
        for a in grid:
            pr = X.single_task_profile(tc, F.CONS_WAIT_CAP, int(a), T=150_000, Q=Q)
            if best is None or pr['meanI'] > best[1]['meanI'] + 1e-12:
                best = (int(a), pr)
        pm = X.single_task_profile(tc, F.CONS_MYOPIC_SUBSET, None, T=150_000, Q=Q)
        _prof_cache[key] = (best[0], best[1], pm)
    return _prof_cache[key]


def evaluate(scn, pol):
    rs = [F.run(scn, pol, T=T_EVAL, burn=BURN, seed=s) for s in SEEDS]
    return dict(wmse=float(np.mean([r['wmse'] for r in rs])),
                se=float(np.std([r['wmse'] for r in rs], ddof=1) / math.sqrt(len(rs))),
                deliv=np.mean([r['deliv'] for r in rs], axis=0).tolist(),
                meanm=np.mean([r['meanm'] for r in rs], axis=0).tolist())


def value_index(tc, Q):
    """MSE reduction per hub qubit if the cluster had the hub to itself (Prop. 2 envelope)."""
    h = envelope(tc.N, tc.w0, tc.v)[0]
    IC = info_classical(tc.N, tc.v)
    m = min(tc.N, Q)
    return tc.beta * (1 / J_star(IC, tc.q) - 1 / J_star(IC + float(h(tc.p * m)), tc.q)) / m


def run_instance(inst):
    t0 = time.time()
    tasks = [F.TaskCfg(**t) for t in inst['tasks']]
    Q, K = inst['Q'], inst['K']
    scn = F.Scenario(tasks, Q, inst['id'])
    profs = [profile(tc, Q) for tc in tasks]
    cut = tuple(p[0] for p in profs)
    IEw = tuple(p[1]['IEbar'] for p in profs)
    IEm = tuple(p[2]['IEbar'] for p in profs)
    none = (None,) * K
    pols = [
        F.PolicyCfg('B0', F.PRIO_STATIC, F.CONS_FULL_ALWAYS, cutoff=(-1,) * K),
        F.PolicyCfg('B2', F.PRIO_THROUGHPUT, F.CONS_FULL_ALWAYS, cutoff=none),
        F.PolicyCfg('B2T', F.PRIO_THROUGHPUT, F.CONS_FULL_THRESH, cutoff=none),
        F.PolicyCfg('B3', F.PRIO_FRESH, F.CONS_FULL_THRESH, cutoff=none, fresh_beta=0.1),
        F.PolicyCfg('B4', F.PRIO_FIFO, F.CONS_FULL_THRESH, cutoff=none),
        F.PolicyCfg('TPW', F.PRIO_THROUGHPUT, F.CONS_WAIT_CAP, cutoff=cut),
        F.PolicyCfg('PA', F.PRIO_MARGINAL, F.CONS_WAIT_CAP, cutoff=cut, IEbar=IEw),
        F.PolicyCfg('PAM', F.PRIO_MARGINAL, F.CONS_MYOPIC_SUBSET, cutoff=none, IEbar=IEm),
    ]
    res = {p.name: evaluate(scn, p) for p in pols}
    lb_s = lower_bound(tasks, Q)['lb']
    lag = lag_lower_bound(tasks, Q, A=10, cache=_lag_cache)
    vi = [value_index(tc, Q) for tc in tasks]
    out = dict(inst, cutoffs=list(cut), results=res, lb_simple=lb_s, lb_lag=lag['lb'], lag_M=lag['M'],
               value_index=vi, seconds=time.time() - t0)
    print(f"{inst['id']} Q={Q} B0={res['B0']['wmse']:.5f} B2T={res['B2T']['wmse']:.5f} "
          f"TPW={res['TPW']['wmse']:.5f} PA={res['PA']['wmse']:.5f} LB={lag['lb']:.5f} ({out['seconds']:.0f}s)", flush=True)
    return out


if __name__ == '__main__':
    insts = make_instances()
    if len(sys.argv) > 1:
        insts = insts[:int(sys.argv[1])]
    with Pool(2) as pool:
        out = pool.map(run_instance, insts, chunksize=1)
    json.dump(out, open(ROOT / 'results' / 'icc_random.json', 'w'), indent=1)
