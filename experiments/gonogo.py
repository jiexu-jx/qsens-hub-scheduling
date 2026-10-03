"""E3 go/no-go: two sensing tasks sharing one repeater with limited memory.

For each scenario:
  1. per-task single-task profiles (best cutoff under the wait-and-subset rule,
     myopic profile) -> inputs for A1 (marginal score) and A2 (Whittle index);
  2. named policies B0-B4, A1, A1W, A2 evaluated on held-out seeds;
  3. tuned references (tuned on training seeds, evaluated on held-out seeds):
       B2T* = throughput priority, consumption rule + cutoffs tuned for MSE
       BEST = best policy found in the whole parametric family (priority rule,
              priority bias, consumption rule, cutoffs) -- the optimality proxy.
Pre-registered reading (fixed before the sweep, see results memo):
  * 'substantial'  : precision-aware (A1/A1W/A2) weighted MSE <= 0.90 x best of the
                     untuned B2/B2T/B3/B4;
  * 'near-optimal' : B2T* within 2% of BEST  -> throughput+threshold suffices.
"""
import itertools, json, math, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qsens import fastsim as F, index as X
from qsens.physics import J_star

T_TUNE, T_EVAL, BURN = 100_000, 200_000, 20_000
TRAIN_SEEDS = (101, 102)
TEST_SEEDS = (1, 2, 3, 4)
W0 = 0.97
REGIMES = {'R1': dict(p=0.1, rho=0.99), 'R2': dict(p=0.3, rho=0.99), 'R3': dict(p=0.6, rho=0.995)}


def task(N, p, rho, qI=1e-3, beta=1.0, w0=W0):
    return F.TaskCfg(N=N, p=p, w0=w0, rho=rho, q=qI / N, v=1.0, beta=beta)


def scenarios():
    out = []
    for rn, r in REGIMES.items():
        p, rho = r['p'], r['rho']
        out += [
            F.Scenario([task(4, p, rho), task(4, p, rho)], 4, f'T1-sym-{rn}'),
            F.Scenario([task(2, p, rho), task(6, p, rho)], 6, f'T2-size-{rn}'),
            F.Scenario([task(4, p, rho), task(4, p, rho, beta=5.0)], 4, f'T3-weight-{rn}'),
            F.Scenario([task(4, p, rho), task(4, p / 3, rho)], 4, f'T4-link-{rn}'),
            F.Scenario([task(2, p, rho), task(6, p / 2, rho, beta=5.0)], 6, f'T5-conflict-{rn}'),
            F.Scenario([task(4, p, rho, qI=1e-4), task(4, p, rho, qI=1e-2)], 4, f'T6-drift-{rn}'),
            F.Scenario([task(4, p, rho), task(4, p, rho)], 6, f'T7-sym-Q6-{rn}'),
            F.Scenario([task(4, p, rho, w0=0.88), task(4, p, rho, w0=0.99)], 4, f'T8-quality-{rn}'),
            F.Scenario([task(4, p, rho), task(4, p, rho, beta=5.0)], 2, f'T9-weight-Q2-{rn}'),
            F.Scenario([task(2, p, rho), task(6, p / 2, rho, beta=5.0)], 3, f'T10-conflict-Q3-{rn}'),
        ]
    r = REGIMES['R2']
    out += [
        F.Scenario([task(4, r['p'], r['rho'], qI=0.1), task(4, r['p'], r['rho'], qI=0.1)], 4, 'T1-sym-fastdrift-R2'),
        F.Scenario([task(4, r['p'], r['rho'], qI=0.1), task(4, r['p'], r['rho'], qI=0.1, beta=5.0)], 4, 'T3-weight-fastdrift-R2'),
    ]
    return out


def evaluate(scn, pol, seeds, T):
    rs = [F.run(scn, pol, T=T, burn=BURN, seed=s) for s in seeds]
    agg = {k: float(np.mean([r[k] for r in rs])) for k in ('wmse', 'throughput', 'useful_throughput', 'disc', 'gen', 'wasted_consumed')}
    agg['wmse_se'] = float(np.std([r['wmse'] for r in rs], ddof=1) / math.sqrt(len(rs))) if len(rs) > 1 else 0.0
    for k in ('mse', 'deliv', 'useful', 'meanV', 'meanm', 'mse_p95'):
        agg[k] = np.mean([r[k] for r in rs], axis=0).tolist()
    return agg


def pcfg_dict(p):
    return dict(name=p.name, prio=p.prio, cons=p.cons, cutoff=list(p.cutoff),
                bias=list(p.bias) if p.bias is not None else None)


def run_scenario(scn):
    t0 = time.time()
    K = len(scn.tasks)
    info = {'name': scn.name, 'Q_R': scn.Q_R,
            'tasks': [dict(N=t.N, p=t.p, w0=t.w0, rho=t.rho, q=t.q, beta=t.beta) for t in scn.tasks]}
    # ---- single-task profiles
    prof_w, prof_m, cut = [], [], []
    for tc in scn.tasks:
        a, pw = X.best_single_task_cutoff(tc, F.CONS_WAIT_SUBSET)
        cut.append(a)
        prof_w.append(pw)
        prof_m.append(X.single_task_profile(tc, F.CONS_MYOPIC_SUBSET, None))
    info['single'] = dict(cutoff=cut, wait=prof_w, myopic=prof_m)
    Jg, Wt, viol = [], [], []
    for tc, pw in zip(scn.tasks, prof_w):
        jg, w, inf = X.whittle_table(tc, pw['s'], pw['IEbar'])
        Jg.append(jg); Wt.append(w); viol.append(inf['indexability_violations'])
    whittle = (np.array(Jg), np.array(Wt))
    info['whittle_indexability_violations'] = viol
    IE_w = tuple(p['IEbar'] for p in prof_w)
    IE_m = tuple(p['IEbar'] for p in prof_m)
    none = (None,) * K
    named = [
        F.PolicyCfg('B0', F.PRIO_STATIC, F.CONS_FULL_ALWAYS, cutoff=(-1,) * K),
        F.PolicyCfg('B1', F.PRIO_RANDOM, F.CONS_FULL_ALWAYS, cutoff=none),
        F.PolicyCfg('B2', F.PRIO_THROUGHPUT, F.CONS_FULL_ALWAYS, cutoff=none),
        F.PolicyCfg('B2T', F.PRIO_THROUGHPUT, F.CONS_FULL_THRESH, cutoff=none),
        F.PolicyCfg('B3', F.PRIO_FRESH, F.CONS_FULL_THRESH, cutoff=none, fresh_beta=0.1),
        F.PolicyCfg('B4', F.PRIO_FIFO, F.CONS_FULL_THRESH, cutoff=none),
        F.PolicyCfg('A1', F.PRIO_MARGINAL, F.CONS_MYOPIC_SUBSET, cutoff=none, IEbar=IE_m),
        F.PolicyCfg('A1W', F.PRIO_MARGINAL, F.CONS_WAIT_SUBSET, cutoff=tuple(cut), IEbar=IE_w),
        F.PolicyCfg('A2', F.PRIO_WHITTLE, F.CONS_WAIT_SUBSET, cutoff=tuple(cut), whittle=whittle),
    ]
    res = {}
    for pol in named:
        res[pol.name] = evaluate(scn, pol, TEST_SEEDS, T_EVAL)
        res[pol.name]['cfg'] = pcfg_dict(pol)

    # ---- tuning
    def tune(cands):
        scored = []
        for pol in cands:
            w = evaluate(scn, pol, TRAIN_SEEDS, T_TUNE)['wmse']
            scored.append((w, pol))
        scored.sort(key=lambda x: x[0])
        return scored

    mults = (0.5, 1.0, 2.0, 4.0, None)
    def cut_grid():
        for ms in itertools.product(mults, repeat=K):
            yield tuple(None if m is None else max(1, int(round(m * c))) for m, c in zip(ms, cut))
    conss = (F.CONS_FULL_THRESH, F.CONS_WAIT_SUBSET, F.CONS_MYOPIC_SUBSET)
    # B2T*: throughput priority, tuned consumption + cutoffs
    cands = []
    for cons in conss:
        for cg in cut_grid():
            cands.append(F.PolicyCfg('B2T*', F.PRIO_THROUGHPUT, cons, cutoff=cg, IEbar=IE_w))
    b2ts = tune(cands)
    # BEST: stage 1 priority x bias at default consumption
    biases = (1 / 16, 1 / 4, 1.0, 4.0, 16.0)
    st1 = []
    for prio in (F.PRIO_THROUGHPUT, F.PRIO_MARGINAL, F.PRIO_WHITTLE):
        for b in biases:
            st1.append(F.PolicyCfg('BEST', prio, F.CONS_WAIT_SUBSET, cutoff=tuple(cut),
                                   bias=(1.0, b), IEbar=IE_w, whittle=whittle))
    for b in (0.5, 2.0):
        st1.append(F.PolicyCfg('BEST', F.PRIO_STATIC, F.CONS_WAIT_SUBSET, cutoff=tuple(cut), bias=(1.0, b)))
    s1 = tune(st1)
    st2 = []
    for _, base in s1[:3]:
        for cons in conss:
            for cg in cut_grid():
                st2.append(F.PolicyCfg('BEST', base.prio, cons, cutoff=cg, bias=base.bias,
                                       IEbar=IE_w, whittle=whittle))
    s2 = tune(st2 + [b2ts[0][1]])
    best_pol = s2[0][1]
    best_b2t = b2ts[0][1]
    for nm, pol in (('B2T*', best_b2t), ('BEST', best_pol)):
        pol = F.PolicyCfg(nm, pol.prio, pol.cons, pol.cutoff, pol.m_min, pol.bias, pol.fresh_beta,
                          pol.IEbar, pol.whittle)
        res[nm] = evaluate(scn, pol, TEST_SEEDS, T_EVAL)
        res[nm]['cfg'] = pcfg_dict(pol)
    info['tuning'] = dict(stage1=[(w, pcfg_dict(p)) for w, p in s1[:6]],
                          stage2_top=[(w, pcfg_dict(p)) for w, p in s2[:6]],
                          b2t_top=[(w, pcfg_dict(p)) for w, p in b2ts[:6]])
    info['results'] = res
    info['seconds'] = time.time() - t0
    print(f"{scn.name}: {info['seconds']:.0f}s  " +
          '  '.join(f"{k}={v['wmse']:.5f}" for k, v in res.items()), flush=True)
    return info


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    scns = scenarios()
    if which != 'all':
        scns = [s for s in scns if any(w in s.name for w in which.split(','))]
    with Pool(2) as pool:
        out = pool.map(run_scenario, scns, chunksize=1)
    tag = 'all' if which == 'all' else which.replace(',', '_')
    json.dump(out, open(ROOT / 'results' / f'gonogo_{tag}.json', 'w'), indent=1)
