"""B5 oracle: exact dynamic programming on a toy go/no-go instance.

Two tasks with N_k = 2 sensors each share a repeater with Q_R = 2 memory qubits.
A GHZ (here: Bell) delivery for task k consumes both of its pairs (m = 1 hybrids
are never beneficial for N = 2). State at the start of a slot:
    inventory class  x  (exploiting within-task symmetry; ages capped at A where a
                         pair can no longer pass threshold (12) even with a fresh
                         partner),
    information states (J_0, J_1) on a log grid (bilinear interpolation).
Slot = attempt decision (how many links of each task attempt) -> heralds ->
consume/discard decision -> sensing + Kalman update -> ageing, exactly as in the
simulator. Average cost sum_k beta_k / J_k is minimised by relative value
iteration. The optimal policy is then executed with *continuous* J (one-step
look-ahead on the interpolated relative value function) in an exact simulation,
and compared with the simulator's policies on the same instance.
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qsens import fastsim as F, index as X
from qsens.physics import J_star

# ---------------------------------------------------------------- instance
P_LINK = 0.3
RHO = 0.99
W0 = 0.97
QI = 1e-3
BETA = np.array([1.0, 5.0])
N = 2
Q = QI / N
IC = 2.0
VSTAR = 1 / math.sqrt(2)
A = int(math.floor(math.log(VSTAR / W0 ** 2) / math.log(RHO)))  # max useful age
while W0 ** 2 * RHO ** (A + 1) > VSTAR:
    A += 1
NA = A + 1  # ages 0..A


# inventory classes ------------------------------------------------------------
# kind 0: empty
# kind 1: one pair of task k, age a            -> (1, k, a, -1)
# kind 2: two pairs of task k, ages a >= b     -> (2, k, a, b)
# kind 3: one pair each, ages a (task0), b (task1) -> (3, -1, a, b)
def build_classes():
    cl = [(0, -1, -1, -1)]
    for k in range(2):
        for a in range(NA):
            cl.append((1, k, a, -1))
    for k in range(2):
        for a in range(NA):
            for b in range(a + 1):
                cl.append((2, k, a, b))
    for a in range(NA):
        for b in range(NA):
            cl.append((3, -1, a, b))
    return cl


CL = build_classes()
IDX = {c: i for i, c in enumerate(CL)}
NC = len(CL)


def counts(c):
    kind, k, a, b = c
    if kind == 0:
        return [[], []]
    if kind == 1:
        out = [[], []]; out[k] = [a]; return out
    if kind == 2:
        out = [[], []]; out[k] = [a, b]; return out
    return [[a], [b]]


def make_class(ages):
    """ages: [list task0 ages, list task1 ages] -> class index (drops ages > A)."""
    ages = [[x for x in ages[0] if x <= A], [x for x in ages[1] if x <= A]]
    n0, n1 = len(ages[0]), len(ages[1])
    if n0 == 0 and n1 == 0:
        return IDX[(0, -1, -1, -1)]
    if n0 + n1 == 1:
        k = 0 if n0 else 1
        return IDX[(1, k, ages[k][0], -1)]
    if n0 == 2:
        a, b = sorted(ages[0], reverse=True); return IDX[(2, 0, a, b)]
    if n1 == 2:
        a, b = sorted(ages[1], reverse=True); return IDX[(2, 1, a, b)]
    return IDX[(3, -1, ages[0][0], ages[1][0])]


# Enumerate, for every class: attempt options (x0, x1), herald outcomes with
# probabilities, and for each outcome the list of post-decision options
# (next class after ageing, consumed V per task).  Flattened into arrays for numba.
def enumerate_transitions():
    att_start, out_start, dec_start = [0], [], []
    out_prob, dec_next, dec_V0, dec_V1 = [], [], [], []
    att_ptr = []  # per class: [start, end) into attempt list
    att_out_ptr = []  # per attempt option: [start,end) into outcomes
    out_dec_ptr = []  # per outcome: [start,end) into decisions
    for c in CL:
        ages = counts(c)
        stored = len(ages[0]) + len(ages[1])
        free = Q_R_TOY - stored
        empty = [N - len(ages[0]), N - len(ages[1])]
        a0 = len(att_out_ptr)
        for x0 in range(empty[0] + 1):
            for x1 in range(empty[1] + 1):
                if x0 + x1 > free:
                    continue
                o0 = len(out_dec_ptr)
                for s0 in range(x0 + 1):
                    for s1 in range(x1 + 1):
                        pr = (math.comb(x0, s0) * P_LINK ** s0 * (1 - P_LINK) ** (x0 - s0) *
                              math.comb(x1, s1) * P_LINK ** s1 * (1 - P_LINK) ** (x1 - s1))
                        new = [ages[0] + [0] * s0, ages[1] + [0] * s1]
                        d0 = len(dec_next)
                        # decisions: per task: 'keep', 'consume' (if 2 present), 'drop oldest', 'drop all'
                        opts = []
                        for k in range(2):
                            ok = [('keep',)]
                            if len(new[k]) == 2:
                                ok.append(('consume',))
                            if len(new[k]) >= 1:
                                ok.append(('drop1',))
                            if len(new[k]) == 2:
                                ok.append(('drop2',))
                            opts.append(ok)
                        for o_a in opts[0]:
                            for o_b in opts[1]:
                                Vs = [0.0, 0.0]
                                rem = [list(new[0]), list(new[1])]
                                for k, o in ((0, o_a), (1, o_b)):
                                    if o[0] == 'consume':
                                        Vs[k] = (W0 * RHO ** rem[k][0]) * (W0 * RHO ** rem[k][1])
                                        rem[k] = []
                                    elif o[0] == 'drop1':
                                        rem[k] = sorted(rem[k])[:-1]
                                    elif o[0] == 'drop2':
                                        rem[k] = []
                                nxt = make_class([[x + 1 for x in rem[0]], [x + 1 for x in rem[1]]])
                                dec_next.append(nxt); dec_V0.append(Vs[0]); dec_V1.append(Vs[1])
                        out_prob.append(pr)
                        out_dec_ptr.append((d0, len(dec_next)))
                att_out_ptr.append((o0, len(out_dec_ptr)))
        att_ptr.append((a0, len(att_out_ptr)))
    return (np.array(att_ptr, dtype=np.int64), np.array(att_out_ptr, dtype=np.int64),
            np.array(out_dec_ptr, dtype=np.int64), np.array(out_prob),
            np.array(dec_next, dtype=np.int64), np.array(dec_V0), np.array(dec_V1))


Q_R_TOY = 2


@njit(cache=True)
def _bilinear(h, c, j0, j1, Jg):
    G = Jg.shape[0]
    # locate j0
    if j0 <= Jg[0]:
        i0, t0 = 0, 0.0
    elif j0 >= Jg[G - 1]:
        i0, t0 = G - 2, 1.0
    else:
        lo, hi = 0, G - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if Jg[mid] <= j0:
                lo = mid
            else:
                hi = mid
        i0, t0 = lo, (j0 - Jg[lo]) / (Jg[lo + 1] - Jg[lo])
    if j1 <= Jg[0]:
        i1, t1 = 0, 0.0
    elif j1 >= Jg[G - 1]:
        i1, t1 = G - 2, 1.0
    else:
        lo, hi = 0, G - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if Jg[mid] <= j1:
                lo = mid
            else:
                hi = mid
        i1, t1 = lo, (j1 - Jg[lo]) / (Jg[lo + 1] - Jg[lo])
    return ((1 - t0) * (1 - t1) * h[c, i0, i1] + t0 * (1 - t1) * h[c, i0 + 1, i1] +
            (1 - t0) * t1 * h[c, i0, i1 + 1] + t0 * t1 * h[c, i0 + 1, i1 + 1])


@njit(cache=True)
def q_value_state(h, c, j0, j1, Jg, att_ptr, att_out_ptr, out_dec_ptr, out_prob,
                  dec_next, dec_V0, dec_V1, q, beta, IC):
    """min over attempt options of E_outcome[min over decisions (cost + h(next))].
    Returns (value, best attempt option index)."""
    p0 = j0 / (1 + q * j0)
    p1 = j1 / (1 + q * j1)
    best = 1e300
    besta = -1
    for a in range(att_ptr[c, 0], att_ptr[c, 1]):
        ev = 0.0
        for o in range(att_out_ptr[a, 0], att_out_ptr[a, 1]):
            bd = 1e300
            for d in range(out_dec_ptr[o, 0], out_dec_ptr[o, 1]):
                V0 = dec_V0[d]; V1 = dec_V1[d]
                I0 = 4.0 * V0 * V0 if V0 > 0 else IC
                I1 = 4.0 * V1 * V1 if V1 > 0 else IC
                n0 = p0 + I0
                n1 = p1 + I1
                val = beta[0] / n0 + beta[1] / n1 + _bilinear(h, dec_next[d], n0, n1, Jg)
                if val < bd:
                    bd = val
            ev += out_prob[o] * bd
        if ev < best:
            best = ev
            besta = a
    return best, besta


@njit(cache=True)
def rvi_sweep(h, Jg, att_ptr, att_out_ptr, out_dec_ptr, out_prob, dec_next, dec_V0, dec_V1,
              q, beta, IC):
    NCc, G, _ = h.shape
    hn = np.empty_like(h)
    for c in range(NCc):
        for i in range(G):
            for j in range(G):
                v, _ = q_value_state(h, c, Jg[i], Jg[j], Jg, att_ptr, att_out_ptr, out_dec_ptr,
                                     out_prob, dec_next, dec_V0, dec_V1, q, beta, IC)
                hn[c, i, j] = v
    return hn


def solve(G=24, iters=3000, tol=1e-9, damp=0.5):
    tr = enumerate_transitions()
    Jc = J_star(IC, Q)
    Jg = np.geomspace(0.85 * Jc, 2.2 * Jc, G)
    h = np.zeros((NC, G, G))
    g_hist = []
    t0 = time.time()
    for it in range(iters):
        hn = rvi_sweep(h, Jg, *tr, Q, BETA, IC)
        hn = damp * hn + (1 - damp) * h  # aperiodicity transform
        ref = hn[0, G // 2, G // 2]
        diff = hn - h
        g_lo, g_hi = diff.min(), diff.max()
        h = hn - ref
        g_hist.append((g_lo / damp, g_hi / damp))
        if it % 50 == 0:
            print(it, g_lo / damp, g_hi / damp, f'{time.time() - t0:.0f}s', flush=True)
        if g_hi - g_lo < tol * damp:
            break
    return Jg, h, tr, g_hist


# ------------------------------------------------------------ continuous-J executor
def simulate_dp_policy(Jg, h, tr, T=300_000, burn=20_000, seed=0):
    att_ptr, att_out_ptr, out_dec_ptr, out_prob, dec_next, dec_V0, dec_V1 = tr
    rng = np.random.default_rng(seed)
    J = np.array([J_star(IC, Q)] * 2)
    c = IDX[(0, -1, -1, -1)]
    cost = 0.0
    n_del = np.zeros(2)
    for t in range(T):
        _, a = q_value_state(h, c, J[0], J[1], Jg, *tr, Q, BETA, IC)
        # sample outcome
        o0, o1 = att_out_ptr[a]
        u = rng.random(); acc = 0.0
        o = o1 - 1
        for oo in range(o0, o1):
            acc += out_prob[oo]
            if u < acc:
                o = oo; break
        p0 = J[0] / (1 + Q * J[0]); p1 = J[1] / (1 + Q * J[1])
        best, bd = 1e300, -1
        for d in range(out_dec_ptr[o, 0], out_dec_ptr[o, 1]):
            I0 = 4 * dec_V0[d] ** 2 if dec_V0[d] > 0 else IC
            I1 = 4 * dec_V1[d] ** 2 if dec_V1[d] > 0 else IC
            n0, n1 = p0 + I0, p1 + I1
            val = BETA[0] / n0 + BETA[1] / n1 + _bilinear(h, dec_next[d], n0, n1, Jg)
            if val < best:
                best, bd = val, d
        I0 = 4 * dec_V0[bd] ** 2 if dec_V0[bd] > 0 else IC
        I1 = 4 * dec_V1[bd] ** 2 if dec_V1[bd] > 0 else IC
        J = np.array([p0 + I0, p1 + I1])
        if t >= burn:
            cost += BETA[0] / J[0] + BETA[1] / J[1]
            n_del += [dec_V0[bd] > 0, dec_V1[bd] > 0]
        c = dec_next[bd]
    return cost / (T - burn), n_del / (T - burn)


def toy_scenario():
    return F.Scenario([F.TaskCfg(N=2, p=P_LINK, w0=W0, rho=RHO, q=Q, beta=BETA[0]),
                       F.TaskCfg(N=2, p=P_LINK, w0=W0, rho=RHO, q=Q, beta=BETA[1])], Q_R_TOY, 'toy')


if __name__ == '__main__':
    print('classes', NC, 'A', A)
    Jg, h, tr, g_hist = solve(G=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
    g_dp = 0.5 * (g_hist[-1][0] + g_hist[-1][1])
    sims = [simulate_dp_policy(Jg, h, tr, seed=s) for s in range(3)]
    dp_exec = float(np.mean([s[0] for s in sims]))
    # compare with simulator policies (reuse gonogo machinery)
    sys.path.insert(0, str(ROOT / 'experiments'))
    import gonogo as GN
    scn = toy_scenario()
    info = GN.run_scenario(scn)
    out = dict(instance=dict(p=P_LINK, rho=RHO, w0=W0, qI=QI, beta=BETA.tolist(), N=[2, 2], Q_R=2, A=A,
                             grid=len(Jg)),
               g_dp_grid=g_dp, g_hist_last=g_hist[-1], dp_policy_simulated=dp_exec,
               dp_policy_deliv=np.mean([s[1] for s in sims], axis=0).tolist(),
               policies={k: v['wmse'] for k, v in info['results'].items()},
               gonogo_info=info)
    json.dump(out, open(ROOT / 'results' / 'oracle_dp.json', 'w'), indent=1, default=float)
    print('DP grid g', g_dp, 'DP executed', dp_exec)
    for k, v in out['policies'].items():
        print(f'  {k:5s} {v:.6f}  gap to DP {100 * (v / dp_exec - 1):+.2f}%')
