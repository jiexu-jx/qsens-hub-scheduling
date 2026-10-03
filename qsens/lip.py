"""Lagrangian index policy (LIP) for K sensor clusters sharing one quantum hub
(ICC paper, Section IV).

The hub problem is a weakly coupled MDP: the clusters evolve independently given
their actions and are coupled only by the per-slot memory constraint
    sum_k (stored pairs_k + attempts_k) <= Q.
Relaxing it with a price mu* (the optimal dual variable of the Lagrangian bound L2)
decouples the clusters. Linearising each cluster's MSE term at its bound-optimal
information rate gives an exchange rate lambda_k (MSE per unit of information), so
cluster k faces the single-cluster MDP
    maximise  E[information gain] - mu_k * E[memory occupancy],   mu_k = mu*/lambda_k,
whose relative value function h_k defines
  * the consumption/keep/discard decision table D_k(s, succ) (greedy w.r.t. h_k), and
  * the attempt Q-values Q_k(s, x) for x attempts in inventory state s.
Each slot the hub hands out its free qubits one at a time to the cluster with the
largest marginal value  lambda_k [Q_k(s_k, x_k+1) - Q_k(s_k, x_k) + mu_k]  (MSE units,
memory price excluded), as long as it is positive; clusters then follow D_k.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from . import fastsim as F
from .lagbound import build, _sweep, _ih, _encode_desc, frontier_lines, separable_dual_min
from .physics import info_classical, J_star


# ---------------------------------------------------------------- per-cluster tables
@njit(cache=True)
def _tables(h, ages, nn, lut, N, Ncap, A, p, w0, rho, v, mu, binom):
    S = nn.shape[0]
    B = A + 1
    IC = N * v * v
    BP = np.full((S, Ncap + 1), -1e300)
    Dmask = np.zeros((S, Ncap + 1), dtype=np.int64)
    Dkeep = np.zeros((S, Ncap + 1), dtype=np.int64)
    Qx = np.full((S, Ncap + 1), -1e300)
    post = np.empty(N + 1, dtype=np.int64)
    rest = np.empty(N + 1, dtype=np.int64)
    buf = np.empty(N + 1, dtype=np.int64)
    for s in range(S):
        n = nn[s]
        for succ in range(Ncap - n + 1):
            npost = succ + n
            for i in range(succ):
                post[i] = 0
            for i in range(n):
                post[succ + i] = ages[s, n - 1 - i]
            best = -1e300
            bm, bk = 0, 0
            for mask in range(1 << npost):
                m = 0
                V = 1.0
                nr = 0
                for i in range(npost):
                    if (mask >> i) & 1:
                        m += 1
                        V *= w0 * rho ** post[i]
                    else:
                        rest[nr] = post[i]
                        nr += 1
                rew = 0.0
                if m > 0:
                    rew = _ih(N, m, V, v) - IC
                for k in range(nr + 1):
                    for i in range(k):
                        a = rest[i] + 1
                        buf[i] = a if a < A else A
                    code = _encode_desc(buf, k, B)
                    val = rew + h[lut[code]]
                    if val > best + 1e-13:
                        best = val
                        bm = mask
                        bk = k
            BP[s, succ] = best
            Dmask[s, succ] = bm
            Dkeep[s, succ] = bk
        for x in range(Ncap - n + 1):
            ev = 0.0
            for succ in range(x + 1):
                ev += binom[x, succ] * BP[s, succ]
            Qx[s, x] = ev - mu * (n + x)
    return Qx, Dmask, Dkeep


def cluster_tables(N, p, w0, rho, Q, mu, v=1.0, A=10, tol=1e-9, max_iter=20000):
    ages, nn, lut, Ncap = build(N, Q, A)
    binom = np.zeros((N + 1, N + 1))
    for x in range(N + 1):
        for s_ in range(x + 1):
            binom[x, s_] = math.comb(x, s_) * p ** s_ * (1 - p) ** (x - s_)
    h = np.zeros(len(nn))
    dummy = np.zeros(len(nn), dtype=np.int64)
    for it in range(max_iter):
        hn, bx = _sweep(h, ages, nn, lut, N, Ncap, A, p, w0, rho, v, mu, binom, dummy, False)
        hn = 0.5 * hn + 0.5 * h
        span = np.max(hn - h) - np.min(hn - h)
        h = hn - hn[0]
        if span < tol:
            break
    Qx, Dm, Dk = _tables(h, ages, nn, lut, N, Ncap, A, p, w0, rho, v, mu, binom)
    return dict(lut=lut, Qx=Qx, Dmask=Dm, Dkeep=Dk, Ncap=Ncap, B=A + 1, S=len(nn))


# ---------------------------------------------------------------- price and exchange rates
def lip_setup(tasks, Q, A=10, cache=None):
    """Solve the Lagrangian bound, return mu*, lambda_k, mu_k, M_k and L2."""
    tasks = list(tasks)
    IC = np.array([info_classical(t.N, t.v) for t in tasks])
    fs, X, Fk = [], [], []
    for k, t in enumerate(tasks):
        lines = np.array(frontier_lines(t, Q, A, cache))
        g, mu = lines[:, 0], lines[:, 1]
        Fk.append(lambda M, g=g, mu=mu: max(0.0, float(np.min(g + mu * M))))
        def f(M, t=t, ic=IC[k], F=Fk[-1]):
            I = ic + F(M)
            return t.beta / (I / 2 + math.sqrt(I * I / 4 + I / t.q))
        fs.append(f); X.append(float(min(t.N, Q)))
    lb, M, mustar = separable_dual_min(fs, X, Q, return_lam=True)
    lam, Jtar = [], []
    for k, t in enumerate(tasks):
        I = IC[k] + Fk[k](M[k])
        Jtar.append(J_star(I, t.q))
        e = 1e-6 * max(I, 1.0)
        fI = lambda I_, t=t: t.beta / (I_ / 2 + math.sqrt(I_ * I_ / 4 + I_ / t.q))
        lam.append(-(fI(I + e) - fI(I - e)) / (2 * e))
    lam = np.array(lam)
    muk = mustar / lam
    return dict(lb=lb, M=list(M), mustar=mustar, lam=lam.tolist(), muk=muk.tolist(), Jtar=Jtar)


# ---------------------------------------------------------------- simulation kernel
@njit(cache=True)
def run_lip_kernel(seed, T, burn, Q_R, link_task, link_p, link_w0, link_rho,
                   task_start, task_N, task_q, task_v, task_beta, task_J0,
                   lut_all, lut_off, Bk, Ncapk, Qx_all, Dm_all, Dk_all, S_off,
                   lamk, muk, use_price_free, bundle, dyn, Jtar):
    np.random.seed(seed)
    L = link_task.shape[0]
    K = task_N.shape[0]
    has = np.zeros(L, dtype=np.bool_)
    age = np.zeros(L, dtype=np.int64)
    J = task_J0.copy()
    IC = np.empty(K)
    for k in range(K):
        IC[k] = task_N[k] * task_v[k] ** 2
    sum_w = 0.0
    sum_mse = np.zeros(K)
    n_del = np.zeros(K)
    sum_I = np.zeros(K)
    st = np.zeros(K, dtype=np.int64)
    nst = np.zeros(K, dtype=np.int64)
    xk = np.zeros(K, dtype=np.int64)
    buf = np.empty(64, dtype=np.int64)
    lst = np.empty(64, dtype=np.int64)
    cand = np.empty(64, dtype=np.int64)
    n_stored = 0
    for t in range(T):
        # ---- state of each cluster (capped ages, descending)
        for k in range(K):
            s0 = task_start[k]
            n = 0
            for j in range(task_N[k]):
                l = s0 + j
                if has[l]:
                    a = age[l]
                    A = Bk[k] - 1
                    buf[n] = a if a < A else A
                    n += 1
            code = _encode_desc(buf, n, Bk[k])
            st[k] = lut_all[lut_off[k] + code]
            nst[k] = n
            xk[k] = 0
        # ---- allocation of free qubits
        free = Q_R - n_stored
        lamt = np.empty(K)
        for k in range(K):
            if dyn:
                Jp = J[k] / (1.0 + task_q[k] * J[k])
                r_ = Jtar[k] / Jp
                lamt[k] = lamk[k] * r_ * r_
            else:
                lamt[k] = lamk[k]
        if bundle:
            # pass 1: whole priced bundles x_k* = argmax_x Q_k(s,x), by value per qubit
            want = np.zeros(K, dtype=np.int64)
            vpq = np.full(K, -1e300)
            for k in range(K):
                row = S_off[k] + st[k]
                xmax = Ncapk[k] - nst[k]
                if task_N[k] - nst[k] < xmax:
                    xmax = task_N[k] - nst[k]
                bx, bq = 0, Qx_all[row, 0]
                for x in range(1, xmax + 1):
                    if Qx_all[row, x] > bq + 1e-15 * (1.0 + abs(bq)):
                        bq = Qx_all[row, x]; bx = x
                want[k] = bx
                if bx > 0:
                    vpq[k] = lamt[k] * (bq - Qx_all[row, 0] + muk[k] * bx) / bx
            order = np.argsort(-vpq)
            for oi in range(K):
                k = order[oi]
                if want[k] == 0 or free == 0:
                    continue
                g = want[k] if want[k] < free else free
                xk[k] = g
                free -= g
        # pass 2 (or only pass): unit greedy by marginal value
        while free > 0:
            bestv = 0.0
            bk = -1
            for k in range(K):
                x = xk[k]
                if nst[k] + x + 1 > Ncapk[k] or nst[k] + x + 1 > task_N[k]:
                    continue
                row = S_off[k] + st[k]
                q1 = Qx_all[row, x + 1]
                q0 = Qx_all[row, x]
                if q1 < -1e299:
                    continue
                val = lamt[k] * (q1 - q0 + (muk[k] if use_price_free else 0.0))
                if val > bestv + 1e-15 * (1.0 + abs(val)):
                    bestv = val
                    bk = k
            if bk < 0:
                break
            xk[bk] += 1
            free -= 1
        # ---- attempts and heralds
        succk = np.zeros(K, dtype=np.int64)
        for k in range(K):
            if xk[k] == 0:
                continue
            s0 = task_start[k]
            nc = 0
            for j in range(task_N[k]):
                if not has[s0 + j]:
                    cand[nc] = s0 + j
                    nc += 1
            for i in range(nc - 1, 0, -1):
                jj = np.random.randint(0, i + 1)
                tmp = cand[i]; cand[i] = cand[jj]; cand[jj] = tmp
            for i in range(xk[k]):
                l = cand[i]
                if np.random.random() < link_p[l]:
                    has[l] = True
                    age[l] = 0
                    n_stored += 1
                    succk[k] += 1
        # ---- consumption / keep / discard per the cluster's decision table
        wm = 0.0
        for k in range(K):
            s0 = task_start[k]
            N = task_N[k]
            # post-herald list ascending by age (new pairs have age 0)
            n = 0
            for j in range(N):
                l = s0 + j
                if has[l]:
                    lst[n] = l
                    n += 1
            for a_ in range(1, n):
                lv = lst[a_]
                b_ = a_ - 1
                while b_ >= 0 and age[lst[b_]] > age[lv]:
                    lst[b_ + 1] = lst[b_]
                    b_ -= 1
                lst[b_ + 1] = lv
            row = S_off[k] + st[k]
            mask = Dm_all[row, succk[k]]
            keep = Dk_all[row, succk[k]]
            m = 0
            V = 1.0
            kept = 0
            for i in range(n):
                l = lst[i]
                if (mask >> i) & 1:
                    m += 1
                    V *= link_w0[l] * link_rho[l] ** age[l]
                    has[l] = False
                    n_stored -= 1
                else:
                    if kept < keep:
                        kept += 1
                    else:
                        has[l] = False
                        n_stored -= 1
            if m > 0:
                I = _ih(N, m, V, task_v[k])
            else:
                I = IC[k]
            J[k] = J[k] / (1.0 + task_q[k] * J[k]) + I
            if t >= burn:
                wm += task_beta[k] / J[k]
                sum_mse[k] += 1.0 / J[k]
                sum_I[k] += I
                if m > 0:
                    n_del[k] += 1
        if t >= burn:
            sum_w += wm
        for l in range(L):
            if has[l]:
                age[l] += 1
    Tm = T - burn
    return sum_w / Tm, sum_mse / Tm, n_del / Tm, sum_I / Tm


def build_lip(scn, A=10, cache=None, setup=None):
    """Tables for all clusters. Identical clusters share tables; inventory lookup
    tables depend only on (min(N,Q), A) and are shared as well."""
    if setup is None:
        setup = lip_setup(scn.tasks, scn.Q_R, A, cache)
    tab_cache, lut_cache = {}, {}
    tabs = []
    for t, mu in zip(scn.tasks, setup['muk']):
        key = (t.N, t.p, t.w0, t.rho, min(t.N, scn.Q_R), t.v, round(mu, 12))
        if key not in tab_cache:
            tab_cache[key] = cluster_tables(t.N, t.p, t.w0, t.rho, scn.Q_R, mu, t.v, A)
        tabs.append(tab_cache[key])
    luts, lut_off = [], []
    off = 0
    for tb in tabs:
        if tb['Ncap'] not in lut_cache:
            lut_cache[tb['Ncap']] = off
            luts.append(tb['lut'])
            off += len(tb['lut'])
        lut_off.append(lut_cache[tb['Ncap']])
    lut_all = np.concatenate(luts)
    width = max(tb['Qx'].shape[1] for tb in tabs)
    def pad(a, fill):
        out = np.full((a.shape[0], width), fill, dtype=a.dtype)
        out[:, :a.shape[1]] = a
        return out
    Qx_all = np.concatenate([pad(tb['Qx'], -1e300) for tb in tabs])
    Dm_all = np.concatenate([pad(tb['Dmask'], 0) for tb in tabs])
    Dk_all = np.concatenate([pad(tb['Dkeep'], 0) for tb in tabs])
    S_off = np.cumsum([0] + [tb['S'] for tb in tabs[:-1]]).astype(np.int64)
    return dict(setup=setup, lut_all=lut_all, lut_off=np.array(lut_off, dtype=np.int64),
                Bk=np.array([tb['B'] for tb in tabs], dtype=np.int64),
                Ncapk=np.array([tb['Ncap'] for tb in tabs], dtype=np.int64),
                Qx_all=Qx_all, Dm_all=Dm_all, Dk_all=Dk_all, S_off=S_off,
                lamk=np.array(setup['lam']), muk=np.array(setup['muk']),
                Jtar=np.array(setup['Jtar']))


def run_lip(scn, lip, T=150_000, burn=15_000, seed=0, price_free=True, bundle=False, dyn=False):
    a = scn.arrays()
    J0 = np.array([J_star(t.N * t.v ** 2, t.q) for t in scn.tasks])
    out = run_lip_kernel(seed, T, burn, scn.Q_R, a['link_task'], a['link_p'], a['link_w0'],
                         a['link_rho'], a['task_start'], a['task_N'], a['task_q'], a['task_v'],
                         a['task_beta'], J0, lip['lut_all'], lip['lut_off'], lip['Bk'],
                         lip['Ncapk'], lip['Qx_all'], lip['Dm_all'], lip['Dk_all'],
                         lip['S_off'], lip['lamk'], lip['muk'], price_free, bundle, dyn, lip['Jtar'])
    return dict(wmse=float(out[0]), mse=out[1].tolist(), deliv=out[2].tolist(), meanI=out[3].tolist())
