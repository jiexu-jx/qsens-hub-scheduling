"""Fast (numba) slot-level simulator for K sensing tasks (stars) sharing one
repeater R -- the go/no-go topology of Section VII-F.

Same model and slot order as the reference simulator in sim.py:
  1. attempts: R has Q_R memory qubits; every link without a stored pair is a
     candidate; free memory is handed to candidate links in policy-priority order
     (an attempt occupies one qubit at R and the sensor's qubit for the slot);
     success w.p. p_l, new pair age 0, w = w0_l.
  2. decisions: cutoff discards (age > cutoff_k), deadlock breaker, consumption
     (full / hybrid GHZ rooted at R, teleportation model: V = prod of w).
  3. sensing + Kalman update  J <- J/(1+qJ) + I.
  4. ageing: w <- w * rho_l (rho_l = eta_R eta_sensor).

Policies are a parametric family (see POLICY_* constants and make_policy()).
Cross-validated against sim.py in tests/test_crossval.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from .physics import prop4_ET

# priority types
PRIO_RANDOM, PRIO_THROUGHPUT, PRIO_FRESH, PRIO_FIFO, PRIO_MARGINAL, PRIO_WHITTLE, PRIO_STATIC = range(7)
# consumption rules
CONS_FULL_ALWAYS, CONS_FULL_THRESH, CONS_MYOPIC_SUBSET, CONS_WAIT_SUBSET, CONS_WAIT_CAP = range(5)
# CONS_WAIT_CAP: like CONS_WAIT_SUBSET but 'full' means min(N_k, Q_R) pairs, so the rule
# also works when the hub memory is smaller than the cluster (used in the ICC paper)


@njit(cache=True)
def _info_hybrid(N, m, V, v):
    if m <= 0:
        return N * v * v
    d = V * V * v ** (2 * m)
    if d <= 1e-300:
        return 0.0
    var_sum = 1.0 / d + (N - m) / (v * v)
    return N * N / var_sum


@njit(cache=True)
def _best_subset(ws_sorted, n, N, v, m_min):
    """ws_sorted: descending w of the n stored pairs. Returns (m*, I*) maximizing
    hybrid information over m in [m_min, n]; m*=0 if none (I* = I_C)."""
    best_m = 0
    best_I = N * v * v
    V = 1.0
    for m in range(1, n + 1):
        V *= ws_sorted[m - 1]
        if m >= m_min:
            I = _info_hybrid(N, m, V, v)
            if I > best_I:
                best_I = I
                best_m = m
    return best_m, best_I


@njit(cache=True)
def _interp(x, xs, ys):
    n = xs.shape[0]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[n - 1]:
        return ys[n - 1]
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    t = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] * (1 - t) + ys[hi] * t


@njit(cache=True)
def run_kernel(seed, T, burn, Q_R,
               link_task, link_p, link_w0, link_rho,
               task_start, task_N, task_q, task_v, task_beta, task_J0,
               prio_type, cons_rule, cutoff, m_min, bias, fresh_beta,
               ET_tab, IEbar, Jgrid, Wtab, sample_every):
    np.random.seed(seed)
    L = link_task.shape[0]
    K = task_N.shape[0]
    has = np.zeros(L, dtype=np.bool_)
    age = np.zeros(L, dtype=np.int64)
    J = task_J0.copy()
    Vstar = np.empty(K)
    IC = np.empty(K)
    for k in range(K):
        Vstar[k] = task_N[k] ** -0.5 * task_v[k] ** (-(task_N[k] - 1))
        IC[k] = task_N[k] * task_v[k] ** 2
    A_fresh = np.zeros(K)   # slots since last useful delivery (B3)
    A_fifo = np.zeros(K)    # slots since last delivery of any kind (B4)

    # accumulators
    sum_wmse = 0.0
    sum_mse = np.zeros(K)
    n_deliv = np.zeros(K)
    n_useful = np.zeros(K)
    n_ent_slots = np.zeros(K)
    sum_V = np.zeros(K)
    sum_V2 = np.zeros(K)
    sum_I = np.zeros(K)
    sum_m = np.zeros(K)
    n_disc = 0.0
    n_gen = 0.0
    n_wasted_consumed = 0.0
    nsamp = (T - burn) // sample_every + 1
    samples = np.zeros((nsamp, K))
    si = 0
    n_stored = 0
    score = np.empty(K)
    order = np.empty(K, dtype=np.int64)
    cand = np.empty(L, dtype=np.int64)
    wbuf = np.empty(L)
    idxbuf = np.empty(L, dtype=np.int64)
    MAXN = 0
    for k in range(K):
        if task_N[k] > MAXN:
            MAXN = task_N[k]

    for t in range(T):
        # ---------------- 1. priorities and attempts
        for k in range(K):
            s0 = task_start[k]
            miss = 0
            for j in range(task_N[k]):
                if not has[s0 + j]:
                    miss += 1
            if prio_type == PRIO_THROUGHPUT:
                sc = 1.0 / ET_tab[k, miss] if miss > 0 else 1e18
            elif prio_type == PRIO_FRESH:
                u = 1.0 / ET_tab[k, miss] if miss > 0 else 1.0
                sc = u * (1.0 + fresh_beta * A_fresh[k])
            elif prio_type == PRIO_FIFO:
                sc = A_fifo[k]
            elif prio_type == PRIO_MARGINAL:
                Jp = J[k] / (1.0 + task_q[k] * J[k])
                gain = task_beta[k] * (1.0 / (Jp + IC[k]) - 1.0 / (Jp + IEbar[k]))
                if gain < 0:
                    gain = 0.0
                sc = gain / ET_tab[k, miss] if miss > 0 else 1e18
            elif prio_type == PRIO_WHITTLE:
                sc = _interp(J[k], Jgrid[k], Wtab[k])
            else:  # STATIC or RANDOM (RANDOM handled below)
                sc = 1.0
            score[k] = sc * bias[k] + 1e-12 * np.random.random()
        # order tasks by descending score
        for k in range(K):
            order[k] = k
        for a in range(K):
            for b in range(a + 1, K):
                if score[order[b]] > score[order[a]]:
                    tmp = order[a]
                    order[a] = order[b]
                    order[b] = tmp
        nc = 0
        if prio_type == PRIO_RANDOM:
            for l in range(L):
                if not has[l]:
                    cand[nc] = l
                    nc += 1
            # shuffle
            for i in range(nc - 1, 0, -1):
                j = np.random.randint(0, i + 1)
                tmp = cand[i]
                cand[i] = cand[j]
                cand[j] = tmp
        else:
            for oi in range(K):
                k = order[oi]
                s0 = task_start[k]
                st = nc
                for j in range(task_N[k]):
                    if not has[s0 + j]:
                        cand[nc] = s0 + j
                        nc += 1
                # random order within task
                for i in range(nc - 1, st, -1):
                    j = st + np.random.randint(0, i - st + 1)
                    tmp = cand[i]
                    cand[i] = cand[j]
                    cand[j] = tmp
        free = Q_R - n_stored
        na = nc if nc < free else free
        for i in range(na):
            l = cand[i]
            if np.random.random() < link_p[l]:
                has[l] = True
                age[l] = 0
                n_stored += 1
                if t >= burn:
                    n_gen += 1

        # ---------------- 2. decisions
        # cutoff discards
        for l in range(L):
            if has[l] and age[l] > cutoff[link_task[l]]:
                has[l] = False
                n_stored -= 1
                if t >= burn:
                    n_disc += 1
        consumed_m = np.zeros(K, dtype=np.int64)
        consumed_V = np.zeros(K)
        for k in range(K):
            s0 = task_start[k]
            N = task_N[k]
            n = 0
            oldest = -1
            for j in range(N):
                l = s0 + j
                if has[l]:
                    wbuf[n] = link_w0[l] * link_rho[l] ** age[l]
                    idxbuf[n] = l
                    n += 1
                    if age[l] > oldest:
                        oldest = age[l]
            if n == 0:
                continue
            # sort descending by w (insertion sort, n small)
            for a in range(1, n):
                wv = wbuf[a]
                iv = idxbuf[a]
                b = a - 1
                while b >= 0 and wbuf[b] < wv:
                    wbuf[b + 1] = wbuf[b]
                    idxbuf[b + 1] = idxbuf[b]
                    b -= 1
                wbuf[b + 1] = wv
                idxbuf[b + 1] = iv
            m_use = 0
            discard_worst = False
            if cons_rule == CONS_FULL_ALWAYS:
                if n == N:
                    m_use = N
            elif cons_rule == CONS_FULL_THRESH:
                if n == N:
                    V = 1.0
                    for a in range(N):
                        V *= wbuf[a]
                    if V > Vstar[k]:
                        m_use = N
                    else:
                        discard_worst = True
            elif cons_rule == CONS_MYOPIC_SUBSET:
                m_b, I_b = _best_subset(wbuf, n, N, task_v[k], m_min[k])
                m_use = m_b
            elif cons_rule == CONS_WAIT_SUBSET:
                trigger = (n == N) or (oldest >= cutoff[k])
                if trigger:
                    m_b, I_b = _best_subset(wbuf, n, N, task_v[k], m_min[k])
                    m_use = m_b
            else:  # CONS_WAIT_CAP
                cap = N if N < Q_R else Q_R
                trigger = (n >= cap) or (oldest >= cutoff[k])
                if trigger:
                    m_b, I_b = _best_subset(wbuf, n, N, task_v[k], m_min[k])
                    m_use = m_b
            capk = N if (cons_rule != CONS_WAIT_CAP or N < Q_R) else Q_R
            if m_use == 0 and n >= capk and cons_rule != CONS_FULL_ALWAYS:
                # all pairs present but no beneficial delivery: waiting only decays
                # them further, so drop the worst one and regenerate it
                discard_worst = True
            if discard_worst:
                l = idxbuf[n - 1]
                has[l] = False
                n_stored -= 1
                if t >= burn:
                    n_disc += 1
            if m_use > 0:
                V = 1.0
                for a in range(m_use):
                    V *= wbuf[a]
                    has[idxbuf[a]] = False
                    n_stored -= 1
                consumed_m[k] = m_use
                consumed_V[k] = V
                # subset rules: pairs left out of the chosen subset are the worst ones
                # (they would lower the information); discard them so they do not
                # block their links indefinitely
                if cons_rule == CONS_MYOPIC_SUBSET or cons_rule == CONS_WAIT_SUBSET or cons_rule == CONS_WAIT_CAP:
                    for a in range(m_use, n):
                        has[idxbuf[a]] = False
                        n_stored -= 1
                        if t >= burn:
                            n_disc += 1
        # deadlock breaker: memory full and nobody can complete -> drop oldest pair
        if n_stored >= Q_R:
            anyfull = False
            for k in range(K):
                cnt = 0
                for j in range(task_N[k]):
                    if has[task_start[k] + j]:
                        cnt += 1
                if cnt == task_N[k]:
                    anyfull = True
            if not anyfull:
                lo = -1
                amax = -1
                for l in range(L):
                    if has[l] and age[l] > amax:
                        amax = age[l]
                        lo = l
                if lo >= 0:
                    has[lo] = False
                    n_stored -= 1
                    if t >= burn:
                        n_disc += 1

        # ---------------- 3. sensing + Kalman
        wm = 0.0
        for k in range(K):
            N = task_N[k]
            if consumed_m[k] > 0:
                I = _info_hybrid(N, consumed_m[k], consumed_V[k], task_v[k])
            else:
                I = IC[k]
            J[k] = J[k] / (1.0 + task_q[k] * J[k]) + I
            A_fresh[k] += 1
            A_fifo[k] += 1
            if consumed_m[k] > 0:
                A_fifo[k] = 0
                if I > IC[k]:
                    A_fresh[k] = 0
            if t >= burn:
                mse = 1.0 / J[k]
                sum_I[k] += I
                wm += task_beta[k] * mse
                sum_mse[k] += mse
                if consumed_m[k] > 0:
                    n_deliv[k] += 1
                    sum_V[k] += consumed_V[k]
                    sum_V2[k] += consumed_V[k] ** 2
                    sum_m[k] += consumed_m[k]
                    if I > IC[k]:
                        n_useful[k] += 1
                        n_ent_slots[k] += 1
                    else:
                        n_wasted_consumed += consumed_m[k]
        if t >= burn:
            sum_wmse += wm
            if (t - burn) % sample_every == 0 and si < nsamp:
                for k in range(K):
                    samples[si, k] = 1.0 / J[k]
                si += 1

        # ---------------- 4. ageing
        for l in range(L):
            if has[l]:
                age[l] += 1

    Tm = T - burn
    return (sum_wmse / Tm, sum_mse / Tm, n_deliv / Tm, n_useful / Tm,
            sum_V / np.maximum(n_deliv, 1), sum_V2 / np.maximum(n_deliv, 1), sum_I / Tm,
            sum_m / np.maximum(n_deliv, 1),
            n_disc / Tm, n_gen / Tm, n_wasted_consumed / Tm, samples[:si])


# ---------------------------------------------------------------------------
# Python-side configuration helpers
# ---------------------------------------------------------------------------

@dataclass
class TaskCfg:
    N: int
    p: float
    w0: float
    rho: float              # combined per-slot decay eta_R * eta_sensor
    q: float = 1e-4
    v: float = 1.0
    beta: float = 1.0


@dataclass
class Scenario:
    tasks: list
    Q_R: int
    name: str = ''

    def arrays(self):
        K = len(self.tasks)
        link_task, link_p, link_w0, link_rho, starts = [], [], [], [], []
        for k, tc in enumerate(self.tasks):
            starts.append(len(link_task))
            for _ in range(tc.N):
                link_task.append(k)
                link_p.append(tc.p)
                link_w0.append(tc.w0)
                link_rho.append(tc.rho)
        return dict(
            link_task=np.array(link_task, dtype=np.int64),
            link_p=np.array(link_p), link_w0=np.array(link_w0), link_rho=np.array(link_rho),
            task_start=np.array(starts, dtype=np.int64),
            task_N=np.array([t.N for t in self.tasks], dtype=np.int64),
            task_q=np.array([t.q for t in self.tasks]),
            task_v=np.array([t.v for t in self.tasks]),
            task_beta=np.array([t.beta for t in self.tasks]),
        )

    def ET_table(self):
        K = len(self.tasks)
        Nmax = max(t.N for t in self.tasks)
        tab = np.ones((K, Nmax + 1))
        for k, tc in enumerate(self.tasks):
            for M in range(1, tc.N + 1):
                tab[k, M] = prop4_ET(M, tc.p)
        return tab


@dataclass
class PolicyCfg:
    name: str
    prio: int
    cons: int
    cutoff: tuple            # per task (large = none)
    m_min: tuple = None      # per task
    bias: tuple = None
    fresh_beta: float = 0.1
    IEbar: tuple = None      # per task expected info per delivery (PRIO_MARGINAL)
    whittle: tuple = None    # (Jgrid[K,G], Wtab[K,G])


BIG = 10 ** 9


def run(scn: Scenario, pol: PolicyCfg, T: int = 200_000, burn: int = 20_000,
        seed: int = 0, sample_every: int = 10, J0=None):
    a = scn.arrays()
    K = len(scn.tasks)
    from .physics import J_star
    if J0 is None:
        J0 = np.array([J_star(t.N * t.v ** 2, t.q) if t.q > 0 else t.N * t.v ** 2 for t in scn.tasks])
    cutoff = np.array([c if c is not None else BIG for c in pol.cutoff], dtype=np.int64)
    m_min = np.array(pol.m_min if pol.m_min is not None else [2] * K, dtype=np.int64)
    bias = np.array(pol.bias if pol.bias is not None else [1.0] * K, dtype=float)
    IEbar = np.array(pol.IEbar if pol.IEbar is not None else [0.0] * K, dtype=float)
    if pol.whittle is not None:
        Jg, Wt = pol.whittle
    else:
        Jg = np.tile(np.linspace(0, 1, 2), (K, 1))
        Wt = np.zeros((K, 2))
    out = run_kernel(seed, T, burn, scn.Q_R,
                     a['link_task'], a['link_p'], a['link_w0'], a['link_rho'],
                     a['task_start'], a['task_N'], a['task_q'], a['task_v'], a['task_beta'],
                     np.asarray(J0, dtype=float),
                     pol.prio, pol.cons, cutoff, m_min, bias, float(pol.fresh_beta),
                     scn.ET_table(), IEbar, np.ascontiguousarray(Jg, dtype=float),
                     np.ascontiguousarray(Wt, dtype=float), sample_every)
    keys = ['wmse', 'mse', 'deliv', 'useful', 'meanV', 'meanV2', 'meanI', 'meanm', 'disc', 'gen',
            'wasted_consumed', 'samples']
    res = dict(zip(keys, out))
    s = res.pop('samples')
    res['mse_p95'] = np.percentile(s, 95, axis=0)
    res['throughput'] = float(np.sum(res['deliv']))
    res['useful_throughput'] = float(np.sum(res['useful']))
    return res
