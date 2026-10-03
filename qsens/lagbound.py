"""Tighter lower bound via Lagrangian relaxation of the hub-memory constraint
(ICC paper, Proposition 3).

Relax "attempts + stored pairs <= Q in every slot" to a time-average constraint
sum_k M_k <= Q, where M_k is task k's average hub qubit-slot usage. Each task then
faces a single-cluster MDP on its own pair inventory (no J state is needed thanks to
the Jensen step, Proposition 1): maximise the average information gain G_k minus a
memory price mu * M_k. Sweeping mu traces the concave frontier G_k = F_k(M_k), and
    wMSE >= min_{M: sum M_k <= Q} sum_k beta_k / J*_k(I_C,k + F_k(M_k)).
Relaxations that keep the bound valid (they can only help the single-cluster problem):
  * ages are capped at A (a pair older than A is treated as age A: optimistic);
  * a single cluster may hold up to min(N_k, Q) pairs (its share of the hub);
  * every consumption subset is allowed; among pairs that are not consumed, keeping
    the youngest ones is w.l.o.g. (a younger inventory dominates an older one).
"""
from __future__ import annotations

import math
from itertools import combinations_with_replacement

import numpy as np
from numba import njit

from .physics import info_classical, J_star


def _states(Ncap, A):
    """All multisets of ages in 1..A of size 0..Ncap, as descending tuples."""
    st = [()]
    for n in range(1, Ncap + 1):
        for c in combinations_with_replacement(range(1, A + 1), n):
            st.append(tuple(sorted(c, reverse=True)))
    return st


def build(N, Q, A):
    Ncap = min(N, Q)
    B = A + 1
    st = _states(Ncap, A)
    S = len(st)
    ages = np.zeros((S, max(Ncap, 1)), dtype=np.int64)
    nn = np.zeros(S, dtype=np.int64)
    lut = -np.ones(B ** max(Ncap, 1), dtype=np.int64)
    for i, t in enumerate(st):
        nn[i] = len(t)
        code = 0
        for j, a in enumerate(t):
            ages[i, j] = a
            code += a * B ** j
        lut[code] = i
    return ages, nn, lut, Ncap


@njit(cache=True)
def _ih(N, m, V, v):
    d = V * V * v ** (2 * m)
    if d <= 1e-300:
        return 0.0
    return N * N / (1.0 / d + (N - m) / (v * v))


@njit(cache=True)
def _encode_desc(buf, k, B):
    # buf[0:k] arbitrary order -> sort descending -> code
    for i in range(1, k):
        x = buf[i]
        j = i - 1
        while j >= 0 and buf[j] < x:
            buf[j + 1] = buf[j]
            j -= 1
        buf[j + 1] = x
    code = 0
    p = 1
    for i in range(k):
        code += buf[i] * p
        p *= B
    return code


@njit(cache=True)
def _sweep(h, ages, nn, lut, N, Ncap, A, p, w0, rho, v, mu, binom, polx, evalonly):
    """One Bellman sweep. Returns new values and greedy attempt counts.
    If evalonly, use the attempt counts in polx (choices are always greedy)."""
    S = nn.shape[0]
    B = A + 1
    IC = N * v * v
    hn = np.empty(S)
    besta = np.zeros(S, dtype=np.int64)
    post = np.empty(N + 1, dtype=np.int64)
    rest = np.empty(N + 1, dtype=np.int64)
    buf = np.empty(N + 1, dtype=np.int64)
    for s in range(S):
        n = nn[s]
        xmax = Ncap - n
        bestv = -1e300
        bx = 0
        x0, x1 = 0, xmax
        if evalonly:
            x0 = polx[s]; x1 = polx[s]
        for x in range(x0, x1 + 1):
            ev = 0.0
            for succ in range(x + 1):
                pr = binom[x, succ]
                if pr == 0.0:
                    continue
                # post-herald ages ascending: succ zeros then stored ages ascending
                npost = succ + n
                for i in range(succ):
                    post[i] = 0
                for i in range(n):
                    post[succ + i] = ages[s, n - 1 - i]
                best = -1e300
                for mask in range(1 << npost):
                    # canonical among equal ages: chosen ones must be a prefix of each run
                    ok = True
                    for i in range(1, npost):
                        if post[i] == post[i - 1] and ((mask >> i) & 1) == 1 and ((mask >> (i - 1)) & 1) == 0:
                            ok = False
                            break
                    if not ok:
                        continue
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
                    # keep the youngest k of rest (rest is ascending)
                    for k in range(nr + 1):
                        for i in range(k):
                            a = rest[i] + 1
                            buf[i] = a if a < A else A
                        code = _encode_desc(buf, k, B)
                        val = rew + h[lut[code]]
                        if val > best:
                            best = val
                ev += pr * best
            val = ev - mu * (n + x)
            if val > bestv:
                bestv = val
                bx = x
        hn[s] = bestv
        besta[s] = bx
    return hn, besta


@njit(cache=True)
def _stationary(ages, nn, lut, N, Ncap, A, p, w0, rho, v, binom, polx, h, iters):
    """Stationary distribution of the greedy policy (attempts polx, choices greedy
    w.r.t. h); returns (avg gain, avg memory)."""
    S = nn.shape[0]
    B = A + 1
    IC = N * v * v
    pi = np.zeros(S)
    pi[0] = 1.0
    post = np.empty(N + 1, dtype=np.int64)
    rest = np.empty(N + 1, dtype=np.int64)
    buf = np.empty(N + 1, dtype=np.int64)
    # precompute transitions: for each s, successor list with probs and rewards
    G = 0.0
    M = 0.0
    nxt_idx = np.zeros((S, N + 1), dtype=np.int64)
    nxt_rew = np.zeros((S, N + 1))
    for s in range(S):
        n = nn[s]
        x = polx[s]
        for succ in range(x + 1):
            npost = succ + n
            for i in range(succ):
                post[i] = 0
            for i in range(n):
                post[succ + i] = ages[s, n - 1 - i]
            best = -1e300
            bidx = 0
            brew = 0.0
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
                    idx = lut[code]
                    val = rew + h[idx]
                    if val > best + 1e-15:
                        best = val
                        bidx = idx
                        brew = rew
            nxt_idx[s, succ] = bidx
            nxt_rew[s, succ] = brew
    for it in range(iters):
        new = np.zeros(S)
        for s in range(S):
            if pi[s] == 0.0:
                continue
            x = polx[s]
            for succ in range(x + 1):
                new[nxt_idx[s, succ]] += pi[s] * binom[x, succ]
        pi = new
    for s in range(S):
        if pi[s] == 0.0:
            continue
        x = polx[s]
        M += pi[s] * (nn[s] + x)
        for succ in range(x + 1):
            G += pi[s] * binom[x, succ] * nxt_rew[s, succ]
    return G, M


def frontier(N, p, w0, rho, Q, v=1.0, A=10, mus=None, tol=1e-9, max_iter=5000):
    """Points (M, G) of the single-cluster frontier for a grid of memory prices."""
    ages, nn, lut, Ncap = build(N, Q, A)
    binom = np.zeros((N + 1, N + 1))
    for x in range(N + 1):
        for s_ in range(x + 1):
            binom[x, s_] = math.comb(x, s_) * p ** s_ * (1 - p) ** (x - s_)
    IC = info_classical(N, v)
    gmax = N * N * w0 ** (2 * N) * v ** (2 * N)  # rough scale of gains
    if mus is None:
        mus = np.concatenate([[0.0], np.geomspace(1e-4, 2.0, 22) * max(gmax, IC)])
    h = np.zeros(len(nn))
    dummy = np.zeros(len(nn), dtype=np.int64)
    pts = []

    def solve(mu, h):
        for it in range(max_iter):
            hn, bx = _sweep(h, ages, nn, lut, N, Ncap, A, p, w0, rho, v, mu, binom, dummy, False)
            hn = 0.5 * hn + 0.5 * h
            span = np.max(hn - h) - np.min(hn - h)
            h = hn - hn[0]
            if span < tol:
                break
        hn, bx = _sweep(h, ages, nn, lut, N, Ncap, A, p, w0, rho, v, mu, binom, dummy, False)
        d = hn - h
        G, M = _stationary(ages, nn, lut, N, Ncap, A, p, w0, rho, v, binom, bx, hn - hn[0], 3000)
        return (float(mu), float(M), float(G), float(np.max(d)), float(np.min(d))), h

    for mu in mus:
        pt, h = solve(mu, h)
        pts.append(pt)
    # refine: g*(mu) is convex piecewise linear in mu; add the breakpoints between
    # consecutive solutions so that the dual frontier min_mu[g*(mu)+mu M] becomes exact
    for _ in range(40):
        pts.sort()
        new_mu = None
        for a, b in zip(pts[:-1], pts[1:]):
            if abs(a[1] - b[1]) > 1e-9:
                m_ = (a[2] - b[2]) / (a[1] - b[1])
                if a[0] + 1e-12 < m_ < b[0] - 1e-12 and all(abs(m_ - q[0]) > 1e-10 * (1 + m_) for q in pts):
                    # only useful if g at m_ could be above the two lines' intersection
                    new_mu = m_
                    break
        if new_mu is None:
            break
        pt, h = solve(new_mu, h)
        pts.append(pt)
    pts.sort()
    return pts


def concave_frontier(pts, Mcap):
    """Outer (dual) approximation of the single-cluster frontier:
    F(M) <= min_mu [ g*(mu) + mu M ] with g*(mu) replaced by its RVI upper bound.
    Returned as breakpoints (xs, ys) of the concave piecewise-linear function on
    [0, Mcap] (valid upper bound on the achievable information gain)."""
    lines = [(g_up, mu) for (mu, M, G, g_up, g_lo) in pts]
    xs = np.linspace(0.0, float(Mcap), 801)
    ys = np.min([g + mu * xs for g, mu in lines], axis=0)
    ys = np.maximum.accumulate(np.maximum(ys, 0.0))  # nondecreasing (free disposal)
    return xs, ys


def separable_dual_min(fs, X, Q, return_lam=False):
    """Lower bound on  min sum_k f_k(x_k)  s.t. sum x_k <= Q, 0 <= x_k <= X_k  for convex f_k,
    by weak duality: D(lam) = sum_k min_x [f_k(x) + lam x] - lam Q  <=  primal min for every
    lam >= 0. Inner minima by golden-section search (convex, 1-D); D maximised over lam.
    Returns (bound, x at the best lam)."""
    from scipy.optimize import minimize_scalar
    K = len(fs)

    def inner(k, lam):
        r = minimize_scalar(lambda x: fs[k](x) + lam * x, bounds=(0.0, X[k]), method='bounded',
                            options={'xatol': 1e-12, 'maxiter': 2000})
        cands = [(fs[k](0.0), 0.0), (fs[k](X[k]) + lam * X[k], X[k]), (r.fun, r.x)]
        v, x = min(cands)
        return v, x

    def D(lam):
        tot, xs = 0.0, []
        for k in range(K):
            v, x = inner(k, lam)
            tot += v; xs.append(x)
        return tot - lam * Q, xs

    # if the budget is slack at lam = 0, the unconstrained minimum is the answer
    v0, x0 = D(0.0)
    if sum(x0) <= Q + 1e-12:
        return (v0, x0, 0.0) if return_lam else (v0, x0)
    lo, hi = 0.0, 1.0
    while D(hi)[1] and sum(D(hi)[1]) > Q:
        hi *= 4.0
        if hi > 1e12:
            break
    r = minimize_scalar(lambda lam: -D(lam)[0], bounds=(lo, hi), method='bounded',
                        options={'xatol': 1e-14 * max(hi, 1.0), 'maxiter': 3000})
    best = max([(D(l)[0], l) for l in (lo, hi, r.x)])
    val, xs = D(best[1])
    return (val, xs, float(best[1])) if return_lam else (val, xs)


def frontier_lines(t, Q, A=10, cache=None):
    key = f"{t.N}|{t.p:.9g}|{t.w0:.9g}|{t.rho:.9g}|{min(t.N, Q)}|{t.v:.9g}|{A}"
    if cache is not None and key in cache:
        pts = cache[key]
    else:
        pts = frontier(t.N, t.p, t.w0, t.rho, Q, t.v, A)
        if cache is not None:
            cache[key] = pts
    return [(g_up, mu) for (mu, M, G, g_up, g_lo) in pts]


def lag_lower_bound(tasks, Q, A=10, cache=None):
    """Lower bound on weighted MSE (Proposition 3), evaluated with the exact dual
    frontier F_k(M) = min_mu [g_up(mu) + mu M] and a weak-duality solver (valid even if
    the outer minimisation is solved inexactly)."""
    tasks = list(tasks)
    K = len(tasks)
    IC = np.array([info_classical(t.N, t.v) for t in tasks])
    fs, X = [], []
    for k, t in enumerate(tasks):
        lines = np.array(frontier_lines(t, Q, A, cache))
        g, mu = lines[:, 0], lines[:, 1]
        def f(M, g=g, mu=mu, t=t, ic=IC[k]):
            F = max(0.0, float(np.min(g + mu * M)))
            I = ic + F
            return t.beta / (I / 2 + math.sqrt(I * I / 4 + I / t.q))
        fs.append(f); X.append(float(min(t.N, Q)))
    lb, M = separable_dual_min(fs, X, Q)
    return dict(lb=float(lb), M=[float(m) for m in M])
