"""Information-rate lower bound on the weighted MSE of ANY scheduling policy at a
shared hub (ICC paper, Proposition 2).

Ingredients
  (i)  Jensen/fixed-point step: for every task and every policy,
          liminf (1/T) sum_t 1/J_k(t) >= 1 / J*_k(Ibar_k),
       where Ibar_k is the long-run average per-slot information and J* is eq. (15).
  (ii) Gain envelope: a delivery that consumes m pairs adds at most
          Delta_k(m) = I_H(N_k, m, w0_k^m) - I_C,k
       (fresh pairs, best visibility); at most one delivery per slot, so with r_k
       pairs consumed per slot   Ibar_k <= I_C,k + h_k(r_k),
          h_k(r) = max { sum_m f_m Delta_k(m)^+ : sum_m m f_m <= r, sum_m f_m <= 1, f >= 0 }.
  (iii) Hub memory: an attempt occupies a hub qubit for the slot, so the attempt rates
       satisfy sum_k a_k <= Q, a_k <= N_k, and r_k <= p_k a_k.
Combining:  wMSE >= min_a sum_k beta_k / J*_k(I_C,k + h_k(p_k a_k))  (a convex program).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linprog, minimize

from .physics import info_hybrid, info_classical, J_star


def gain_points(N: int, w0: float, v: float = 1.0) -> np.ndarray:
    """Delta(m)^+ for m = 0..N (Delta(0) = 0)."""
    IC = info_classical(N, v)
    d = np.zeros(N + 1)
    for m in range(1, N + 1):
        d[m] = max(0.0, info_hybrid(N, m, w0 ** m, v) - IC)
    return d


def envelope(N: int, w0: float, v: float = 1.0):
    """Return h(r) as a callable: concave, nondecreasing, piecewise linear on [0, N]
    (upper concave envelope of (0,0),(m,Delta(m)) made nondecreasing)."""
    d = gain_points(N, w0, v)
    pts = [(0.0, 0.0)] + [(float(m), d[m]) for m in range(1, N + 1)]
    # upper hull (monotone chain)
    hull = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    # make nondecreasing: cut at the maximum
    imax = int(np.argmax([y for _, y in hull]))
    hull = hull[:imax + 1]
    xs = np.array([x for x, _ in hull])
    ys = np.array([y for _, y in hull])

    def h(r):
        r = np.asarray(r, dtype=float)
        return np.interp(np.minimum(r, xs[-1]), xs, ys) if len(xs) > 1 else np.zeros_like(r)
    return h, xs, ys


def lower_bound(tasks, Q: int) -> dict:
    """Proposition 2 bound, solved by weak duality (valid regardless of solver accuracy)."""
    from .lagbound import separable_dual_min
    tasks = list(tasks)
    fs, X = [], []
    for t in tasks:
        h = envelope(t.N, t.w0, t.v)[0]
        IC = info_classical(t.N, t.v)
        def f(a, h=h, t=t, IC=IC):
            I = IC + float(h(t.p * a))
            return t.beta / (I / 2 + math.sqrt(I * I / 4 + I / t.q))
        fs.append(f); X.append(float(min(t.N, Q)))
    lb, a = separable_dual_min(fs, X, Q)
    B0 = float(sum(t.beta / J_star(info_classical(t.N, t.v), t.q) for t in tasks))
    return dict(lb=float(lb), a=[float(x) for x in a], B0=B0)
