"""Single-task profiles and Whittle indices (Approach 2, Section V-D).

* single_task_profile(): run one task with dedicated repeater memory under a given
  consumption rule / cutoff and measure the delivery rate s and the mean
  information per delivery (used by A1's marginal score and A2's abstraction).
* best_single_task_cutoff(): E1-style optimisation of the cutoff a_max.
* whittle_table(): Whittle index W(J) of the entanglement-opportunity abstraction
  (Definition 1) by relative value iteration on a J grid, plus an indexability check.
"""
from __future__ import annotations

import math

import numpy as np

from . import fastsim as F
from .physics import J_star, info_classical


def single_task_profile(tc: 'F.TaskCfg', cons: int, cutoff, m_min: int = 2,
                        T: int = 400_000, seed: int = 12345, Q: int | None = None) -> dict:
    scn = F.Scenario([tc], Q_R=tc.N if Q is None else min(Q, tc.N))
    pol = F.PolicyCfg('single', F.PRIO_STATIC, cons, cutoff=(cutoff,), m_min=(m_min,))
    r = F.run(scn, pol, T=T, burn=2000, seed=seed)
    s = float(r['deliv'][0])
    IC = info_classical(tc.N, tc.v)
    # mean info per delivery slot: meanI = IC + s*(IEbar - IC)
    IEbar = IC + (r['meanI'][0] - IC) / s if s > 0 else IC
    return dict(s=s, IEbar=float(IEbar), meanI=float(r['meanI'][0]), mse=float(r['mse'][0]),
                useful_frac=float(r['useful'][0] / max(s, 1e-12)))


def best_single_task_cutoff(tc: 'F.TaskCfg', cons: int = F.CONS_WAIT_SUBSET,
                            grid=None, T: int = 200_000, Q: int | None = None) -> tuple[int, dict]:
    if grid is None:
        amax = int(min(400, 6 / max(1 - tc.rho, 1e-4)))
        grid = sorted(set(np.unique(np.round(np.geomspace(1, amax, 28)).astype(int))))
    best = None
    for a in grid:
        prof = single_task_profile(tc, cons, int(a), T=T, Q=Q)
        if best is None or prof['mse'] < best[1]['mse'] - 1e-12:
            best = (int(a), prof)
    return best


def whittle_table(tc: 'F.TaskCfg', s: float, IEbar: float, n_grid: int = 160,
                  n_lambda: int = 120, iters: int = 4000, tol: float = 1e-10):
    """Whittle index for: active -> I = IEbar w.p. s else I_C; passive -> I_C.
    Cost per slot beta/J (after the update) + lambda * active. Average cost.
    Returns (Jgrid, W, info) where info has the indexability check."""
    IC = info_classical(tc.N, tc.v)
    q, beta = tc.q, tc.beta
    J_lo = 0.7 * J_star(IC, q)
    J_hi = 1.5 * J_star(IC + s * max(IEbar - IC, 0) + 1e-9, q) + IEbar
    Jg = np.geomspace(J_lo, J_hi, n_grid)
    phi = Jg / (1 + q * Jg)
    Jp_C = np.clip(phi + IC, J_lo, J_hi)
    Jp_E = np.clip(phi + IEbar, J_lo, J_hi)
    cost_C = beta / (phi + IC)
    cost_E = beta / (phi + IEbar)

    def interp(Vv, x):
        return np.interp(x, Jg, Vv)

    gain_scale = beta * (1 / (J_star(IC, q)) - 1 / (J_star(IC, q) + IEbar))
    lams = np.concatenate([[0.0], np.geomspace(1e-4 * abs(gain_scale) + 1e-15,
                                               50 * abs(gain_scale) + 1e-12, n_lambda - 1)])
    act = np.zeros((len(lams), n_grid), dtype=bool)
    for li, lam in enumerate(lams):
        h = np.zeros(n_grid)
        for it in range(iters):
            QC = cost_C + interp(h, Jp_C)
            QA = lam + s * (cost_E + interp(h, Jp_E)) + (1 - s) * QC
            hn = np.minimum(QC, QA)
            hn -= hn[0]
            if np.max(np.abs(hn - h)) < tol:
                h = hn
                break
            h = hn
        act[li] = QA < QC - 1e-14
    # W(J) = largest lambda for which active is optimal
    W = np.zeros(n_grid)
    viol = 0
    for j in range(n_grid):
        col = act[:, j]
        idx = np.where(col)[0]
        W[j] = lams[idx.max()] if len(idx) else 0.0
        # indexable: active set in lambda must be {lambda < W}: no passive-then-active
        if len(idx):
            first_passive = np.argmin(col) if not col.all() else len(col)
            if col[first_passive:].any():
                viol += 1
    # monotone smoothing for interpolation robustness (W should decrease in J)
    return Jg, W, dict(indexability_violations=int(viol), lambdas=lams.tolist())
