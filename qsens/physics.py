"""Physics layer of the slot-level model (Section III-IV of the tech report).

Werner algebra, memory decay, GHZ visibility (Lemma 3), per-round Fisher
information (eqs. (5), (6) and the hybrid of Prop. 2), Kalman information
recursion (eq. (7)), and the closed forms of Section IV used for validation.
Equation numbers refer to the PDF of the technical report (draft v0.1).
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Werner algebra
# ---------------------------------------------------------------------------

def fidelity_from_w(w: float) -> float:
    """Fidelity <Phi+|rho_W(w)|Phi+> = (1+3w)/4."""
    return (1.0 + 3.0 * w) / 4.0


def w_from_fidelity(F: float) -> float:
    return (4.0 * F - 1.0) / 3.0


def eta_from_tau(tau: float) -> float:
    """Per-slot depolarizing factor eta = exp(-1/tau); tau=inf -> 1."""
    return 1.0 if math.isinf(tau) else math.exp(-1.0 / tau)


def decayed_w(w0: float, eta_u: float, eta_v: float, age: int) -> float:
    """Eq. (3): w(t) = w0 (eta_u eta_v)^(t-c)."""
    return w0 * (eta_u * eta_v) ** age


def swap_w(*ws: float) -> float:
    """Lemma 1: swapping Werner pairs multiplies Werner parameters."""
    return float(np.prod(ws))


def ghz_visibility(ws: Iterable[float]) -> float:
    """Lemma 3: V = product of Werner parameters of all consumed pairs."""
    return float(np.prod(list(ws)))


# ---------------------------------------------------------------------------
# Fisher information per round (about psi = average phase)
# ---------------------------------------------------------------------------

def info_classical(N: int, v: float = 1.0) -> float:
    """Eq. (5): I_C = N v^2."""
    return N * v * v


def info_entangled(N: int, V: float, v: float = 1.0) -> float:
    """Eq. (6): I_E = N^2 V^2 v^(2N) (parity readout, Lemma 4)."""
    return N * N * V * V * v ** (2 * N)


def info_hybrid(N: int, m: int, V: float, v: float = 1.0) -> float:
    """Prop. 2 generalised to v<1 and arbitrary V: m sensors share a GHZ state of
    visibility V (parity readout), the other N-m sense classically.
    Var(psi) = [1/(V^2 v^(2m)) + (N-m)/v^2] / N^2.  m=0 -> classical, m=N -> (6).
    """
    if m <= 0:
        return info_classical(N, v)
    if V <= 0.0:
        # GHZ round gives no information about the subset sum -> psi unidentifiable
        return 0.0
    var_sum = 1.0 / (V * V * v ** (2 * m)) + (N - m) / (v * v)
    return N * N / var_sum


def advantage_threshold(N: int, v: float = 1.0) -> float:
    """Eq. (12): V*_N = N^{-1/2} v^{-(N-1)}."""
    return N ** -0.5 * v ** (-(N - 1))


# ---------------------------------------------------------------------------
# Kalman information recursion (eq. (7)) and Prop. 3
# ---------------------------------------------------------------------------

def kalman_predict(J: float, q: float) -> float:
    """phi(J) = J / (1 + qJ)."""
    return J / (1.0 + q * J)


def kalman_step(J: float, q: float, I: float) -> float:
    return kalman_predict(J, q) + I


def J_star(I: float, q: float) -> float:
    """Eq. (15): fixed point of J -> phi(J) + I.  q=0 -> inf."""
    if q == 0:
        return math.inf
    return I / 2.0 + math.sqrt(I * I / 4.0 + I / q)


# ---------------------------------------------------------------------------
# Corollary 1 and Proposition 2 (uniform star, v = 1)
# ---------------------------------------------------------------------------

def cor1_Nopt(w: float) -> float:
    return 1.0 / math.log(1.0 / w)


def cor1_max_helpful_N(w: float, Nmax: int = 100000) -> int:
    """Largest N with N w^{2N} > 1 (0 if none)."""
    best = 0
    for N in range(2, Nmax):
        if N * w ** (2 * N) > 1.0:
            best = N
        elif N > cor1_Nopt(w) + 2:
            break
    return best


def prop2_var(N: int, m: int, w: float) -> float:
    """Eq. (13)."""
    return (w ** (-2 * m) + N - m) / N ** 2


def prop2_mopt_continuous(w: float) -> float:
    lam = math.log(1.0 / w)
    if 2 * lam >= 1:
        return float('nan')
    return math.log(1.0 / (2 * lam)) / (2 * lam)


def prop2_mopt_integer(N: int, w: float) -> int:
    """Integer minimiser of eq. (13) over m in {2..N} (brute force)."""
    ms = range(2, N + 1)
    return min(ms, key=lambda m: prop2_var(N, m, w))


# ---------------------------------------------------------------------------
# Proposition 4 (collect-and-consume in a star)
# ---------------------------------------------------------------------------

def prop4_ET(N: int, p: float, tmax: int | None = None) -> float:
    """Eq. (18): E[T] = sum_{t>=0} [1 - (1 - (1-p)^t)^N]."""
    if tmax is None:
        tmax = int(60 / p) + 200
    t = np.arange(tmax + 1)
    return float(np.sum(1.0 - (1.0 - (1.0 - p) ** t) ** N))


def prop4_EV2(N: int, p: float, rho: float, w0: float, tmax: int | None = None) -> float:
    """Eq. (19) with A_t computed by the finite geometric sum."""
    if tmax is None:
        tmax = int(60 / p) + 200
    t = np.arange(1, tmax + 1, dtype=float)
    r2 = rho * rho
    if abs(r2 - (1 - p)) > 1e-14:
        A = p * (r2 ** t - (1 - p) ** t) / (r2 - (1 - p))
    else:
        A = p * t * (1 - p) ** (t - 1)
    B = A - p * (1 - p) ** (t - 1)
    B = np.maximum(B, 0.0)
    return float(w0 ** (2 * N) * np.sum(A ** N - B ** N))


def prop4_gain(N: int, p: float, rho: float, w0: float, v: float = 1.0,
               regime: str = 'concurrent') -> float:
    """Eq. (17) (concurrent) or the exclusive-regime average information per slot."""
    ET = prop4_ET(N, p)
    EV2 = prop4_EV2(N, p, rho, w0)
    if regime == 'concurrent':
        return (N * N * v ** (2 * N) * EV2 - N * v * v) / ET
    return N * N * v ** (2 * N) * EV2 / ET
