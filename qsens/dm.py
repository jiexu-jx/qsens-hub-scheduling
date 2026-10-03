"""Small density-matrix toolkit for checking Lemmas 1-4 (Exercise 2) and for the
QFI-vs-parity gap of Remark 3.  Dense numpy; fine for N <= 6 qubits."""
from __future__ import annotations

import itertools
from functools import reduce

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = (I2, X, Y, Z)


def kron(*ops):
    return reduce(np.kron, ops)


def ket(bits: str) -> np.ndarray:
    v = np.zeros(2 ** len(bits), dtype=complex)
    v[int(bits, 2)] = 1.0
    return v


PHI_PLUS = (ket('00') + ket('11')) / np.sqrt(2)
BELL = [np.kron(I2, P) @ PHI_PLUS for P in PAULIS]  # Phi_P = (I ⊗ P)|Phi+>


def werner(w: float) -> np.ndarray:
    P = np.outer(PHI_PLUS, PHI_PLUS.conj())
    return w * P + (1 - w) * np.eye(4) / 4


def ghz(N: int) -> np.ndarray:
    v = (ket('0' * N) + ket('1' * N)) / np.sqrt(2)
    return np.outer(v, v.conj())


def apply_1q_channel(rho: np.ndarray, kraus: list[np.ndarray], q: int, n: int) -> np.ndarray:
    out = np.zeros_like(rho)
    for K in kraus:
        op = kron(*[K if i == q else I2 for i in range(n)])
        out += op @ rho @ op.conj().T
    return out


def depolarizing_kraus(W: float) -> list[np.ndarray]:
    """D_W(s) = W s + (1-W) I/2  ==  Pauli channel with p_I=(1+3W)/4, p_XYZ=(1-W)/4."""
    pI = (1 + 3 * W) / 4
    pP = (1 - W) / 4
    return [np.sqrt(pI) * I2, np.sqrt(pP) * X, np.sqrt(pP) * Y, np.sqrt(pP) * Z]


def dephasing_kraus(v: float) -> list[np.ndarray]:
    """Coherence |0><1| scaled by v."""
    pZ = (1 - v) / 2
    return [np.sqrt(1 - pZ) * I2, np.sqrt(pZ) * Z]


def partial_trace(rho: np.ndarray, keep: list[int], n: int) -> np.ndarray:
    rho = rho.reshape([2] * (2 * n))
    trace_out = [i for i in range(n) if i not in keep]
    # move axes: trace pairs (i, i+n)
    for k, i in enumerate(sorted(trace_out, reverse=True)):
        m = rho.ndim // 2
        rho = np.trace(rho, axis1=i, axis2=i + m)
    d = 2 ** len(keep)
    return rho.reshape(d, d)


def swap_werner_numeric(w1: float, w2: float) -> float:
    """Entanglement swapping of rho_W(w1) on (A,B) and rho_W(w2) on (C,D), Bell
    measurement on (B,C) with Pauli correction on D; returns fidelity of (A,D)
    with Phi+ averaged over outcomes (Lemma 1)."""
    rho = np.kron(werner(w1), werner(w2))  # qubits A,B,C,D
    n = 4
    out = np.zeros((4, 4), dtype=complex)
    for k, P in enumerate(PAULIS):
        # projector onto Bell state Phi_P on (B,C)
        b = BELL[k]
        proj = np.outer(b, b.conj())
        Pi = kron(I2, proj, I2)
        post = Pi @ rho @ Pi
        # correction on D: outcome Phi_P leaves (A,D) in Phi_P-type state; apply P
        corr = kron(I2, I2, I2, P)
        post = corr @ post @ corr.conj().T
        out += partial_trace(post, [0, 3], n)
    F = np.real(PHI_PLUS.conj() @ out @ PHI_PLUS)
    return float(F)


def teleport_channel_numeric(w: float, sigma: np.ndarray) -> np.ndarray:
    """Teleport a qubit state sigma through rho_W(w) (Lemma 2). Qubits: S, A, B;
    Bell measurement on (S,A), Pauli correction on B."""
    rho = np.kron(sigma, werner(w))
    n = 3
    out = np.zeros((2, 2), dtype=complex)
    for k, P in enumerate(PAULIS):
        b = BELL[k]
        Pi = kron(np.outer(b, b.conj()), I2)
        post = Pi @ rho @ Pi
        corr = kron(I2, I2, P)
        post = corr @ post @ corr.conj().T
        out += partial_trace(post, [2], n)
    return out


def teleported_ghz(Ws) -> np.ndarray:
    """(⊗_i D_{W_i})(GHZ_N) -- the delivered state of Lemma 3."""
    N = len(Ws)
    rho = ghz(N)
    for i, W in enumerate(Ws):
        rho = apply_1q_channel(rho, depolarizing_kraus(W), i, N)
    return rho


def encode_phases(rho: np.ndarray, thetas, v: float = 1.0) -> np.ndarray:
    """Each qubit: |1> acquires e^{i theta}; dephasing with visibility v."""
    N = len(thetas)
    for i, th in enumerate(thetas):
        U = np.diag([1, np.exp(1j * th)])
        op = kron(*[U if j == i else I2 for j in range(N)])
        rho = op @ rho @ op.conj().T
        if v < 1:
            rho = apply_1q_channel(rho, dephasing_kraus(v), i, N)
    return rho


def parity_expectation(rho: np.ndarray) -> float:
    N = int(np.log2(rho.shape[0]))
    return float(np.real(np.trace(kron(*[X] * N) @ rho)))


def qfim_phases(rho0: np.ndarray) -> np.ndarray:
    """QFIM for theta under U = exp(-i sum_i theta_i Z_i/2) at theta=0:
    F_ij = 2 sum_{k,l} (l_k - l_l)^2/(l_k + l_l) Re(<k|H_i|l><l|H_j|k>)."""
    N = int(np.log2(rho0.shape[0]))
    lam, U = np.linalg.eigh(rho0)
    lam = np.clip(lam, 0, None)
    Hs = [U.conj().T @ (kron(*[Z / 2 if j == i else I2 for j in range(N)])) @ U for i in range(N)]
    L = lam[:, None] + lam[None, :]
    D = (lam[:, None] - lam[None, :]) ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        Wt = np.where(L > 1e-15, D / L, 0.0)
    F = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            F[i, j] = F[j, i] = 2 * np.sum(Wt * np.real(Hs[i] * Hs[j].T))
    return F


def qfi_average_phase(rho0: np.ndarray) -> dict:
    """Information about psi = mean(theta) from the delivered state rho0.

    Returns:
      'multi'  : QFI with the other N-1 phase combinations as unknown nuisance
                 parameters, 1/(a^T F^+ a) with a = 1/N (inf-safe);
      'equal'  : single-parameter QFI if all local phases are equal (1^T F 1);
      'parity' : parity-readout Fisher information N^2 <X^N>^2 (Lemma 4).
    """
    N = int(np.log2(rho0.shape[0]))
    F = qfim_phases(rho0)
    a = np.ones(N) / N
    Fp = np.linalg.pinv(F, rcond=1e-10)
    # check a in range(F)
    resid = np.linalg.norm(F @ Fp @ a - a)
    multi = 1.0 / float(a @ Fp @ a) if resid < 1e-8 else 0.0
    equal = float(np.ones(N) @ F @ np.ones(N))
    V = parity_expectation(rho0)
    return {'multi': multi, 'equal': equal, 'parity': N * N * V * V, 'V': V}
