"""Sensing tasks (Kalman information states) and communication queues."""
from __future__ import annotations

from dataclasses import dataclass, field

from .physics import (info_classical, info_hybrid, kalman_predict, J_star,
                      advantage_threshold)


@dataclass
class SensingTask:
    name: str
    sensors: list[str]
    root: str
    q: float = 1e-4          # random-walk variance per slot (Assumption 4)
    v: float = 1.0           # single-qubit sensing visibility (Assumption 5)
    beta: float = 1.0        # importance weight
    J: float = 0.0           # information state

    @property
    def N(self) -> int:
        return len(self.sensors)

    @property
    def I_C(self) -> float:
        return info_classical(self.N, self.v)

    def V_star(self) -> float:
        return advantage_threshold(self.N, self.v)

    def J_pred(self) -> float:
        return kalman_predict(self.J, self.q)

    def info(self, m: int, V: float) -> float:
        """Per-slot information: m=0 classical; else hybrid/entangled."""
        return info_hybrid(self.N, m, V, self.v)

    def update(self, I: float) -> float:
        self.J = kalman_predict(self.J, self.q) + I
        return self.J

    def reset_steady_classical(self):
        self.J = J_star(self.I_C, self.q) if self.q > 0 else self.I_C


@dataclass
class CommClass:
    name: str
    s: str
    d: str
    lam: float
    w_min: float
    Q: int = 0
