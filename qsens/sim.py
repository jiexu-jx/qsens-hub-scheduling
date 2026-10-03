"""Reference slot-level simulator (general graphs; readable, not fast).

Slot order (fixed throughout the project, consistent with Prop. 4):
  1. generation attempts on the links the policy selects (heralded; new pairs age 0)
  2. policy decisions on the current inventory: discards, GHZ deliveries (and
     communication deliveries)
  3. sensing: each task runs entangled/hybrid mode if it received a delivery,
     classical mode otherwise; Kalman update (7); queue update (8)
  4. all stored pairs age by one slot (eq. (3))
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .network import Network
from .tasks import SensingTask, CommClass


@dataclass
class Delivery:
    task: int          # task index
    m: int             # number of sensors in the GHZ state (hybrid if m < N)
    pids: list         # consumed pair ids
    V: float = 0.0     # filled in by the simulator


class Policy:
    def attempts(self, sim: 'Simulator') -> list:
        raise NotImplementedError

    def decide(self, sim: 'Simulator') -> tuple[list, list[Delivery]]:
        raise NotImplementedError


class Simulator:
    def __init__(self, net: Network, tasks: list[SensingTask], policy: Policy,
                 comm: list[CommClass] | None = None):
        self.net = net
        self.tasks = tasks
        self.policy = policy
        self.comm = comm or []
        self.t = 0
        self.log = {'J': [], 'deliv': [], 'V': [], 'I': []}

    def step(self):
        net = self.net
        keys = self.policy.attempts(self)
        net.attempt(keys)
        discards, deliveries = self.policy.decide(self)
        net.discard(discards)
        got = {}
        for d in deliveries:
            d.V = net.consume(d.pids)
            got[d.task] = d
        Js, Is = [], []
        for k, task in enumerate(self.tasks):
            if k in got:
                I = task.info(got[k].m, got[k].V)
            else:
                I = task.I_C
            task.update(I)
            Js.append(task.J)
            Is.append(I)
        self.log['J'].append(Js)
        self.log['I'].append(Is)
        self.log['deliv'].append([(d.task, d.m, d.V) for d in deliveries])
        net.end_slot()
        self.t += 1

    def run(self, T: int):
        for _ in range(T):
            self.step()
        return self

    # ------------------------------------------------------------------ metrics
    def weighted_mse(self, burn: int = 0) -> float:
        J = np.array(self.log['J'][burn:])
        beta = np.array([t.beta for t in self.tasks])
        return float(np.mean((beta[None, :] / J).sum(axis=1)))
