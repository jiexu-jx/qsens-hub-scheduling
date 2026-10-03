"""Reference (pure-Python) policies for the general simulator in sim.py.

* CollectAndConsume -- the single-star policy of Prop. 4.
* RepeaterPolicy    -- the parametric repeater-sharing family used in the go/no-go
  experiment; mirrors fastsim.run_kernel line by line so the fast kernel can be
  cross-validated against the general engine (tests/test_crossval.py).
"""
from __future__ import annotations

import math

import numpy as np

from .sim import Policy, Delivery
from .physics import info_hybrid, kalman_predict, prop4_ET, advantage_threshold
from . import fastsim as F


class CollectAndConsume(Policy):
    """Every link without a stored pair attempts; consume all N as soon as present."""

    def __init__(self, task_index: int = 0):
        self.k = task_index

    def attempts(self, sim):
        task = sim.tasks[self.k]
        net = sim.net
        keys = []
        for s in task.sensors:
            key = frozenset((task.root, s))
            if not net.pairs_on(key):
                keys.append(key)
        return keys

    def decide(self, sim):
        task = sim.tasks[self.k]
        net = sim.net
        pids = []
        for s in task.sensors:
            ps = net.pairs_on(frozenset((task.root, s)))
            if not ps:
                return [], []
            pids.append(ps[0].pid)
        return [], [Delivery(self.k, task.N, pids)]


class RepeaterPolicy(Policy):
    """Python mirror of the fastsim parametric policy (subset of priority types)."""

    def __init__(self, cfg: 'F.PolicyCfg', scn: 'F.Scenario', root: str = 'R'):
        self.cfg = cfg
        self.scn = scn
        self.root = root
        self.ET = scn.ET_table()
        K = len(scn.tasks)
        self.cutoff = [c if c is not None else F.BIG for c in cfg.cutoff]
        self.m_min = list(cfg.m_min) if cfg.m_min is not None else [2] * K
        self.bias = list(cfg.bias) if cfg.bias is not None else [1.0] * K
        self.IEbar = list(cfg.IEbar) if cfg.IEbar is not None else [0.0] * K

    def _links_of(self, sim, k):
        return [frozenset((self.root, s)) for s in sim.tasks[k].sensors]

    def attempts(self, sim):
        net, rng, cfg = sim.net, sim.net.rng, self.cfg
        K = len(sim.tasks)
        scores = []
        for k, task in enumerate(sim.tasks):
            miss = sum(1 for key in self._links_of(sim, k) if not net.pairs_on(key))
            if cfg.prio == F.PRIO_THROUGHPUT:
                sc = 1.0 / self.ET[k, miss] if miss > 0 else 1e18
            elif cfg.prio == F.PRIO_MARGINAL:
                Jp = kalman_predict(task.J, task.q)
                gain = max(0.0, task.beta * (1 / (Jp + task.I_C) - 1 / (Jp + self.IEbar[k])))
                sc = gain / self.ET[k, miss] if miss > 0 else 1e18
            elif cfg.prio in (F.PRIO_STATIC, F.PRIO_RANDOM):
                sc = 1.0
            else:
                raise NotImplementedError
            scores.append(sc * self.bias[k] + 1e-12 * rng.random())
        cand = []
        if cfg.prio == F.PRIO_RANDOM:
            cand = [key for k in range(K) for key in self._links_of(sim, k) if not net.pairs_on(key)]
            rng.shuffle(cand)
        else:
            for k in sorted(range(K), key=lambda k: -scores[k]):
                c = [key for key in self._links_of(sim, k) if not net.pairs_on(key)]
                rng.shuffle(c)
                cand += c
        return cand  # Network.attempt enforces the memory limit in this order

    def decide(self, sim):
        net, cfg = sim.net, self.cfg
        discards, deliveries = [], []
        # cutoff discards
        for p in list(net.inventory.values()):
            k = self._task_of(sim, p)
            if p.age > self.cutoff[k]:
                discards.append(p.pid)
        gone = set(discards)
        for k, task in enumerate(sim.tasks):
            stored = []
            for key in self._links_of(sim, k):
                for p in net.pairs_on(key):
                    if p.pid not in gone:
                        stored.append(p)
            n = len(stored)
            if n == 0:
                continue
            stored.sort(key=lambda p: -net.w(p))
            ws = [net.w(p) for p in stored]
            oldest = max(p.age for p in stored)
            m_use = 0
            if cfg.cons == F.CONS_FULL_ALWAYS:
                m_use = task.N if n == task.N else 0
            elif cfg.cons == F.CONS_FULL_THRESH:
                if n == task.N:
                    if np.prod(ws) > advantage_threshold(task.N, task.v):
                        m_use = task.N
                    else:
                        discards.append(stored[-1].pid)
                        gone.add(stored[-1].pid)
            else:
                trig = True if cfg.cons == F.CONS_MYOPIC_SUBSET else (n == task.N or oldest >= self.cutoff[k])
                if trig:
                    best_m, best_I, V = 0, task.I_C, 1.0
                    for m in range(1, n + 1):
                        V *= ws[m - 1]
                        if m >= self.m_min[k]:
                            I = info_hybrid(task.N, m, V, task.v)
                            if I > best_I:
                                best_I, best_m = I, m
                    m_use = best_m
            if m_use == 0 and n == task.N and cfg.cons in (F.CONS_MYOPIC_SUBSET, F.CONS_WAIT_SUBSET):
                discards.append(stored[-1].pid)
                gone.add(stored[-1].pid)
            if m_use > 0:
                pids = [p.pid for p in stored[:m_use]]
                gone.update(pids)
                deliveries.append(Delivery(k, m_use, pids))
                if cfg.cons in (F.CONS_MYOPIC_SUBSET, F.CONS_WAIT_SUBSET):
                    for p in stored[m_use:]:
                        discards.append(p.pid)
                        gone.add(p.pid)
        # deadlock breaker
        remaining = [p for p in net.inventory.values() if p.pid not in gone]
        if len(remaining) >= net.Q[self.root]:
            anyfull = any(sum(1 for key in self._links_of(sim, k) for p in net.pairs_on(key)
                              if p.pid not in gone) == t.N for k, t in enumerate(sim.tasks))
            if not anyfull and remaining:
                oldest = max(remaining, key=lambda p: p.age)
                discards.append(oldest.pid)
        return discards, deliveries

    def _task_of(self, sim, p):
        for k, task in enumerate(sim.tasks):
            if self.root in (p.u, p.v) and (p.u in task.sensors or p.v in task.sensors):
                return k
        raise KeyError


def build_repeater_sim(scn: 'F.Scenario', cfg: 'F.PolicyCfg', seed: int = 0):
    """General-engine instance of a K-star-sharing-one-repeater scenario."""
    from .network import Network, Link
    from .tasks import SensingTask
    from .sim import Simulator
    from .physics import J_star
    rng = np.random.default_rng(seed)
    nodes = {'R': {'Q': scn.Q_R, 'tau': math.inf}}
    links, tasks = [], []
    for k, tc in enumerate(scn.tasks):
        sensors = [f't{k}s{i}' for i in range(tc.N)]
        for s in sensors:
            # put all decay on the sensor side: eta_s = rho, eta_R = 1
            tau_s = -1.0 / math.log(tc.rho) if tc.rho < 1 else math.inf
            nodes[s] = {'Q': 1, 'tau': tau_s}
            links.append(Link('R', s, tc.p, tc.w0))
        t = SensingTask(f'task{k}', sensors, 'R', q=tc.q, v=tc.v, beta=tc.beta)
        t.J = J_star(t.I_C, t.q) if t.q > 0 else t.I_C
        tasks.append(t)
    net = Network(nodes, links, rng)
    pol = RepeaterPolicy(cfg, scn)
    return Simulator(net, tasks, pol)
