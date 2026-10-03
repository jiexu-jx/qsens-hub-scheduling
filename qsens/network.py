"""Network layer: graph, links, memories, stored-pair inventory, heralds.

Implements Assumptions 1-3 of the report exactly:
  * one generation attempt per link per slot, needing a free memory qubit at both
    ends (the attempt occupies the qubit during the slot);
  * success w.p. p_e, heralded, producing rho_W(w0_e) with age 0;
  * stored qubits depolarize with eta_u per slot, so a pair on (u,v) of age a has
    w = w0 (eta_u eta_v)^a  (eq. (3));
  * swaps/teleportations are perfect and complete within the slot.
"""
from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .physics import eta_from_tau


@dataclass
class Link:
    u: str
    v: str
    p: float
    w0: float

    @property
    def key(self):
        return frozenset((self.u, self.v))


@dataclass
class Pair:
    pid: int
    link: frozenset
    u: str
    v: str
    w0: float
    created: int
    age: int = 0

    def w(self, eta: dict) -> float:
        return self.w0 * (eta[self.u] * eta[self.v]) ** self.age


class Network:
    def __init__(self, nodes: dict, links: list[Link], rng: np.random.Generator):
        """nodes: {name: {'Q': memory qubits, 'tau': lifetime in slots}}"""
        self.nodes = nodes
        self.Q = {n: d['Q'] for n, d in nodes.items()}
        self.eta = {n: eta_from_tau(d.get('tau', math.inf)) for n, d in nodes.items()}
        self.links = {l.key: l for l in links}
        self.rng = rng
        self.inventory: dict[int, Pair] = {}
        self._pid = itertools.count()
        self.t = 0
        # statistics
        self.n_generated = 0
        self.n_discarded = 0
        self.n_consumed = 0

    # ---------------------------------------------------------------- memory
    def used(self, node: str) -> int:
        return sum(1 for p in self.inventory.values() if node in (p.u, p.v))

    def free(self, node: str) -> int:
        return self.Q[node] - self.used(node)

    def pairs_on(self, key) -> list[Pair]:
        return [p for p in self.inventory.values() if p.link == key]

    def w(self, pair: Pair) -> float:
        return pair.w(self.eta)

    # ---------------------------------------------------------------- slot ops
    def attempt(self, keys) -> list[Pair]:
        """Attempt generation on the given links (Assumption 1). Memory is checked
        jointly: each attempt reserves one qubit at both endpoints."""
        new = []
        attempted = []
        reserved = {n: 0 for n in self.nodes}
        for key in keys:
            l = self.links[key]
            if self.free(l.u) - reserved[l.u] <= 0 or self.free(l.v) - reserved[l.v] <= 0:
                continue
            reserved[l.u] += 1
            reserved[l.v] += 1
            attempted.append(l)
        for l in attempted:
            if self.rng.random() < l.p:
                pr = Pair(next(self._pid), l.key, l.u, l.v, l.w0, self.t)
                self.inventory[pr.pid] = pr
                new.append(pr)
                self.n_generated += 1
        self.last_attempted = attempted
        return new

    def discard(self, pids):
        for pid in pids:
            if pid in self.inventory:
                del self.inventory[pid]
                self.n_discarded += 1

    def consume(self, pids) -> float:
        """Consume pairs (swaps/teleportation). Returns product of their w."""
        V = 1.0
        for pid in pids:
            V *= self.w(self.inventory[pid])
            del self.inventory[pid]
            self.n_consumed += 1
        return V

    def end_slot(self):
        for p in self.inventory.values():
            p.age += 1
        self.t += 1

    # ---------------------------------------------------------------- routing
    def best_path(self, src: str, dst: str, exclude: set) -> tuple[float, list[int]] | None:
        """Max-product (min sum of -log w) path of stored pairs from src to dst on the
        inventory graph, not using pairs in `exclude` (eq. (11))."""
        adj: dict[str, list[tuple[str, float, int]]] = {}
        for p in self.inventory.values():
            if p.pid in exclude:
                continue
            c = -math.log(max(self.w(p), 1e-300))
            adj.setdefault(p.u, []).append((p.v, c, p.pid))
            adj.setdefault(p.v, []).append((p.u, c, p.pid))
        dist = {src: 0.0}
        prev: dict[str, tuple[str, int]] = {}
        h = [(0.0, src)]
        while h:
            d, x = heapq.heappop(h)
            if x == dst:
                break
            if d > dist.get(x, math.inf):
                continue
            for y, c, pid in adj.get(x, []):
                nd = d + c
                if nd < dist.get(y, math.inf):
                    dist[y] = nd
                    prev[y] = (x, pid)
                    heapq.heappush(h, (nd, y))
        if dst not in dist:
            return None
        path = []
        x = dst
        while x != src:
            x, pid = prev[x]
            path.append(pid)
        return math.exp(-dist[dst]), path[::-1]

    def ghz_delivery(self, root: str, sensors: list[str], m: int | None = None):
        """Teleportation-based GHZ delivery (Lemma 3) rooted at `root` to the best
        m of `sensors`: greedy pair-disjoint best paths. The root's own qubit (if the
        root is a sensor) needs no pairs. Returns (V, chosen sensors, pair ids) or
        None if fewer than m sensors are reachable."""
        m = len(sensors) if m is None else m
        used: set[int] = set()
        chosen, pids, V = [], [], 1.0
        cands = []
        for s in sensors:
            if s == root:
                cands.append((1.0, s, []))
                continue
            r = self.best_path(root, s, used)
            if r is not None:
                cands.append((r[0], s, r[1]))
        # greedy: take best paths first, recompute disjointness
        cands.sort(key=lambda c: -c[0])
        for wpath, s, path in cands:
            if len(chosen) >= m:
                break
            if s != root:
                r = self.best_path(root, s, used)
                if r is None:
                    continue
                wpath, path = r
            chosen.append(s)
            pids.extend(path)
            used.update(path)
            V *= wpath
        if len(chosen) < m:
            return None
        return V, chosen, pids
