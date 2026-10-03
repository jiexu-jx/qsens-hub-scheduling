# qsens: sensing-aware entanglement scheduling at a shared quantum hub

Simulator, bounds, and policies for the paper *"Scheduling Entanglement for Distributed
Quantum Sensing at a Shared Quantum Hub"* (under submission).

A hub with `Q` memory qubits serves `K` clusters of sensors. In every slot it decides
which links attempt entanglement generation and when to fuse decaying Bell pairs into
(partial) GHZ states. Each cluster tracks a drifting phase with a Kalman filter; the
objective is the long-run weighted MSE.

## Contents

| Path | What it does |
|---|---|
| `qsens/physics.py` | Werner-pair algebra, GHZ visibility, hybrid information formula, Kalman recursion |
| `qsens/fastsim.py` | numba slot-level simulator for `K` clusters sharing one hub, plus baseline policies (B0, B2, B2T, B3, B4, TPW, PAI) |
| `qsens/lagbound.py` | single-cluster MDP (relative value iteration), frontier `F_k(M)`, Lagrangian lower bound `L2` |
| `qsens/lip.py` | Lagrangian index policy (LIP): offline tables at price `mu_k = mu*/lambda_k`, online greedy allocation |
| `qsens/bound.py` | closed-form bound (kept for the journal version) |
| `qsens/dm.py` | density-matrix checks of the teleportation/visibility model |
| `qsens/network.py`, `sim.py`, `schedulers.py`, `tasks.py`, `index.py` | reference engine for general graphs and earlier policies |
| `experiments/` | scripts that produce every number and figure in the paper (see below) |
| `results/` | saved outputs (JSON) of the experiments, so figures can be regenerated without rerunning |
| `tests/` | unit tests |

Comments in some modules refer to lemma and proposition numbers of an earlier technical
report, not of the conference paper.

## Setup

```
pip install -r requirements.txt
python -m pytest -q tests
```

Tested with Python 3.13, numpy 2.5, scipy 1.18, matplotlib 3.11, numba 0.68. The figure
scripts render text with LaTeX (`text.usetex`), so a TeX installation with the `cm-super`
fonts is needed for them.

## Regenerating the paper's figures from saved results (seconds)

```
python experiments/icc_figures.py     # Fig. 2 and figures/icc_stats.json (all numbers quoted in the text)
python experiments/icc_policy_fig.py  # Fig. 3 (LIP consumption rule)
```

Outputs go to `figures/`.

## Rerunning the experiments

Run in this order; each script reads and updates files in `results/`. Seeds are fixed.
The random-instance, scaling, and LIP evaluations use all CPU cores and take the longest.

1. `icc_toy.py`: two-cluster benchmark with the DP policy
2. `icc_regime.py`: three-cluster hub, sweep of the link success probability `p`
3. `icc_random.py`: 120 random instances, baseline policies
4. `icc_recompute_lb.py results/icc_toy.json results/icc_regime.json results/icc_random.json`: lower bounds (frontiers cached in `results/frontier_cache.json`)
5. `icc_lip.py`: LIP with and without the staleness factor
6. `icc_scaling.py`: replicated hubs (`r = 1, 2, 4, 8`)
7. `icc_memsize.py`, `icc_memsize_lip.py`: memory sizing (sweep of `Q`)
8. `icc_walkthrough.py`: the one-slot example of Section IV
9. `icc_figures.py`, `icc_policy_fig.py`: figures and statistics

`validate.py` checks the simulator against closed forms and the density-matrix model;
`gonogo.py`, `analyze_gonogo.py`, `oracle_dp.py`, and `lagcheck_gonogo.py` are earlier
exploratory studies.

## Slot order

attempts and heralds → consumption, keep, discard → sensing and Kalman update → ageing.
