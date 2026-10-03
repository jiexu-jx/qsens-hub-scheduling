"""Fast unit tests (pytest -q). Heavier statistical validation: experiments/validate.py."""
import math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qsens import physics as P, dm, fastsim as F
from qsens.schedulers import build_repeater_sim


def test_lemma1_swap():
    assert abs(dm.swap_werner_numeric(0.9, 0.7) - P.fidelity_from_w(0.63)) < 1e-12


def test_lemma3_ghz_visibility():
    r = dm.teleported_ghz([0.9, 0.8, 0.95])
    assert abs(2 * abs(r[0, 7]) - 0.684) < 1e-12
    assert abs(dm.parity_expectation(r) - 0.684) < 1e-12


def test_lemma4_parity_mean():
    r = dm.encode_phases(dm.teleported_ghz([0.9, 0.85]), [0.3, 0.4], v=0.9)
    assert abs(dm.parity_expectation(r) - 0.9 * 0.85 * 0.81 * math.cos(0.7)) < 1e-12


def test_cor1_numbers():
    assert P.cor1_max_helpful_N(0.99) == 280
    assert P.cor1_max_helpful_N(0.95) == 34
    assert P.cor1_max_helpful_N(0.90) == 11
    assert P.cor1_max_helpful_N(0.80) == 0
    assert abs(P.cor1_Nopt(0.95) - 19.5) < 0.01


def test_prop2_numbers():
    assert abs(P.prop2_mopt_continuous(0.95) - 22.2) < 0.01
    assert abs(P.prop2_mopt_continuous(0.90) - 7.39) < 0.01
    assert abs(P.info_hybrid(5, 5, 0.8) - P.info_entangled(5, 0.8)) < 1e-12
    assert P.info_hybrid(5, 0, 0.8) == P.info_classical(5)


def test_prop3_fixed_point():
    J = 0.0
    for _ in range(20000):
        J = P.kalman_step(J, 1e-2, 4.0)
    assert abs(J - P.J_star(4.0, 1e-2)) < 1e-9


def test_prop4_report_values():
    assert abs(P.prop4_ET(4, 0.1) - 20.2734) < 1e-3
    assert abs(P.prop4_EV2(4, 0.1, 0.98, 0.95) - 0.19815) < 1e-5


def test_fast_kernel_prop4():
    scn = F.Scenario([F.TaskCfg(N=3, p=0.2, w0=0.95, rho=0.98, q=0.0)], Q_R=3)
    pol = F.PolicyCfg('cc', F.PRIO_STATIC, F.CONS_FULL_ALWAYS, cutoff=(None,))
    r = F.run(scn, pol, T=600_000, burn=1000, seed=3, J0=[1.0])
    assert abs(1 / r['deliv'][0] / P.prop4_ET(3, 0.2) - 1) < 0.01
    assert abs(r['meanV2'][0] / P.prop4_EV2(3, 0.2, 0.98, 0.95) - 1) < 0.01


def test_crossval_reference_engine():
    scn = F.Scenario([F.TaskCfg(N=2, p=0.25, w0=0.95, rho=0.99, q=1e-3),
                      F.TaskCfg(N=3, p=0.15, w0=0.97, rho=0.995, q=1e-3)], Q_R=3)
    pol = F.PolicyCfg('t', F.PRIO_THROUGHPUT, F.CONS_WAIT_SUBSET, cutoff=(6, 10))
    T = 40000
    sim = build_repeater_sim(scn, pol, seed=1).run(T)
    w_ref = np.mean((1 / np.array(sim.log['J'][2000:])).sum(1))
    w_fast = np.mean([F.run(scn, pol, T=T, burn=2000, seed=s)['wmse'] for s in range(5)])
    assert abs(w_ref / w_fast - 1) < 0.003


def test_bounds_toy():
    from qsens.bound import lower_bound
    from qsens.lagbound import lag_lower_bound
    tasks = [F.TaskCfg(N=2, p=0.3, w0=0.97, rho=0.99, q=5e-4, beta=1.0),
             F.TaskCfg(N=2, p=0.3, w0=0.97, rho=0.99, q=5e-4, beta=5.0)]
    l1 = lower_bound(tasks, 2)['lb']
    l2 = lag_lower_bound(tasks, 2)['lb']
    assert l1 <= l2 <= 0.08827  # 0.08827 = exact DP optimum (results/oracle_dp.json)
    assert abs(l2 - 0.08821) < 1e-4
