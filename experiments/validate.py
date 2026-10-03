"""E0/E1-validation: check the simulator against the closed forms of Section IV.

V1  Prop. 4  (waiting time E[T], E[V^2], gain G) -- fast kernel and reference engine
V2  Cor. 1 / Prop. 2 (cluster size): closed-form numbers + estimator-level Monte Carlo
V3  Lemmas 1-4 by density matrices (Exercise 2) + exact QFI vs parity (Remark 3)
V4  Prop. 3 (steady state / square-root law)
Outputs results/validation.json and figures in results/.
"""
import json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qsens import physics as P, fastsim as F, dm
from qsens.schedulers import CollectAndConsume
from qsens.network import Network, Link
from qsens.tasks import SensingTask
from qsens.sim import Simulator

OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
res = {}

# ---------------------------------------------------------------- V1 Prop. 4
def star_ref(N, p, rho, w0, T, seed):
    rng = np.random.default_rng(seed)
    tau = -1 / math.log(rho)
    nodes = {'R': {'Q': N, 'tau': math.inf}}
    links = []
    sensors = [f's{i}' for i in range(N)]
    for s in sensors:
        nodes[s] = {'Q': 1, 'tau': tau}
        links.append(Link('R', s, p, w0))
    task = SensingTask('t', sensors, 'R', q=0.0)
    task.J = 1.0
    sim = Simulator(Network(nodes, links, rng), [task], CollectAndConsume()).run(T)
    Vs = np.array([d[0][2] for d in sim.log['deliv'] if d])
    return T / len(Vs), float(np.mean(Vs ** 2)), len(Vs)

t0 = time.time()
rows = []
cfg_id = 0
NSEED = 8
for N in (2, 4, 8):
    for p in (0.05, 0.1, 0.3):
        for rho in (0.95, 0.98, 0.995):
            w0 = 0.95
            ET, EV2 = P.prop4_ET(N, p), P.prop4_EV2(N, p, rho, w0)
            G = P.prop4_gain(N, p, rho, w0)
            scn = F.Scenario([F.TaskCfg(N=N, p=p, w0=w0, rho=rho, q=0.0)], Q_R=N)
            pol = F.PolicyCfg('cc', F.PRIO_STATIC, F.CONS_FULL_ALWAYS, cutoff=(None,))
            cfg_id += 1
            reps = [F.run(scn, pol, T=750_000, burn=1000, seed=1000 * cfg_id + s, J0=[1.0]) for s in range(NSEED)]
            ETs = np.array([1 / r['deliv'][0] for r in reps])
            EV2s = np.array([r['meanV2'][0] for r in reps])
            Gs = np.array([r['meanI'][0] - N for r in reps])
            ncyc = sum(r['deliv'][0] for r in reps) * (750_000 - 1000)
            rows.append(dict(N=N, p=p, rho=rho, w0=w0, ET=ET, ET_sim=ETs.mean(), ET_se=ETs.std(ddof=1) / math.sqrt(NSEED),
                             EV2=EV2, EV2_sim=EV2s.mean(), EV2_se=EV2s.std(ddof=1) / math.sqrt(NSEED),
                             G=G, G_sim=Gs.mean(), G_se=Gs.std(ddof=1) / math.sqrt(NSEED), cycles=int(ncyc)))
# the report's own check, also with the reference engine
ref = star_ref(4, 0.1, 0.98, 0.95, T=400_000, seed=11)
res['V1'] = dict(rows=rows, ref_engine_N4=dict(ET_sim=ref[0], EV2_sim=ref[1], cycles=ref[2],
                 ET=P.prop4_ET(4, 0.1), EV2=P.prop4_EV2(4, 0.1, 0.98, 0.95)))
z_ET = [abs(r['ET_sim'] - r['ET']) / r['ET_se'] for r in rows]
z_EV = [abs(r['EV2_sim'] - r['EV2']) / max(r['EV2_se'], 1e-12) for r in rows]
z_G = [abs(r['G_sim'] - r['G']) / max(r['G_se'], 1e-12) for r in rows]
res['V1']['max_rel_err_ET'] = max(abs(r['ET_sim'] / r['ET'] - 1) for r in rows)
res['V1']['max_rel_err_EV2'] = max(abs(r['EV2_sim'] / r['EV2'] - 1) for r in rows if r['EV2'] > 1e-6)
res['V1']['max_z'] = dict(ET=max(z_ET), EV2=max(z_EV), G=max(z_G))
print('V1 done', time.time() - t0, res['V1']['max_rel_err_ET'], res['V1']['max_rel_err_EV2'], res['V1']['max_z'])

# ---------------------------------------------------------------- V2 cluster size
v2 = {'cor1': [], 'prop2': []}
for w in (0.99, 0.95, 0.90, 0.80):
    v2['cor1'].append(dict(w=w, N_opt=P.cor1_Nopt(w), N_max_helpful=P.cor1_max_helpful_N(w),
                           max_NwN=max(N * w ** (2 * N) for N in range(1, 2000))))
for w in (0.95, 0.90):
    v2['prop2'].append(dict(w=w, m_cont=P.prop2_mopt_continuous(w),
                            m_int_N50=P.prop2_mopt_integer(50, w)))

def mc_hybrid(N, m, w, shots=4000, reps=400, seed=0):
    """Estimator-level MC: m sensors share GHZ of visibility w^m (parity readout at the
    optimal bias), N-m sensors Ramsey (v=1). Each repetition uses `shots` rounds;
    returns shots * Var(psi_hat) (should equal Var_m of eq. (13))."""
    rng = np.random.default_rng(seed)
    V = w ** m
    psi_true = 0.0
    est = np.empty(reps)
    for r in range(reps):
        # GHZ part: mean parity at operating point = -V sin(delta), delta = sum of phases offset
        s_hat = 0.0
        if m > 0:
            pplus = (1 + (-V * math.sin(0.0))) / 2
            k = rng.binomial(shots, pplus)
            mean = 2 * k / shots - 1
            s_hat = -math.asin(np.clip(mean / V, -1, 1))
        th = 0.0
        if N - m > 0:
            ks = rng.binomial(shots, 0.5, size=N - m)
            th = float(np.sum(-np.arcsin(np.clip(2 * ks / shots - 1, -1, 1))))
        est[r] = (s_hat + th) / N
    return shots * est.var()

mc_rows = []
for (N, w) in ((8, 0.9), (14, 0.9), (12, 0.95)):
    for m in range(0, N + 1):
        if m == 1:
            continue
        mc = mc_hybrid(N, m, w, seed=N * 100 + m)
        th = P.prop2_var(N, m, w) if m >= 2 else 1 / N
        mc_rows.append(dict(N=N, w=w, m=m, var_mc=mc, var_formula=th))
v2['mc_hybrid'] = mc_rows
# empirical optimum m and helpfulness thresholds (full GHZ, m=N) for w=0.9
full = []
for N in range(2, 16):
    mc = mc_hybrid(N, N, 0.9, seed=7000 + N)
    full.append(dict(N=N, var_mc=mc, var_formula=P.prop2_var(N, N, 0.9), var_classical=1 / N))
v2['full_w09'] = full
res['V2'] = v2
print('V2 done')

# ---------------------------------------------------------------- V3 density matrices
v3 = {}
v3['swap_0.9_0.7_fidelity'] = dm.swap_werner_numeric(0.9, 0.7)
r3 = dm.teleported_ghz([0.9, 0.8, 0.95])
v3['ghz3_coherence'] = 2 * abs(r3[0, 7])
v3['ghz3_parity'] = dm.parity_expectation(r3)
errs = []
rng = np.random.default_rng(5)
for _ in range(20):
    N = rng.integers(2, 5)
    Ws = rng.uniform(0.6, 1.0, size=N)
    v = rng.uniform(0.8, 1.0)
    phis = rng.uniform(-1, 1, size=N)
    rr = dm.encode_phases(dm.teleported_ghz(Ws), phis, v)
    errs.append(abs(dm.parity_expectation(rr) - np.prod(Ws) * v ** N * math.cos(phis.sum())))
v3['lemma4_max_abs_err'] = float(max(errs))
# teleportation channel
sig = np.array([[0.6, 0.2 - 0.3j], [0.2 + 0.3j, 0.4]])
v3['lemma2_max_abs_err'] = float(np.abs(dm.teleport_channel_numeric(0.7, sig) - (0.7 * sig + 0.3 * np.eye(2) / 2)).max())
# QFI gap (Remark 3)
gap = []
for N in (2, 3, 4, 5):
    for W in (0.99, 0.95, 0.9, 0.8):
        q = dm.qfi_average_phase(dm.teleported_ghz([W] * N))
        gap.append(dict(N=N, W=W, V=W ** N, parity=q['parity'], qfi_multi=q['multi'], qfi_equal=q['equal'],
                        ratio=q['multi'] / q['parity'], classical=N))
v3['qfi_gap'] = gap
res['V3'] = v3
print('V3 done')

# ---------------------------------------------------------------- V4 Prop. 3
v4 = []
for (I, q) in ((4.0, 1e-3), (4.0, 1e-2), (16 * 0.8, 1e-3), (4.0, 0.5)):
    J = 0.0
    for _ in range(100000):
        J = P.kalman_step(J, q, I)
    v4.append(dict(I=I, q=q, J_iter=J, J_star=P.J_star(I, q), sqrt_approx=math.sqrt(I / q)))
# stochastic delivery vs constant delivery with same mean (concavity, TODO in Sec V-E)
rng = np.random.default_rng(1)
q = 1e-3
for pdel, Ihi, Ilo in ((0.1, 40.0, 4.0), (0.05, 76.0, 4.0)):
    Ibar = pdel * Ihi + (1 - pdel) * Ilo
    J = P.J_star(Ibar, q); s = 0.0; T = 400000
    for t in range(T):
        I = Ihi if rng.random() < pdel else Ilo
        J = P.kalman_step(J, q, I); s += 1 / J
    v4.append(dict(kind='bursty_vs_constant', p=pdel, Ibar=Ibar, mse_bursty=s / T,
                   mse_constant=1 / P.J_star(Ibar, q)))
res['V4'] = v4
print('V4 done')

json.dump(res, open(OUT / 'validation.json', 'w'), indent=1, default=float)
print(json.dumps({k: (v if k != 'V1' else {kk: vv for kk, vv in v.items() if kk != 'rows'}) for k, v in res.items() if k in ('V1', 'V3')}, indent=1, default=float)[:3000])

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
cols = {2: S1, 4: S2, 8: S3}
for ax, key, lab in ((axes[0], 'ET', 'E[T] (slots)'), (axes[1], 'EV2', 'E[V$^2$]')):
    for N in (2, 4, 8):
        rr = [r for r in rows if r['N'] == N]
        ax.scatter([r[key] for r in rr], [r[key + '_sim'] for r in rr], s=14, color=cols[N], label=f'N={N}', zorder=3)
    lo = min(min(r[key] for r in rows), min(r[key + '_sim'] for r in rows)); hi = max(r[key] for r in rows) * 1.05
    if key == 'EV2':
        ax.set_xscale('log'); ax.set_yscale('log'); lo = max(lo, 3e-5)
    ax.plot([lo, hi], [lo, hi], color='#8a8984', lw=0.8, zorder=1)
    ax.set_xlabel(f'{lab}, closed form'); ax.set_ylabel(f'{lab}, simulation')
axes[0].legend(frameon=False)
fig_r = [r for r in v2['full_w09']]
axes[2].plot([r['N'] for r in fig_r], [r['var_formula'] for r in fig_r], color=S1, lw=2, label='GHZ, eq. (13) m=N')
axes[2].scatter([r['N'] for r in fig_r], [r['var_mc'] for r in fig_r], s=14, color=S1, label='GHZ, Monte Carlo', zorder=3)
axes[2].plot([r['N'] for r in fig_r], [r['var_classical'] for r in fig_r], color=S2, lw=2, label='classical 1/N')
axes[2].set_xlabel('cluster size N (w = 0.9)'); axes[2].set_ylabel('per-round Var($\\hat\\psi$)')
axes[2].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(OUT / 'fig_validation.png', dpi=200)
fig.savefig(OUT / 'fig_validation.pdf')
