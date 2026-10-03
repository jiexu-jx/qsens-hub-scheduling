"""ICC paper figures and summary statistics."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
S1, S2, S3, GRAY, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#9a9993', '#0b0b0b', '#52514e'
plt.rcParams.update({'text.usetex': True, 'text.latex.preamble': r'\usepackage{times}\usepackage{amsmath}',
                     'font.size': 8, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman'], 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '#8a8984', 'axes.linewidth': 0.6,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6})

d = json.load(open(R / 'icc_random.json'))
POLS = ['B4', 'B3', 'B2', 'B2T', 'TPW', 'PA', 'LIP']
LABEL = {'B4': 'FIFO (B4)', 'B3': 'Fidelity-Age (B3)', 'B2': 'Throughput (B2)', 'B2T': 'Throughput+thr. (B2T)',
         'TPW': 'Throughput + our rule', 'PA': 'Ours (PAI)', 'LIP': 'Ours (LIP)'}
stats = {}
cap = {p: [] for p in POLS}
for x in d:
    B0 = x['results']['B0']['wmse']; LB = x['lb_lag']
    for p in POLS:
        cap[p].append(100 * (B0 - x['results'][p]['wmse']) / (B0 - LB))
for p in POLS:
    c = np.array(cap[p])
    stats[f'capture_{p}'] = dict(median=float(np.median(c)), mean=float(c.mean()), p10=float(np.percentile(c, 10)),
                                 p90=float(np.percentile(c, 90)))
cs = np.array([100 * (x['results']['B0']['wmse'] - x['results']['LIPs']['wmse']) / (x['results']['B0']['wmse'] - x['lb_lag']) for x in d])
stats['capture_LIPs'] = dict(median=float(np.median(cs)), p10=float(np.percentile(cs, 10)))
pa = np.array([x['results']['PA']['wmse'] for x in d])
for b in ['B2', 'B2T', 'B3', 'B4', 'TPW', 'PAM']:
    bw = np.array([x['results'][b]['wmse'] for x in d]); g = 100 * (1 - pa / bw)
    stats[f'PA_vs_{b}'] = dict(median=float(np.median(g)), max=float(g.max()), min=float(g.min()),
                               frac_gt5=float((g > 5).mean()), frac_gt10=float((g > 10).mean()),
                               frac_worse=float((g < -0.2).mean()))
head = np.array([100 * (1 - x['lb_lag'] / x['results']['B0']['wmse']) for x in d])
achieved = np.array([100 * (1 - x['results']['PA']['wmse'] / x['results']['B0']['wmse']) for x in d])
gap = np.array([100 * (x['results']['PA']['wmse'] / x['lb_lag'] - 1) for x in d])
simple = np.array([100 * (1 - x['lb_simple'] / x['results']['B0']['wmse']) for x in d])
stats['headroom_lag'] = dict(median=float(np.median(head)), min=float(head.min()), max=float(head.max()))
stats['headroom_simple'] = dict(median=float(np.median(simple)), min=float(simple.min()), max=float(simple.max()))
stats['PA_reduction_vs_B0'] = dict(median=float(np.median(achieved)), max=float(achieved.max()))
stats['PA_gap_to_LB'] = dict(median=float(np.median(gap)), p90=float(np.percentile(gap, 90)), max=float(gap.max()))
lipw = np.array([x['results']['LIP']['wmse'] for x in d])
for b in ['B2', 'B2T', 'B3', 'B4', 'TPW', 'PA']:
    bw = np.array([x['results'][b]['wmse'] for x in d]); g = 100 * (1 - lipw / bw)
    stats[f'LIP_vs_{b}'] = dict(median=float(np.median(g)), max=float(g.max()), min=float(g.min()))
gl = np.array([100 * (x['results']['LIP']['wmse'] / x['lb_lag'] - 1) for x in d])
stats['LIP_gap_to_LB'] = dict(median=float(np.median(gl)), p90=float(np.percentile(gl, 90)), max=float(gl.max()))
stats['LIP_better'] = int((lipw < pa).sum())
stats['max_se_rel'] = float(max(v['se'] / v['wmse'] for x in d for v in x['results'].values()))
# per-K capture
for K in sorted(set(x['K'] for x in d)):
    for p in ('PA', 'B2T', 'TPW'):
        c = [100 * (x['results']['B0']['wmse'] - x['results'][p]['wmse']) / (x['results']['B0']['wmse'] - x['lb_lag'])
             for x in d if x['K'] == K]
        stats[f'capture_{p}_K{K}'] = float(np.median(c))
# heterogeneity: value-index spread vs PA advantage over TPW (allocation effect)
het = np.array([np.log10(max(x['value_index']) / max(min(x['value_index']), 1e-12)) for x in d])
adv = np.array([100 * (x['results']['TPW']['wmse'] - x['results']['PA']['wmse']) /
                (x['results']['B0']['wmse'] - x['lb_lag']) for x in d])
stats['spearman_het_adv'] = float(np.corrcoef(np.argsort(np.argsort(het)), np.argsort(np.argsort(adv)))[0, 1])
lo, hi = het < np.median(het), het >= np.median(het)
stats['alloc_adv_low_het'] = float(np.median(adv[lo])); stats['alloc_adv_high_het'] = float(np.median(adv[hi]))
stats['n_instances'] = len(d)

# ---- Fig. 3 (figure*): (a) capture distribution, (b) regime sweep, (c) memory sizing
rg = json.load(open(R / 'icc_regime.json'))
ps = [x['p'] for x in rg]
def red(x, key):
    B0 = x['results']['B0']['wmse']
    return 100 * (1 - (x[key] if key.startswith('lb') else x['results'][key]['wmse']) / B0)
ms = json.load(open(R / 'icc_memsize.json'))
mrows = ms['memsize']['0.8']
SHORT = {'B4': 'B4', 'B3': 'B3', 'B2': 'B2', 'B2T': 'B2T', 'TPW': 'TPW',
         'PA': 'PAI (ours)', 'LIP': 'LIP (ours)'}
fig = plt.figure(figsize=(7.16, 1.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.32, left=0.085, right=0.995, top=0.81, bottom=0.2)
ax = fig.add_subplot(gs[0])
data = [cap[p] for p in POLS]
bp = ax.boxplot(data, orientation='horizontal', widths=0.6, patch_artist=True, showfliers=True,
                medianprops=dict(color=INK, lw=1.0), whiskerprops=dict(color=MUTED, lw=0.7),
                capprops=dict(color=MUTED, lw=0.7),
                flierprops=dict(marker='o', ms=1.8, mfc=GRAY, mec='none', alpha=0.7))
for patch, p in zip(bp['boxes'], POLS):
    patch.set_facecolor(S1 if p in ('PA', 'LIP') else '#d9d8d3')
    patch.set_edgecolor(S1 if p in ('PA', 'LIP') else GRAY)
    patch.set_alpha(0.9 if p == 'LIP' else (0.45 if p == 'PA' else 1.0))
ax.set_yticks(range(1, len(POLS) + 1)); ax.set_yticklabels([SHORT[p] for p in POLS], fontsize=7)
ax.set_xlabel(r'Gain over B0, \% of $\bar{\mathcal{E}}_0 - L_2$')
ax.set_xlim(-3, 103); ax.set_xticks([0, 25, 50, 75, 100])
ax.grid(axis='x', color='#e6e5e0', lw=0.5); ax.set_axisbelow(True)
ax.set_title(r'(a) Random instances ($n=%d$)' % len(d), fontsize=8, loc='left', pad=3)
# (b) regime
ax = fig.add_subplot(gs[1])
h = []
h += ax.plot(ps, [red(x, 'lb_lag') for x in rg], color=INK, lw=1.2, ls='--', label='Bound $L_2$')
h += ax.plot(ps, [red(x, 'LIP') for x in rg], color=S1, lw=1.6, marker='o', ms=2.6, label='LIP (ours)')
h += ax.plot(ps, [red(x, 'PA') for x in rg], color=S1, lw=1.0, ls='-.', alpha=0.6, label='PAI (ours)')
h += ax.plot(ps, [red(x, 'B2T') for x in rg], color=S3, lw=1.4, marker='^', ms=2.6, label='B2T')
ax.set_xlabel('Link success probability $p$ ($Q=6$)')
ax.set_ylabel(r'MSE reduction vs.\ B0 (\%)')
ax.grid(color='#e6e5e0', lw=0.5); ax.set_axisbelow(True)
ax.set_xlim(0, 1); ax.set_ylim(0, 37)
ax.set_title('(b) Varying link quality', fontsize=8, loc='left', pad=3)
# (c) memory sizing
ax = fig.add_subplot(gs[2])
Qs = [r['Q'] for r in mrows]
for q, lab, ha in ((6, r'$N_3$', 'right'), (10, r'$N_2{+}N_3$', 'right'), (13, r'$\sum_k\! N_k$', 'right')):
    ax.axvline(q, color='#d9d8d3', lw=0.8, ls=(0, (2, 2)), zorder=0)
    ax.text(q - 0.12, 1.0, lab, ha=ha, va='bottom', fontsize=6.5, color=MUTED,
            bbox=dict(fc='white', ec='none', pad=0.6))
ax.plot(Qs, [r['head'] for r in mrows], color=INK, lw=1.2, ls='--')
ax.plot(Qs, [r['LIP'] for r in mrows], color=S1, lw=1.6, marker='o', ms=2.6)
ax.plot(Qs, [r['B2T'] for r in mrows], color=S3, lw=1.4, marker='^', ms=2.6)
ax.set_xlabel('Hub memory $Q$ ($p=0.8$)')
ax.set_xticks(range(2, 14, 2)); ax.set_xlim(1.6, 13.4); ax.set_ylim(0, 37)
ax.grid(axis='y', color='#e6e5e0', lw=0.5); ax.set_axisbelow(True)
ax.set_title('(c) Memory sizing', fontsize=8, loc='left', pad=3)
fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.68, 1.0), ncol=4, frameon=False, fontsize=7,
           handlelength=2.0, columnspacing=1.1)
fig.savefig(OUT / 'fig_results.pdf'); fig.savefig(OUT / 'fig_results.png', dpi=220)
stats['memsize'] = {p: [{k: r[k] for k in ('Q', 'head', 'LIP', 'B2T')} for r in rows] for p, rows in ms['memsize'].items()}
stats['walk'] = json.load(open(R / 'icc_walkthrough.json'))
stats['perK'] = ms['perK']; stats['runtime'] = ms['runtime']
stats['regime'] = [{'p': x['p'], 'lb_lag': red(x, 'lb_lag'), 'lb_simple': red(x, 'lb_simple'), 'PA': red(x, 'PA'), 'LIP': red(x, 'LIP'),
                    'TPW': red(x, 'TPW'), 'B2T': red(x, 'B2T'),
                    'PA_vs_B2T': 100 * (1 - x['results']['PA']['wmse'] / x['results']['B2T']['wmse']),
                    'LIP_vs_B2T': 100 * (1 - x['results']['LIP']['wmse'] / x['results']['B2T']['wmse'])} for x in rg]
toy = json.load(open(R / 'icc_toy.json'))
stats['toy'] = dict(dp=toy['dp_optimal_executed'], lb_lag=toy['lb_lag'], lb_simple=toy['lb_simple'],
                    **{k: v['wmse'] for k, v in toy['results'].items()})
sc = json.load(open(R / 'icc_scaling.json'))
stats['scaling'] = {}
for r in (1, 2, 4, 8):
    for p in ('LIP', 'LIPs', 'PA', 'B2T'):
        g = [100 * (row[p] / x['lb'] - 1) for x in sc for row in x['rows'] if row['r'] == r]
        stats['scaling'][f'{p}_{r}'] = dict(median=float(np.median(g)), max=float(max(g)))
stats['scaling']['n'] = len(sc)
json.dump(stats, open(OUT / 'icc_stats.json', 'w'), indent=1)
print(json.dumps(stats, indent=1))
