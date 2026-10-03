"""ICC paper Fig. 5: memory sizing for the regime hub (N=(3,4,6)) at p=0.8."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
S1, S3, INK, MUTED, GRID = '#2a78d6', '#1baf7a', '#0b0b0b', '#52514e', '#e6e5e0'
plt.rcParams.update({'text.usetex': True, 'text.latex.preamble': r'\usepackage{times}\usepackage{amsmath}',
                     'font.size': 8, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman'],
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#8a8984',
                     'axes.linewidth': 0.6, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6})
o = json.load(open(ROOT / 'results' / 'icc_memsize.json'))
rows = o['memsize']['0.8']
Q = [r['Q'] for r in rows]
fig, ax = plt.subplots(figsize=(3.45, 1.75))
for q, lab in ((6, r'$N_3$'), (10, r'$N_2{+}N_3$'), (13, r'$\sum_k N_k$')):
    ax.axvline(q, color=GRID, lw=0.8, zorder=0)
    ax.text(q, 37.5, lab, ha='center', va='bottom', fontsize=7, color=MUTED)
ax.plot(Q, [r['head'] for r in rows], color=INK, lw=1.2, ls='--', label='Bound $L_2$')
ax.plot(Q, [r['LIP'] for r in rows], color=S1, lw=1.8, marker='o', ms=3, label='LIP')
ax.plot(Q, [r['B2T'] for r in rows], color=S3, lw=1.5, marker='^', ms=3, label='B2T')
ax.set_xlabel('Hub memory $Q$ (qubits)'); ax.set_ylabel(r'MSE reduction vs.\ B0 (\%)')
ax.set_xticks(range(2, 14)); ax.set_xlim(1.6, 13.4); ax.set_ylim(0, 37)
ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=7, loc='lower right', handlelength=1.8)
fig.tight_layout(pad=0.3)
fig.savefig(ROOT / 'figures' / 'fig_memsize.pdf'); fig.savefig(ROOT / 'figures' / 'fig_memsize.png', dpi=220)
