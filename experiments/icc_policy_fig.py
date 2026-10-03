"""ICC paper Fig. 4: structure of LIP's deliver-or-wait rule u_k^*(s, S) for the largest
cluster (N=6, beta=10) of the regime instance, at two link success probabilities.
States shown: n stored pairs, all of age l, and no new heralds in the slot."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qsens.lip import cluster_tables
from qsens.lagbound import _encode_desc

S1, S2, INK, MUTED = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'
plt.rcParams.update({'text.usetex': True, 'text.latex.preamble': r'\usepackage{times}\usepackage{amsmath}',
                     'font.size': 8, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman'], 'axes.edgecolor': '#8a8984',
                     'axes.linewidth': 0.6, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.labelcolor': INK, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6})
AGES = range(1, 11)
K = 2  # the N=6, beta=10 cluster


def decisions(x, k):
    t = x['tasks'][k]; mu = x['lip_setup']['muk'][k]
    tb = cluster_tables(t['N'], t['p'], t['w0'], t['rho'], x['Q'], mu, t['v'], 10)
    N = min(t['N'], x['Q'])
    D = np.zeros((N, len(AGES), 2), dtype=int)  # (consumed m, kept)
    for n in range(1, N + 1):
        for j, l in enumerate(AGES):
            buf = np.array([l] * n + [0] * 8, dtype=np.int64)
            s = tb['lut'][_encode_desc(buf, n, tb['B'])]
            D[n - 1, j] = (bin(tb['Dmask'][s, 0]).count('1'), tb['Dkeep'][s, 0])
    return D, N, mu


def blend(c, a):
    c = np.array(to_rgb(c)); return tuple(1 - a + a * c)


rg = json.load(open(ROOT / 'results' / 'icc_regime.json'))
pick = {round(x['p'], 2): x for x in rg}
panels = [0.3, 0.8]
fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.45), sharey=True)
summary = {}
for ax, p in zip(axes, panels):
    D, N, mu = decisions(pick[p], K)
    summary[p] = dict(mu=mu, D=D.tolist())
    for i in range(N):
        for j in range(len(AGES)):
            m, kp = D[i, j]; n = i + 1
            if m > 0:
                fc = blend(S1, 0.25 + 0.75 * m / N); txt = str(m)
                tc = 'white' if m / N > 0.55 else INK
            elif kp < n:
                fc = blend(S2, 0.55); txt = r'$\times$'; tc = INK
            else:
                fc = '#efeeea'; txt = ''; tc = INK
            ax.add_patch(plt.Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, fc=fc, ec='none'))
            if txt:
                ax.text(j + 0.5, i + 0.5, txt, ha='center', va='center', fontsize=7, color=tc)
    ax.set_xlim(0, len(AGES)); ax.set_ylim(0, N)
    ax.set_xticks([j + 0.5 for j in range(0, len(AGES), 3)])
    ax.set_xticklabels([str(AGES[j]) for j in range(0, len(AGES), 3)])
    ax.set_yticks([i + 0.5 for i in range(N)]); ax.set_yticklabels([str(i + 1) for i in range(N)])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f'$p={p}$', fontsize=8, pad=2)
    ax.set_xlabel(r'Age of stored pairs, $\ell$')
axes[0].set_ylabel('Pairs held, $n$')
fig.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig(ROOT / 'figures' / 'fig_policy.pdf'); fig.savefig(ROOT / 'figures' / 'fig_policy.png', dpi=220)
json.dump(summary, open(ROOT / 'results' / 'icc_policy_fig.json', 'w'))
for p in panels:
    print(p, 'mu', round(summary[p]['mu'], 3))
    for i, row in enumerate(summary[p]['D']):
        print(' n', i + 1, ' '.join(f'{a}/{b}' for a, b in row))
