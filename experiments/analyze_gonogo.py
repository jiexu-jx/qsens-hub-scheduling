"""Summarise the go/no-go sweep: tables (markdown/CSV) and figures."""
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'
data = json.load(open(R / 'gonogo_all.json'))

PRIO = {0: 'random', 1: 'throughput', 2: 'freshness', 3: 'FIFO', 4: 'marginal', 5: 'Whittle', 6: 'static'}
CONS = {0: 'full-always', 1: 'full+thr', 2: 'myopic-subset', 3: 'wait-subset'}

rows = []
for s in data:
    r = {k: v['wmse'] for k, v in s['results'].items()}
    B0, BEST = r['B0'], r['BEST']
    BL = min(r['B2'], r['B2T'], r['B3'], r['B4'])
    BLn = min(('B2', 'B2T', 'B3', 'B4'), key=lambda k: r[k])
    PA = min(r['A1'], r['A1W'], r['A2'])
    PAn = min(('A1', 'A1W', 'A2'), key=lambda k: r[k])
    ref = min(BEST, PA)  # best observed policy on test seeds
    head = B0 - ref
    cap = lambda x: (B0 - x) / head if head > 0 else float('nan')
    bc = s['results']['BEST']['cfg']
    b2c = s['results']['B2T*']['cfg']
    rows.append(dict(
        scenario=s['name'], regime=s['name'].split('-')[-1],
        headroom_pct=100 * head / B0,
        B2T_vs_best_pct=100 * (r['B2T'] / ref - 1),
        BL_name=BLn, BL_vs_best_pct=100 * (BL / ref - 1),
        B2Tstar_vs_best_pct=100 * (r['B2T*'] / ref - 1),
        PA_name=PAn, PA_vs_BL_pct=100 * (1 - PA / BL),
        PA_vs_B2Tstar_pct=100 * (1 - PA / r['B2T*']),
        cap_B2T=cap(r['B2T']), cap_BL=cap(BL), cap_B2Tstar=cap(r['B2T*']), cap_A1W=cap(r['A1W']),
        cap_A2=cap(r['A2']), cap_A1=cap(r['A1']), cap_PA=cap(PA),
        go1=PA <= 0.90 * BL, near_opt_B2Tstar=r['B2T*'] <= 1.02 * ref,
        best_prio=PRIO[bc['prio']], best_cons=CONS[bc['cons']], best_cutoff=bc['cutoff'], best_bias=bc['bias'],
        b2t_cons=CONS[b2c['cons']], b2t_cutoff=b2c['cutoff'],
        single_cutoff=s['single']['cutoff'],
        deliv_BEST=s['results']['BEST']['deliv'], meanm_BEST=s['results']['BEST']['meanm'],
        deliv_B2T=s['results']['B2T']['deliv'],
        se_max=max(v['wmse_se'] for v in s['results'].values()) / ref * 100,
        whittle_viol=s['whittle_indexability_violations'],
        wmse={k: round(v, 6) for k, v in r.items()}))

with open(R / 'gonogo_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != 'wmse'])
    w.writeheader()
    for row in rows:
        w.writerow({k: v for k, v in row.items() if k != 'wmse'})
json.dump(rows, open(R / 'gonogo_summary.json', 'w'), indent=1, default=str)

# markdown table
lines = ['| Scenario | Headroom B0→best | B2T gap to best | Best untuned baseline gap | B2T* gap | Precision-aware vs best baseline | Best policy found |',
         '|---|---|---|---|---|---|---|']
for x in rows:
    lines.append(f"| {x['scenario']} | {x['headroom_pct']:.1f}% | {x['B2T_vs_best_pct']:+.1f}% | {x['BL_name']} {x['BL_vs_best_pct']:+.1f}% | "
                 f"{x['B2Tstar_vs_best_pct']:+.1f}% | {x['PA_name']} {x['PA_vs_BL_pct']:.1f}% lower | {x['best_prio']}, {x['best_cons']}, cut={x['best_cutoff']} |")
open(R / 'gonogo_table.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print('max relative SE (%)', max(x['se_max'] for x in rows))
print('go1 count', sum(x['go1'] for x in rows), 'of', len(rows))
print('B2T* near-optimal count', sum(x['near_opt_B2Tstar'] for x in rows))
print('whittle violations', sum(sum(x['whittle_viol']) for x in rows))

# ------------------------------------------------------------------ figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

S1, S2, S3, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#52514e'
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#8a8984', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e',
                     'ytick.color': '#52514e'})
types = ['T1-sym', 'T7-sym-Q6', 'T2-size', 'T4-link', 'T5-conflict', 'T10-conflict-Q3',
         'T3-weight', 'T6-drift', 'T8-quality', 'T9-weight-Q2']
labels = ['T1 symmetric', 'T7 symmetric, Q=6', 'T2 size N=(2,6)', 'T4 link p ratio 3', 'T5 conflict',
          'T10 conflict, Q=3', 'T3 weight β=(1,5)', 'T6 drift q ratio 100', 'T8 link quality', 'T9 weight, Q=2']
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.6), sharey=True)
for ax, reg in zip(axes, ('R1', 'R2', 'R3')):
    y = np.arange(len(types))
    vals = {k: [] for k in ('cap_BL', 'cap_B2Tstar', 'cap_PA')}
    hd = []
    for t in types:
        x = next(r for r in rows if r['scenario'] == f'{t}-{reg}')
        for k in vals:
            vals[k].append(100 * max(x[k], 0))
        hd.append(x['headroom_pct'])
    h = 0.26
    ax.barh(y - h, vals['cap_BL'], h * 0.92, color=S1, label='best untuned baseline (B2–B4)')
    ax.barh(y, vals['cap_B2Tstar'], h * 0.92, color=S2, label='B2T* (throughput, tuned consumption)')
    ax.barh(y + h, vals['cap_PA'], h * 0.92, color=S3, label='best of A1/A1W/A2 (precision-aware, untuned)')
    for yi, hv in zip(y, hd):
        ax.text(103, yi, f'{hv:.0f}%', va='center', fontsize=7, color=MUTED)
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 50, 100])
    ax.grid(axis='x', color='#e6e5e0', lw=0.6)
    ax.set_axisbelow(True)
    p = {'R1': 'p=0.1, ρ=0.99', 'R2': 'p=0.3, ρ=0.99', 'R3': 'p=0.6, ρ=0.995'}[reg]
    ax.set_title(f'{reg}: {p}', fontsize=8, color='#0b0b0b')
    ax.set_xlabel('% of achievable benefit captured')
axes[0].set_yticks(np.arange(len(types)))
axes[0].set_yticklabels(labels)
axes[0].invert_yaxis()
axes[2].text(103, -1.0, 'head-\nroom', fontsize=6.5, color=MUTED, va='bottom')
handles, labs = axes[0].get_legend_handles_labels()
fig.legend(handles, labs, loc='lower center', ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.55, -0.01))
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(R / 'fig_gonogo_capture.png', dpi=200)
fig.savefig(R / 'fig_gonogo_capture.pdf')
print('saved figure')
