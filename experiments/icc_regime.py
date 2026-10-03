"""ICC paper: headroom vs entanglement attempts per sensing round.
A fixed 3-cluster hub instance; the per-slot link success probability p (which grows
with the number of generation attempts per sensing interrogation) is swept."""
import json, sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
import icc_random as R

PS = (0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 0.95)


def inst(p):
    tasks = [dict(N=3, p=p, w0=0.97, rho=0.995, q=1e-3 / 3, v=1.0, beta=1.0),
             dict(N=4, p=p, w0=0.98, rho=0.995, q=1e-3 / 4, v=1.0, beta=3.0),
             dict(N=6, p=p, w0=0.96, rho=0.995, q=1e-3 / 6, v=1.0, beta=10.0)]
    return dict(id=f'regime-p{p}', K=3, Q=6, tasks=tasks, p=p)


if __name__ == '__main__':
    with Pool(2) as pool:
        out = pool.map(R.run_instance, [inst(p) for p in PS], chunksize=1)
    json.dump(out, open(ROOT / 'results' / 'icc_regime.json', 'w'), indent=1)
