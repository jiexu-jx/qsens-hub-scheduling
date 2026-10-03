"""ICC paper Table I: DP-optimal policy vs bounds vs policies on the toy instance."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import icc_random as R
inst = dict(id='toy', K=2, Q=2, tasks=[dict(N=2, p=0.3, w0=0.97, rho=0.99, q=5e-4, v=1.0, beta=1.0),
                                       dict(N=2, p=0.3, w0=0.97, rho=0.99, q=5e-4, v=1.0, beta=5.0)])
out = R.run_instance(inst)
dp = json.load(open(ROOT / 'results' / 'oracle_dp.json'))
out['dp_optimal_executed'] = dp['dp_policy_simulated']
out['dp_grid_gain'] = dp['g_dp_grid']
json.dump(out, open(ROOT / 'results' / 'icc_toy.json', 'w'), indent=1)
