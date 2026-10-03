import sys; sys.path.insert(0, ".")
import json, time
from qsens import fastsim as F
from qsens.lagbound import lag_lower_bound
from qsens.bound import lower_bound
d=json.load(open('results/gonogo_all.json'))
cache={}
out=[]
for s in d:
    t0=time.time()
    tasks=[F.TaskCfg(**t) for t in s['tasks']]
    lb=lag_lower_bound(tasks, s['Q_R'], A=10, cache=cache)['lb']
    lb0=lower_bound(tasks, s['Q_R'])['lb']
    best=min(v['wmse'] for v in s['results'].values())
    B0=s['results']['B0']['wmse']
    out.append(dict(name=s['name'],B0=B0,best=best,lag=lb,simple=lb0))
    print(f"{s['name']:24s} best={best:.5f} lag={lb:.5f} simple={lb0:.5f} gap {100*(best/lb-1):5.2f}%  captured {(B0-best)/(B0-lb):.2f}  ({time.time()-t0:.0f}s)", flush=True)
json.dump(out, open('results/lagbound_gonogo.json','w'), indent=1)
