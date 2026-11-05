#!/usr/bin/env python3
"""Session-load, working-set-complexity and reporting-interval sweeps."""
import argparse,copy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fluidrender.core import load_config,merge_config
from fluidrender.experiments import run_experiment,write_summary
p=argparse.ArgumentParser()
p.add_argument('--config',default=str(ROOT/'configs/default.json'))
p.add_argument('--suite',choices=['sessions','complexity','reporting'],required=True)
p.add_argument('--methods',default='RR,A-LALL')
p.add_argument('--checkpoint',action='append',default=[])
p.add_argument('--seconds',type=float,default=1800)
p.add_argument('--seeds',type=int,default=10)
p.add_argument('--out',default=str(ROOT/'results/sweeps'))
a=p.parse_args();cfg=load_config(a.config)
cfg['simulation'].update(duration_s=a.seconds,warmup_s=min(300.,a.seconds*.15))
for pair in a.checkpoint:
    name,path=pair.split('=',1);cfg.setdefault('checkpoints',{})[name]=path
if a.suite=='sessions':
    cases=[(str(n),{'workload':{'active_sessions':n}}) for n in (50,75,100,125,150,175,200)]
elif a.suite=='complexity':
    cases=[('low',{'workload':{'high_complexity_share':0.}}),('mixed',{}),('high',{'workload':{'high_complexity_share':1.}})]
else:
    cases=[(str(n),{'telemetry':{'cache_report_interval_ms':n,'cache_stale_ms':500}}) for n in (50,100,250,500,1000)]
rows=[]
for condition,overrides in cases:
    result=run_experiment(merge_config(cfg,overrides),a.methods.split(','),a.seeds,Path(a.out)/a.suite/condition)
    rows.extend(dict(condition=condition,method=m,**v) for m,v in result.items())
write_summary(Path(a.out)/a.suite/'summary.csv',rows)
