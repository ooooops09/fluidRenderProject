#!/usr/bin/env python3
import argparse,json,sys,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fluidrender.core import load_config,write_json
from fluidrender.simulator import CloudRenderingSimulator,build_trace
from fluidrender.schedulers import make_scheduler
from fluidrender.metrics import summarize,aggregate,tail_composition
from fluidrender.protocols import replay_cases
from fluidrender.experiments import write_records,write_summary,provenance

p=argparse.ArgumentParser(description='Link, tail, freshness and recovery replay')
p.add_argument('--suite',choices=['network','tail','freshness','recovery','all'],default='all')
p.add_argument('--config',default=str(ROOT/'configs/replay.json'))
p.add_argument('--seeds',type=int,default=10)
p.add_argument('--smoke',action='store_true',help='Short test: 8-second run with stress during seconds 3--5')
p.add_argument('--out',default=str(ROOT/'results/replay'))
a=p.parse_args();base=load_config(a.config);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
start,end=600.,720.
if a.smoke:
    base['simulation'].update(duration_s=8.,warmup_s=0.);start,end=3.,5.
suites=['network','tail','freshness','recovery'] if a.suite=='all' else [a.suite]
for suite in suites:
    rows=[];bins=[]
    for name,cfg,method in replay_cases(base,suite,start,end):
        for seed in range(a.seeds):
            assets,tasks=build_trace(cfg,seed+7)
            sim=CloudRenderingSimulator(cfg,assets,tasks,seed+7)
            records=sim.run(make_scheduler(method,cfg,seed+7))
            cohort=[r for r in records if start*1000 <= r.arrival_ms < end*1000]
            row=summarize(cohort,end-start,sim.state.network.window_mb,sim.state.network.window_store_mb)
            row.update(scenario=name,seed=seed+7)
            if suite=='tail':
                row.update(tail_composition(records,start*1000,end*1000))
            if suite=='recovery':
                clear=end*1000
                outstanding=[r for r in records if r.arrival_ms < clear and r.finish_ms>clear]
                row['outstanding_drain_ms']=max((r.finish_ms-clear for r in outstanding),default=0.)
                row['directory_refresh_ms']=sim.state.recovery_refresh_ms
                post=[r for r in records if clear <= r.arrival_ms < clear+10000]
                row['post_clearance_miss_rate']=sum(r.slo_miss for r in post)/len(post) if post else None
                for i in range(-10,20):
                    left=clear+i*500
                    cohort_bin=[r for r in records if left <= r.arrival_ms < left+500]
                    bins.append(dict(scenario=name,seed=seed+7,seconds_after_clearance=i*.5,
                        offered=len(cohort_bin),miss_rate=sum(r.slo_miss for r in cohort_bin)/len(cohort_bin) if cohort_bin else None))
            rows.append(row)
            write_records(out/suite/name/f'seed{seed+7}.csv',records)
            write_json(out/suite/name/'config.json',cfg)
        print(suite,name,'finished',flush=True)
    write_summary(out/suite/'per_run.csv',rows)
    write_summary(out/suite/'summary.csv',[dict(scenario=name,**aggregate([r for r in rows if r['scenario']==name]))
                                          for name,_,_ in replay_cases(base,suite,start,end)])
    if bins:
        write_summary(out/suite/'recovery_bins.csv',bins)
write_json(out/'provenance.json',provenance(base))
