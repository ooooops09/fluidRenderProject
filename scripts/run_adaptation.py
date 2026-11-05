#!/usr/bin/env python3
"""Independent adaptation per shift with optimizer state, disjoint train/test seeds."""
import argparse,copy,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fluidrender.core import load_config,merge_config,write_json
from fluidrender.learning import Policy,train
from fluidrender.protocols import generalization_configs
from fluidrender.experiments import run_experiment,write_summary

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True)
p.add_argument('--config',default=str(ROOT/'configs/default.json'))
p.add_argument('--steps',type=int)
p.add_argument('--test-seeds',default='2001,2002,2003,2004,2005,2006,2007,2008,2009,2010')
p.add_argument('--smoke',action='store_true')
p.add_argument('--out',default=str(ROOT/'results/adaptation'))
a=p.parse_args();base=load_config(a.config);original=Policy.load(a.checkpoint);out=Path(a.out)
if original.model=='reference':
    reduced_net={k:base['network'][k]*.3 for k in ('worker_access_gbps','rack_uplink_gbps','object_store_gbps')}
    cases=[('network',merge_config(base,{'network':reduced_net})),
           ('telemetry',merge_config(base,{'telemetry':{'observation_missing_probability':.3}})),
           ('combined',merge_config(base,{'network':reduced_net,'telemetry':{'observation_missing_probability':.3}}))]
else:cases=generalization_configs(base)
steps=a.steps if a.steps is not None else 32768 if original.model=='reference' else 400000
if a.smoke:steps=16
rows=[]
for name,cfg in cases:
    ev=copy.deepcopy(cfg)
    if a.smoke:ev['simulation'].update(duration_s=2.,warmup_s=0.)
    ev['checkpoints']={original.method:a.checkpoint}
    seeds=[int(s) for s in a.test_seeds.split(',')]
    before=run_experiment(ev,[original.method],seed_values=seeds,out_dir=out/name/'frozen')[original.method]
    pol=Policy.load(a.checkpoint)  # each shift starts from identical actor and optimizer state
    pol.seed+=900000  # reserve a separate workload stream for adaptation
    train_cfg=copy.deepcopy(cfg);train_cfg['simulation'].update(duration_s=20.,warmup_s=0.)
    start_interactions=pol.interactions;checkpoint=out/name/'adapted.npz'
    cpu=time.process_time();wall=time.perf_counter()
    train(pol,train_cfg,pol.interactions+steps,pretrain_total=0,batch_size=16 if a.smoke else 256)
    cpu=time.process_time()-cpu;wall=time.perf_counter()-wall
    pol.save(checkpoint,train_cfg)
    ev['checkpoints']={pol.method:str(checkpoint)}
    after=run_experiment(ev,[pol.method],seed_values=seeds,out_dir=out/name/'adapted')[pol.method]
    for mode,result in [('frozen',before),('adapted',after)]:
        rows.append(dict(scenario=name,mode=mode,training_seed=original.seed,**result))
    write_json(out/name/'adaptation_cost.json',dict(interactions=pol.interactions-start_interactions,cpu_s=cpu,wall_s=wall,
               original_training_seed=original.seed,model=original.model,excludes='evaluation and checkpoint serialization'))
    print(name,'finished',flush=True)
write_summary(out/'summary.csv',rows)
