#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fluidrender.experiments import load_and_override,run_experiment
p=argparse.ArgumentParser()
p.add_argument('--config',default=str(ROOT/'configs/default.json'))
p.add_argument('--seconds',type=float)
p.add_argument('--warmup',type=float)
p.add_argument('--sessions',type=int)
p.add_argument('--seeds',type=int,default=1)
p.add_argument('--methods',default='RR,SAG,A-LALL,K8s-FS')
p.add_argument('--checkpoint',action='append',default=[],help='METHOD=path.npz; may be repeated')
p.add_argument('--out',default=str(ROOT/'results/evaluation'))
a=p.parse_args();cfg=load_and_override(a.config,a.seconds,a.warmup,a.sessions)
for item in a.checkpoint:
    name,path=item.split('=',1);cfg.setdefault('checkpoints',{})[name]=path
print(json.dumps(run_experiment(cfg,a.methods.split(','),a.seeds,a.out),indent=2))
