#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fluidrender.core import load_config
from fluidrender.learning import Policy,train,LEARNED
from fluidrender.experiments import run_experiment,write_summary

p=argparse.ArgumentParser(description='Train actor policies; the interaction budget includes pre-training.')
p.add_argument('--config',default=str(ROOT/'configs/default.json'))
p.add_argument('--method',choices=LEARNED,default='FluidRender')
p.add_argument('--model',choices=['backend','reference'],default='backend')
p.add_argument('--steps',type=int,default=3600000)
p.add_argument('--pretrain-total',type=int,default=2000000)
p.add_argument('--batch',type=int,default=256)
p.add_argument('--rollout-seconds',type=float,default=20.)
p.add_argument('--seed',type=int,default=7)
p.add_argument('--resume')
p.add_argument('--out',default=str(ROOT/'checkpoints/FluidRender.npz'))
p.add_argument('--checkpoint-every',type=int,default=200000)
p.add_argument('--validation-seeds',default='1001,1002,1003,1004,1005')
p.add_argument('--validation-seconds',type=float,default=1800.)
p.add_argument('--actor-lr',type=float,default=1e-4)
p.add_argument('--critic-lr',type=float,default=3e-4)
p.add_argument('--entropy',type=float,default=1e-3)
a=p.parse_args()
if a.steps<=0 or a.pretrain_total<0 or (not a.resume and a.method=='FluidRender' and a.pretrain_total>=a.steps):
    p.error('Positive budget and a joint stage after pre-training are required')
cfg=load_config(a.config)
cfg['simulation'].update(duration_s=a.rollout_seconds,warmup_s=0.)
policy=Policy.load(a.resume) if a.resume else Policy(a.method,len(cfg['cluster']['nodes']),a.seed,
           hidden=32 if a.model=='reference' else 64,actor_lr=a.actor_lr,critic_lr=a.critic_lr,entropy=a.entropy,model=a.model)
rows=[]
def validate(pol):
    pol.save(a.out,cfg)
    ev=load_config(a.config);ev['simulation'].update(duration_s=a.validation_seconds,warmup_s=min(300.,a.validation_seconds*.15))
    ev['checkpoints']={pol.method:a.out}
    result=run_experiment(ev,[pol.method],out_dir=Path(a.out).parent/f'validation_{pol.interactions}',
                          seed_values=[int(s) for s in a.validation_seeds.split(',')])[pol.method]
    rows.append(dict(interactions=pol.interactions,**result))
    write_summary(Path(a.out).with_suffix('.validation.csv'),rows)
    print(json.dumps(rows[-1]),flush=True)
train(policy,cfg,a.steps,a.pretrain_total,a.batch,a.out,a.checkpoint_every,
      validate if a.checkpoint_every else None)
policy.save(a.out,cfg)
print(json.dumps({'checkpoint':a.out,'interactions':policy.interactions,'training_wall_s':policy.elapsed_s,
                  'actor_parameters':sum(n.parameters for n in policy.actors),'critic_parameters':sum(n.parameters for n in policy.critics)}))
