#!/usr/bin/env python3
"""Independent searches; final-budget validation selection, then fresh train/test seeds."""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fluidrender.core import load_config,write_json
from fluidrender.learning import Policy,train
from fluidrender.experiments import run_experiment,write_summary,provenance
from fluidrender.metrics import hierarchical_aggregate

p=argparse.ArgumentParser()
p.add_argument('--config',default=str(ROOT/'configs/default.json'))
p.add_argument('--model',choices=['backend','reference'],default='reference')
p.add_argument('--methods',default='FluidRender,1A-Scalar,1A-MO,JointFromScratch')
p.add_argument('--smoke',action='store_true')
p.add_argument('--out',default=str(ROOT/'results/matched_tuning'))
a=p.parse_args();base=load_config(a.config);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
protocol=base['training'];budget=protocol['reference_budget'] if a.model=='reference' else protocol['default_budget']
pre=protocol['reference_pretrain_total'] if a.model=='reference' else 2000000
n_trials=12;tr_seeds=protocol['search_training_seeds'];v_seeds=protocol['validation_seeds'];final_seeds=protocol['final_training_seeds'];test_seeds=protocol['test_seeds']
eval_cfg=copy.deepcopy(base);train_cfg=copy.deepcopy(base);train_cfg['simulation'].update(duration_s=20.,warmup_s=0.)
if a.smoke:
    budget,pre,n_trials=48,32,2;tr_seeds=[11];v_seeds=[1001];final_seeds=[201];test_seeds=[2001]
    eval_cfg['simulation'].update(duration_s=2.,warmup_s=0.)
write_json(out/'protocol.json',dict(model=a.model,budget=budget,pretrain_total=pre,search_trials=n_trials,
    search_training_seeds=tr_seeds,validation_seeds=v_seeds,final_training_seeds=final_seeds,test_seeds=test_seeds,smoke=a.smoke))
all_final=[];all_search=[]
for mi,method in enumerate(a.methods.split(',')):
    rng=np.random.default_rng(9100+mi)
    candidates=[dict(actor_lr=float(10**rng.uniform(-5,-3)),critic_lr=float(10**rng.uniform(-5,-3)),entropy=float(10**rng.uniform(-4,-2))) for _ in range(n_trials)]
    scores=[]
    for i,hp in enumerate(candidates):
        costs=[]
        for seed in tr_seeds:
            target=out/method/f'trial{i}_seed{seed}.npz'
            pol=Policy(method,len(base['cluster']['nodes']),seed,hidden=32 if a.model=='reference' else 64,model=a.model,**hp)
            train(pol,train_cfg,budget,pre,batch_size=16 if a.smoke else 256,checkpoint=target)
            cfg=copy.deepcopy(eval_cfg);cfg['checkpoints']={method:str(target)}
            summary=run_experiment(cfg,[method],seed_values=v_seeds,out_dir=out/method/f'trial{i}_seed{seed}_validation')[method]
            costs.append(summary['validation_cost'])
        cost=float(np.mean(costs));scores.append(cost)
        all_search.append(dict(method=method,trial=i,validation_cost=cost,**hp))
        write_summary(out/'search.csv',all_search)
        print(method,'trial',i,'cost',cost,flush=True)
    selected=int(np.argmin(scores));hp=candidates[selected]
    write_json(out/method/'selected.json',dict(trial=selected,hyperparameters=hp,selection='lowest final-budget mean validation cost'))
    for seed in final_seeds:
        pol=Policy(method,len(base['cluster']['nodes']),seed,hidden=32 if a.model=='reference' else 64,model=a.model,**hp)
        target=out/method/f'final_seed{seed}.npz'
        train(pol,train_cfg,budget,pre,batch_size=16 if a.smoke else 256,checkpoint=target)
        for factor in ([1,2] if method=='JointFromScratch' else [1]):
            if factor==2:
                train(pol,train_cfg,2*budget,pre,batch_size=16 if a.smoke else 256,checkpoint=target)
            checkpoint=out/method/f'final_seed{seed}_{factor}B.npz';pol.save(checkpoint,train_cfg)
            cfg=copy.deepcopy(eval_cfg);cfg['checkpoints']={method:str(checkpoint)}
            test_out=out/method/f'final_seed{seed}_{factor}B_test'
            run_experiment(cfg,[method],seed_values=test_seeds,out_dir=test_out)
            import csv
            for row in csv.DictReader((test_out/'per_run.csv').open()):
                converted={k:float(v) if v else None for k,v in row.items() if k not in ('scheduler',)}
                converted.update(method=method+('-2B' if factor==2 else ''),training_seed=seed)
                all_final.append(converted)
    write_summary(out/'final_per_run.csv',all_final)
summary=[]
for method in dict.fromkeys(r['method'] for r in all_final):
    rows=[{k:v for k,v in r.items() if k!='method'} for r in all_final if r['method']==method]
    summary.append(dict(method=method,**hierarchical_aggregate(rows)))
write_summary(out/'final_summary.csv',summary)
write_json(out/'provenance.json',provenance(base))
