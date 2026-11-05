from __future__ import annotations
import csv
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from .core import default_method_names, load_config, merge_config, write_json
from .metrics import summarize, aggregate
from .schedulers import make_scheduler
from .simulator import CloudRenderingSimulator, build_trace
from .reference import ReferenceSimulator


def write_summary(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def write_records(path, records):
    write_summary(path,[asdict(r) for r in records])


def provenance(cfg):
    paths=list(Path(__file__).parent.glob('*.py'))
    return {'python':sys.version,'platform':platform.platform(),
            'workload_source':cfg['workload'].get('trace_file') or 'generated workload',
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'checkpoint_sha256':{k:hashlib.sha256(Path(v).read_bytes()).hexdigest()
                                 for k,v in cfg.get('checkpoints',{}).items()},
            'results_origin':'simulation'}


def run_experiment(cfg, methods=None, seeds=1, out_dir='results', seed_values=None):
    methods=methods or default_method_names()
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    allrows=[];summary={}
    values=seed_values if seed_values is not None else range(int(cfg.get('seed',7)),int(cfg.get('seed',7))+seeds)
    for seed in values:
        assets,tasks=build_trace(cfg,seed)
        for method in methods:
            scheduler=make_scheduler(method,cfg,seed+17)
            model=getattr(getattr(scheduler,'policy',None),'model','backend')
            engine=ReferenceSimulator if model=='reference' else CloudRenderingSimulator
            sim=engine(cfg,assets,tasks,seed)
            records=sim.run(scheduler)
            row=summarize(records,cfg['simulation']['duration_s']-cfg['simulation']['warmup_s'],
                          sim.state.network.window_mb,sim.state.network.window_store_mb)
            row.update(seed=seed,scheduler=method)
            allrows.append(row)
            write_records(out/f'{method}_seed{seed}_events.csv',records)
    for method in methods:
        summary[method]=aggregate([r for r in allrows if r['scheduler']==method])
    write_summary(out/'per_run.csv',allrows)
    write_summary(out/'summary.csv',[dict(scheduler=k,**v) for k,v in summary.items()])
    write_json(out/'run_config.json',cfg)
    write_json(out/'provenance.json',provenance(cfg))
    return summary


def load_and_override(config_path=None, seconds=None, warmup=None, sessions=None):
    cfg=load_config(config_path)
    overrides={}
    if seconds is not None:
        overrides['simulation']={'duration_s':seconds,'warmup_s':min(cfg['simulation']['warmup_s'],seconds*.15)}
    if warmup is not None:
        overrides.setdefault('simulation',{})['warmup_s']=warmup
    if sessions is not None:
        overrides['workload']={'active_sessions':sessions}
    return merge_config(cfg,overrides)
