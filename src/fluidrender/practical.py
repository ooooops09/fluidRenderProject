"""Cryptographic verification and online placement benchmarks."""
import hashlib,hmac,os,platform,sys,time
from pathlib import Path
from .core import percentile,write_json
from .experiments import write_summary
from .learning import Policy,encode
from .simulator import SimulationState,build_trace


def run_crypto(out,quick=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    key=os.urandom(32);rows=[]
    for operation,sizes,count in [('HMAC-SHA256',[389,2426,17366],100 if quick else 50000),
                                  ('SHA-256',[4*2**20,16*2**20,64*2**20],3 if quick else 500)]:
        for size in sizes:
            payload=os.urandom(size)
            if operation=='HMAC-SHA256':
                tag=hmac.digest(key,payload,'sha256')
                def fn():
                    return hmac.compare_digest(hmac.digest(key,payload,'sha256'),tag)
            else:
                def fn():
                    return hashlib.sha256(payload).digest()
            for _ in range(3 if quick else 20):fn()
            times=[]
            for _ in range(count):
                start=time.perf_counter_ns();fn();times.append((time.perf_counter_ns()-start)/1e6)
            row=dict(operation=operation,input_bytes=size,calls=count,p50_ms=percentile(times,.5),
                     p95_ms=percentile(times,.95),p99_ms=percentile(times,.99))
            rows.append(row)
            write_summary(out/f'{operation}_{size}_samples.csv',[{'elapsed_ms':t} for t in times])
    write_summary(out/'crypto.csv',rows)
    write_json(out/'environment.json',{'python':sys.version,'platform':platform.platform(),'quick':quick,
              'excludes':'allocation, parsing, I/O, directory processing and network transfer'})
    return rows


def run_serving(cfg,checkpoint,out,samples=120):
    import numpy as np
    policy=Policy.load(checkpoint)
    if policy.nodes != len(cfg['cluster']['nodes']):
        raise ValueError('Use a checkpoint trained for this worker count')
    assets,tasks=build_trace(cfg,cfg.get('seed',7)+30000)
    state=SimulationState(cfg,assets,cfg.get('seed',7)+90000)
    measurements=[]
    # Replay actual global-state feature construction and actor matrix operations.
    # Snapshots here use generated initial state; an external trace/checkpoint can be supplied.
    for task in tasks[:samples]:
        state.now_ms=task.arrival_ms;state._prediction_cache.clear()
        t0=time.perf_counter_ns();x=encode(task,state);t1=time.perf_counter_ns()
        mask=state.mask(task);t2=time.perf_counter_ns()
        logits=[a.forward(x) for a in policy.actors];t3=time.perf_counter_ns()
        z=np.sum(logits,axis=0);z=np.where(mask,z,-np.inf)
        action=int(np.argmax(z)) if any(mask) else None;t4=time.perf_counter_ns()
        response={'task_id':task.task_id,'node_id':state.nodes[action].node_id if action is not None else None}
        t5=time.perf_counter_ns()
        measurements.append(dict(feature_assembly_ms=(t1-t0)/1e6,action_mask_ms=(t2-t1)/1e6,
          actor_inference_ms=(t3-t2)/1e6,fusion_ms=(t4-t3)/1e6,response_ms=(t5-t4)/1e6,total_ms=(t5-t0)/1e6))
    if not measurements:raise ValueError('No requests in the serving trace')
    rows=[dict(component=k,p50_ms=percentile([r[k] for r in measurements],.5),
          p95_ms=percentile([r[k] for r in measurements],.95),p99_ms=percentile([r[k] for r in measurements],.99)) for k in measurements[0]]
    write_summary(Path(out)/'serving_samples.csv',measurements);write_summary(Path(out)/'serving.csv',rows)
    return rows
