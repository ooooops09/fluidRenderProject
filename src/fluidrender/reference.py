"""Reduced aggregate-residency/serialization model, separate from link replay.

Cache residency changes at task transitions. Concurrent link sharing is
implemented separately by the backend simulator.
"""
from .simulator import SimulationState, PlacementRecord
from .training import team_reward


class ReferenceSimulator:
    def __init__(self,cfg,assets,tasks,seed):
        self.cfg,self.assets,self.tasks,self.seed=cfg,assets,tasks,seed
        self.state=None

    def run(self,scheduler):
        state=SimulationState(self.cfg,self.assets,self.seed+991)
        self.state=state
        records=[]
        report_every=self.cfg['telemetry']['cache_report_interval_ms']
        last_report=0.
        total_mb=0.
        for task in self.tasks:
            state.now_ms=task.arrival_ms
            state._prediction_cache.clear()
            if state.now_ms-last_report >= report_every:
                for n in state.nodes:
                    state.directory.heartbeat(n.node_id,state.now_ms)
                    state._report(n,full=True,immediate=True)
                last_report=state.now_ms
            drop=self.cfg['telemetry'].get('observation_missing_probability',0.)
            state.directory.observation_missing={n.node_id for n in state.nodes if state.rng.random()<drop}
            nid=scheduler.select(task,state)
            r=PlacementRecord(scheduler.name,self.seed,task.task_id,task.arrival_ms,nid or '',task.deadline_ms,
                              warmup=int(task.arrival_ms<self.cfg['simulation']['warmup_s']*1000),
                              working_set_mb=state.working_set_mb(task))
            if nid is None:
                r.status='rejected';r.slo_miss=1;r.load_skew=state.load_skew();r.finish_ms=state.now_ms
            else:
                n=state.node_by_id[nid]
                sync=0.
                for a in task.working_set:
                    if a not in n.cache.items:
                        src=state._source(a,nid,state.now_ms)
                        sync+=state.network.estimate(src,nid,self.assets[a].size_mb,state.now_ms)
                        r.sync_mb+=self.assets[a].size_mb
                r.sync_ms=sync
                r.load_skew=state.post_load_skew(nid,task)
                r.jain_util=state.jain_util()
                if sync>=task.deadline_ms:
                    r.status='failed';r.slo_miss=1;r.finish_ms=state.now_ms+task.deadline_ms
                else:
                    r.render_ms=state.render_ms(task,n)
                    r.queue_ms,_,r.finish_ms=n.assign(state.now_ms+sync,r.render_ms)
                    r.latency_ms=r.finish_ms-task.arrival_ms
                    r.status='completed';r.slo_miss=int(r.latency_ms>task.deadline_ms)
                    # Aggregate residency transition; no concurrent chunk transport.
                    for a in task.working_set:
                        n.cache.insert(self.assets[a],state.now_ms,pinned_until_ms=r.finish_ms)
                if not r.warmup:
                    total_mb+=r.sync_mb
            r.team_reward=team_reward(r.latency_ms,r.deadline_ms,r.load_skew,completed=r.status=='completed',**self.cfg['reward'])
            records.append(r)
            scheduler.observe(task,r,state)
        state.network.window_mb=total_mb
        state.network.window_store_mb=0.
        return records
