"""Event-driven backend with asset transfer, FIFO rendering and telemetry."""
from __future__ import annotations
import hashlib
import heapq
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from .core import (Asset, Task, NodeRuntime, WorkloadGenerator, capacity_weighted_jain,
                   capacity_weighted_load_skew)
from .network import FluidNetwork
from .telemetry import Report, ValidatedDirectory, sign
from .training import team_reward


@dataclass
class PlacementRecord:
    scheduler: str
    seed: int
    task_id: int
    arrival_ms: float
    node_id: str
    deadline_ms: float
    predicted_ms: float | None = None
    latency_ms: float | None = None
    sync_ms: float = 0.
    queue_ms: float = 0.
    render_ms: float = 0.
    sync_mb: float = 0.
    working_set_mb: float = 0.
    load_skew: float = 0.
    jain_util: float = 1.
    slo_miss: int = 0
    inference_ms: float = 0.
    warmup: int = 0
    status: str = 'pending'
    finish_ms: float | None = None
    peer_failures: int = 0
    team_reward: float = 0.


class SimulationState:
    def __init__(self, cfg, assets, seed):
        self.cfg, self.assets, self.seed = cfg, assets, seed
        self.rng = random.Random(seed)
        self.nodes = [NodeRuntime(s, cfg['cluster']['cache_mb_per_node'], cfg['simulation']['util_window_ms'])
                      for s in cfg['cluster']['nodes']]
        self.node_by_id = {n.node_id: n for n in self.nodes}
        self.network = FluidNetwork(cfg, self.nodes)
        # Reproducible simulator-only keys. Production workers require secret provisioned keys.
        self.keys = {n.node_id: hashlib.sha256(('simulation:' + str(seed) + n.node_id).encode()).digest()
                     for n in self.nodes}
        tc = cfg['telemetry']
        self.directory = ValidatedDirectory(self.keys, tc['cache_stale_ms'], tc['heartbeat_timeout_ms'])
        self.now_ms = 0.
        self.events, self.serial = [], 0
        self.report_seq = dict.fromkeys(self.node_by_id, 0)
        self.pins = {n.node_id: {} for n in self.nodes}
        self.jobs, self.records = {}, []
        self._prediction_cache = {}
        self.scheduler = None
        self.replay = cfg.get('controlled_replay', {}).get('enabled', False)
        self.end_ms = cfg['simulation']['duration_s'] * 1000.
        self.recovery_refresh_ms = None
        self.clear_ms = max((e['end_s']*1000 for e in tc.get('delay_events', [])), default=math.inf)
        self._initial_fill()
        for n in self.nodes:
            self.directory.heartbeat(n.node_id, 0.)
            self._report(n, full=True, immediate=True)
        for n in self.nodes:
            self.push(self.rng.uniform(0., tc['cache_report_interval_ms']), 'report_worker', n.node_id)
        if self.replay:
            self.push(1000., 'churn', 1)
        for d in cfg['network'].get('degradations', []) + tc.get('delay_events', []):
            self.push(d['start_s'] * 1000, 'boundary', None)
            self.push(d['end_s'] * 1000, 'boundary', None)

    def _initial_fill(self):
        assets = list(self.assets.values())
        for n in self.nodes:
            order = self.rng.sample(assets, len(assets))
            if self.replay:
                order = order[:self.cfg['controlled_replay'].get('resident_chunks', 350)]
            target = n.cache.capacity_mb * self.cfg['cluster']['initial_cache_fill']
            for a in order:
                if not self.replay and n.cache.used_mb >= target:
                    break
                n.cache.insert(a, 0.)

    def push(self, at, kind, payload):
        self.serial += 1
        priority = 0 if kind in ('chunk', 'finish') else 2 if kind == 'deadline' else 1
        heapq.heappush(self.events, (at, priority, self.serial, kind, payload))

    def _report(self, node, full=False, present=(), removed=(), immediate=False):
        self.report_seq[node.node_id] += 1
        seq = self.report_seq[node.node_id]
        report = Report(node.node_id, 0, seq, self.now_ms, seq, full,
                        tuple(sorted(node.cache.items)) if full else tuple(present), tuple(removed))
        tc = self.cfg['telemetry']
        jitter_seed = hashlib.sha256(f'{self.seed}:{node.node_id}:{self.now_ms:.9f}:{full}'.encode()).digest()
        jitter_rng = random.Random(int.from_bytes(jitter_seed[:8], 'big'))
        delay = 0. if immediate else jitter_rng.uniform(*tc.get('delivery_delay_ms', [2., 8.]))
        delay += 0. if immediate else tc.get('extra_delay_ms', 0.)
        for e in tc.get('delay_events', []):
            if e['start_s'] * 1000 <= self.now_ms < e['end_s'] * 1000:
                delay += e['extra_delay_ms']
        tag = sign(report, self.keys[node.node_id])
        if immediate:
            self.directory.receive(report, tag, self.now_ms)
        else:
            self.push(self.now_ms + delay, 'report', (report, tag))

    def live_nodes(self):
        return [n for n in self.nodes if n.alive]

    def capacities(self):
        return [n.capacity for n in self.nodes]

    def utilizations(self, now_ms=None):
        now = self.now_ms if now_ms is None else now_ms
        return [n.physical_utilization(now) for n in self.nodes]

    def load_skew(self, now_ms=None):
        return capacity_weighted_load_skew(self.utilizations(now_ms), self.capacities())

    def jain_util(self, now_ms=None):
        return capacity_weighted_jain(self.utilizations(now_ms), self.capacities())

    def working_set_mb(self, task):
        return sum(self.assets[a].size_mb for a in task.working_set)

    def render_ms(self, task, node, realized=False):
        # Task demand is part of the paired exogenous trace, not policy-specific noise.
        return task.demand_ms / node.rate

    def _source(self, chunk, target, now):
        choices = [None] + [n.node_id for n in self.nodes if n.alive and n.node_id != target and
                           self.directory.observed_has(n.node_id, chunk, now)]
        return min(choices, key=lambda src: self.network.estimate(src, target, self.assets[chunk].size_mb, now))

    def predicted_components(self, task, node_id, now_ms=None):
        now = self.now_ms if now_ms is None else now_ms
        key = (task.task_id, node_id, now)
        if key in self._prediction_cache:
            return self._prediction_cache[key]
        n = self.node_by_id[node_id]
        missing = [a for a in task.working_set if not self.directory.observed_has(node_id, a, now)]
        sync = sum(self.network.estimate(self._source(a, node_id, now), node_id, self.assets[a].size_mb, now)
                   for a in missing)
        queue, render = n.queue_ms(now), self.render_ms(task, n)
        result = dict(sync_ms=sync, queue_ms=queue, render_ms=render, latency_ms=sync+queue+render,
                      missing_mb=sum(self.assets[a].size_mb for a in missing), working_set_mb=self.working_set_mb(task))
        self._prediction_cache[key] = result
        return result

    def post_load_skew(self, node_id, task, service_ms=None, now_ms=None):
        now = self.now_ms if now_ms is None else now_ms
        utils = [n.projected_utilization(now, self.render_ms(task, n)) if n.node_id == node_id
                 else n.physical_utilization(now) for n in self.nodes]
        return capacity_weighted_load_skew(utils, self.capacities())

    def mask(self, task):
        return [n.alive and self.predicted_components(task, n.node_id)['latency_ms'] <= task.deadline_ms
                for n in self.nodes]

    def _pin(self, node_id, chunk):
        self.pins[node_id][chunk] = self.pins[node_id].get(chunk, 0) + 1
        entry = self.node_by_id[node_id].cache.items.get(chunk)
        if entry:
            entry.pinned_until_ms = math.inf

    def _release(self, job):
        nid = job['record'].node_id
        for a in job['task'].working_set:
            if a not in job['pinned']:
                continue
            self.pins[nid][a] -= 1
            if self.pins[nid][a] == 0:
                del self.pins[nid][a]
                entry = self.node_by_id[nid].cache.items.get(a)
                if entry:
                    entry.pinned_until_ms = self.now_ms

    def _arrive(self, task):
        self._prediction_cache.clear()
        start = time.perf_counter()
        selected = self.scheduler.select(task, self)
        infer = (time.perf_counter() - start) * 1000
        record = PlacementRecord(self.scheduler.name, self.seed, task.task_id, task.arrival_ms,
                                 selected or '', task.deadline_ms, working_set_mb=self.working_set_mb(task),
                                 warmup=int(task.arrival_ms < self.cfg['simulation']['warmup_s'] * 1000),
                                 inference_ms=infer)
        job = dict(task=task, record=record, pending=[], outstanding=0, pinned=set(), ready=False)
        self.jobs[task.task_id] = job
        self.records.append(record)
        if selected is None:
            record.status = 'rejected'
            self._terminal(job)
            return
        if selected not in self.node_by_id or not self.node_by_id[selected].alive:
            raise ValueError('Scheduler selected an invalid worker')
        if not self.replay and not self.mask(task)[list(self.node_by_id).index(selected)]:
            raise ValueError('Scheduler violated the common feasibility mask')
        record.predicted_ms = self.predicted_components(task, selected)['latency_ms']
        record.load_skew = self.post_load_skew(selected, task)
        record.jain_util = self.jain_util()
        node = self.node_by_id[selected]
        for a in task.working_set:
            if node.cache.contains(a, self.now_ms):
                self._pin(selected, a)
                job['pinned'].add(a)
            else:
                job['pending'].append(a)
        self.push(task.arrival_ms + task.deadline_ms, 'deadline', task.task_id)
        self._launch(job)

    def _launch(self, job):
        if job['record'].status != 'pending':
            return
        limit = self.cfg['network'].get('max_fetches_per_task', 4)
        while job['pending'] and job['outstanding'] < limit:
            a = job['pending'].pop(0)
            job['outstanding'] += 1
            nid = job['record'].node_id
            src = self._source(a, nid, self.now_ms)
            if src is not None and a not in self.node_by_id[src].cache.items:
                job['record'].peer_failures += 1
                self.directory.invalidate(src, a)
                self.push(self.now_ms + self.cfg['telemetry'].get('failed_lookup_ms', 5.),
                          'fallback', (job['task'].task_id, a))
            else:
                self.network.add(job['task'].task_id, self.assets[a], src, nid)
        if not job['pending'] and job['outstanding'] == 0 and not job['ready']:
            job['ready'] = True
            r, task = job['record'], job['task']
            r.sync_ms = self.now_ms - task.arrival_ms
            r.render_ms = self.render_ms(task, self.node_by_id[r.node_id])
            r.queue_ms, _, finish = self.node_by_id[r.node_id].assign(self.now_ms, r.render_ms)
            self.push(finish, 'finish', task.task_id)

    def _chunk(self, flow):
        job = self.jobs[flow.task_id]
        if job['record'].status != 'pending':
            return
        node = self.node_by_id[flow.destination]
        a = self.assets[flow.asset_id]
        if not self.replay:
            evicted = node.cache.insert(a, self.now_ms, math.inf)
            if self.cfg['telemetry'].get('event_reports', True):
                self._report(node, present=(a.asset_id,) if a.asset_id in node.cache.items else (), removed=evicted)
        self._pin(flow.destination, flow.asset_id)
        job['pinned'].add(flow.asset_id)
        job['outstanding'] -= 1
        self._launch(job)

    def _terminal(self, job):
        r, task = job['record'], job['task']
        r.finish_ms = self.now_ms
        if r.status == 'completed':
            r.latency_ms = self.now_ms - task.arrival_ms
        r.slo_miss = int(r.status != 'completed' or r.latency_ms > task.deadline_ms + 1e-9)
        if r.status == 'rejected':
            r.load_skew, r.jain_util = self.load_skew(), self.jain_util()
        reward_cfg = self.cfg.get('reward', {})
        r.team_reward = team_reward(r.latency_ms, task.deadline_ms, r.load_skew, completed=r.status == 'completed', **reward_cfg)
        self._release(job)
        self.scheduler.observe(task, r, self)

    def _event(self, kind, payload):
        self._prediction_cache.clear()
        if kind == 'arrival':
            self._arrive(payload)
        elif kind == 'report':
            self.directory.receive(*payload, self.now_ms)
            if self.recovery_refresh_ms is None and all(
                    self.directory.last_full_generated.get(n.node_id, -math.inf) >= self.clear_ms for n in self.nodes):
                self.recovery_refresh_ms = self.now_ms - self.clear_ms
        elif kind == 'report_worker':
            n = self.node_by_id[payload]
            if n.alive:
                self.directory.heartbeat(n.node_id, self.now_ms)
                self._report(n, full=True)
            nxt = self.now_ms + self.cfg['telemetry']['cache_report_interval_ms']
            if nxt <= self.end_ms:
                self.push(nxt, 'report_worker', payload)
        elif kind == 'churn':
            # Exogenous, seed/time-addressed residency changes independent of placements.
            count = self.cfg['controlled_replay'].get('replacements_per_second', 17)
            for idx, n in enumerate(self.nodes):
                rng = random.Random(self.seed * 1000003 + payload * 1009 + idx)
                candidates = sorted(n.cache.items)
                victims = rng.sample(candidates, min(count, len(candidates)))
                fresh = rng.sample(sorted(set(self.assets) - set(n.cache.items)), len(victims))
                for a in victims:
                    n.cache.used_mb -= n.cache.items.pop(a).size_mb
                for a in fresh:
                    n.cache.insert(self.assets[a], self.now_ms)
            if self.now_ms + 1000 <= self.end_ms:
                self.push(self.now_ms + 1000, 'churn', payload + 1)
        elif kind == 'fallback':
            tid, a = payload
            job = self.jobs[tid]
            if job['record'].status == 'pending' and self.now_ms < job['task'].arrival_ms + job['task'].deadline_ms:
                self.network.add(tid, self.assets[a], None, job['record'].node_id)
        elif kind == 'chunk':
            self._chunk(payload)
        elif kind == 'finish':
            job = self.jobs[payload]
            if job['record'].status == 'pending':
                job['record'].status = 'completed'
                self._terminal(job)
        elif kind == 'deadline':
            job = self.jobs[payload]
            if job['record'].status == 'pending' and not job['ready']:
                job['record'].status = 'failed'
                job['record'].sync_ms = self.now_ms - job['task'].arrival_ms
                for f in self.network.cancel(payload):
                    job['record'].sync_mb += f.sent_mb
                self._terminal(job)

    def run(self, tasks, scheduler):
        self.scheduler = scheduler
        for task in tasks:
            self.push(task.arrival_ms, 'arrival', task)
        while self.events or self.network.flows:
            t_event = self.events[0][0] if self.events else math.inf
            t_flow = self.network.next_completion(self.now_ms)
            next_time = min(t_event, t_flow)
            if math.isinf(next_time):
                raise RuntimeError('No progress possible with active flows')
            completed = self.network.advance(self.now_ms, next_time)
            self.now_ms = next_time
            for flow in completed:
                self.jobs[flow.task_id]['record'].sync_mb += flow.sent_mb
                self.push(next_time + flow.delay_ms, 'chunk', flow)
            while self.events and self.events[0][0] <= self.now_ms + 1e-9:
                _, _, _, kind, payload = heapq.heappop(self.events)
                self._event(kind, payload)
        if any(r.status == 'pending' for r in self.records):
            raise RuntimeError('Unresolved requests after drain')
        return self.records


class CloudRenderingSimulator:
    def __init__(self, cfg, assets, tasks, seed):
        self.cfg, self.assets, self.tasks, self.seed = cfg, assets, tasks, seed
        self.state = None

    def run(self, scheduler):
        self.state = SimulationState(self.cfg, self.assets, self.seed + 991)
        return self.state.run(self.tasks, scheduler)


def build_trace(cfg, seed):
    trace_path = cfg.get('workload', {}).get('trace_file')
    if trace_path:
        path = str(trace_path).format(seed=seed)
        data = json.loads(Path(path).read_text())
        assets = {a['asset_id']: Asset(**a) for a in data['assets']}
        tasks = [Task(**{**t, 'working_set': tuple(t['working_set'])}) for t in data['tasks']
                 if t['arrival_ms'] < cfg['simulation']['duration_s'] * 1000]
        if any(t.priority != 1 for t in tasks):
            raise ValueError('Experiments require unit task priorities')
        if len({t.task_id for t in tasks}) != len(tasks):
            raise ValueError('Duplicate task IDs in trace')
        if any(t.arrival_ms < 0 or t.deadline_ms <= 0 or t.demand_ms < 0 or any(a not in assets for a in t.working_set) for t in tasks):
            raise ValueError('Invalid task or missing asset in trace')
        return assets, sorted(tasks, key=lambda t: t.arrival_ms)
    if cfg.get('controlled_replay', {}).get('enabled'):
        rng = random.Random(seed)
        assets = {f'chunk{i}': Asset(f'chunk{i}', 32., i // 125, None, 'replay') for i in range(1000)}
        tasks, now = [], 0.
        while True:
            now += rng.expovariate(.020)
            if now >= cfg['simulation']['duration_s'] * 1000:
                break
            app = rng.choices(range(8), weights=[(i+1)**(-.8) for i in range(8)])[0]
            chunks = tuple(f'chunk{i}' for i in rng.sample(range(app*125, (app+1)*125), 4))
            demand = rng.uniform(*cfg['controlled_replay'].get('render_demand_ms', [12., 24.]))
            deadline = rng.choices([50., 80., 100.], [.1, .7, .2])[0]
            tasks.append(Task(len(tasks), now, len(tasks)%100, app, 0, chunks, demand, deadline, 1., 'replay'))
        return assets, tasks
    generator = WorkloadGenerator(cfg, seed)
    tasks = generator.generate_tasks(cfg['simulation']['duration_s'] * 1000.)
    for t in tasks:
        t.priority = 1.
        t.demand_ms *= cfg['workload'].get('render_demand_scale', 1.)
    return generator.assets, tasks
