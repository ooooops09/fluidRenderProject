from __future__ import annotations

import math
import random
from collections import deque
from typing import Dict, List, Sequence

from .core import minmax


class Scheduler:
    name = "base"

    def __init__(self, cfg: dict, seed: int = 0):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.deadline_relaxation = 1.0

    def candidate_nodes(self, task, state, enforce_deadline: bool = True):
        live = state.live_nodes()
        feasible = []
        for node in live:
            comp = state.predicted_components(task, node.node_id, task.arrival_ms)
            if comp["latency_ms"] > task.deadline_ms:
                continue
            feasible.append(node)
        return feasible

    def observe(self, task, record, state) -> None:
        return None

    def select(self, task, state) -> str:
        raise NotImplementedError


class FifoScheduler(Scheduler):
    name = "FIFO"

    def select(self, task, state) -> str:
        nodes = self.candidate_nodes(task, state)
        return nodes[0].node_id if nodes else None


class SptScheduler(Scheduler):
    name = "SPT"

    def select(self, task, state) -> str:
        nodes = self.candidate_nodes(task, state, enforce_deadline=False)
        if not nodes:
            return None
        return min(nodes, key=lambda n: n.queue_ms(task.arrival_ms) + state.render_ms(task, n)).node_id


class RoundRobinScheduler(Scheduler):
    name = "RR"

    def __init__(self, cfg: dict, seed: int = 0):
        super().__init__(cfg, seed)
        self.next_idx = 0

    def select(self, task, state) -> str:
        nodes = self.candidate_nodes(task, state, enforce_deadline=False)
        if not nodes:
            return None
        node_ids = [n.node_id for n in nodes]
        all_ids = [n.node_id for n in state.nodes]
        for _ in all_ids:
            node_id = all_ids[self.next_idx % len(all_ids)]
            self.next_idx += 1
            if node_id in node_ids:
                return node_id
        return nodes[0].node_id


class SagScheduler(Scheduler):
    name = "SAG"

    def select(self, task, state) -> str:
        nodes = self.candidate_nodes(task, state, enforce_deadline=True)
        if not nodes:
            return None
        comps = [state.predicted_components(task, n.node_id, task.arrival_ms) for n in nodes]
        lat = minmax([c["latency_ms"] for c in comps])
        miss = minmax([c["missing_mb"] / max(1.0, c["working_set_mb"]) for c in comps])
        queue = minmax([c["queue_ms"] for c in comps])
        skew = minmax([state.post_load_skew(n.node_id, task, c["sync_ms"] + c["render_ms"], task.arrival_ms) for n, c in zip(nodes, comps)])
        best = None
        for idx, node in enumerate(nodes):
            score = 0.24 * lat[idx] + 0.50 * miss[idx] + 0.16 * queue[idx] + 0.10 * skew[idx]
            if best is None or score < best[0]:
                best = (score, node.node_id)
        return best[1]


class AdaptiveLallScheduler(Scheduler):
    name = "A-LALL"

    def __init__(self, cfg: dict, seed: int = 0):
        super().__init__(cfg, seed)
        params = cfg.get("scheduler", {}).get("alall", {})
        self.load_weight = float(params.get("initial_load_weight", 0.58))
        self.min_weight = float(params.get("min_load_weight", 0.25))
        self.max_weight = float(params.get("max_load_weight", 1.35))
        self.adapt_up = float(params.get("adapt_up", 1.035))
        self.adapt_down = float(params.get("adapt_down", 0.996))
        self.history = deque(maxlen=int(params.get("window", 200)))

    def select(self, task, state) -> str:
        nodes = self.candidate_nodes(task, state, enforce_deadline=True)
        if not nodes:
            return None
        best = None
        for node in nodes:
            c = state.predicted_components(task, node.node_id, task.arrival_ms)
            locality = 1.0 - c["missing_mb"] / max(1.0, c["working_set_mb"])
            skew = state.post_load_skew(node.node_id, task, c["sync_ms"] + c["render_ms"], task.arrival_ms)
            score = c["latency_ms"] / task.deadline_ms + self.load_weight * 18.0 * skew - 0.10 * locality
            if best is None or score < best[0]:
                best = (score, node.node_id)
        return best[1]

    def observe(self, task, record, state) -> None:
        self.history.append((record.slo_miss, record.load_skew))
        if len(self.history) < 16:
            return
        miss_rate = sum(x[0] for x in self.history) / len(self.history)
        avg_skew = sum(x[1] for x in self.history) / len(self.history)
        if miss_rate > 0.05 or avg_skew > 0.018:
            self.load_weight = min(self.max_weight, self.load_weight * self.adapt_up)
        else:
            self.load_weight = max(self.min_weight, self.load_weight * self.adapt_down)


class K8sFilterScoreScheduler(Scheduler):
    name = "K8s-FS"

    def select(self, task, state) -> str:
        nodes = self.candidate_nodes(task, state, enforce_deadline=True)
        if not nodes:
            return None
        comps = [state.predicted_components(task, n.node_id, task.arrival_ms) for n in nodes]
        util = minmax([n.utilization(task.arrival_ms) for n in nodes])
        locality = [1.0 - c["missing_mb"] / max(1.0, c["working_set_mb"]) for c in comps]
        net = minmax([c["sync_ms"] for c in comps])
        deadline = [max(0.0, task.deadline_ms - c["latency_ms"]) / task.deadline_ms for c in comps]
        render_fit = minmax([state.render_ms(task, n) for n in nodes])
        best = None
        for idx, node in enumerate(nodes):
            score = (
                0.06 * (1.0 - util[idx])
                + 0.55 * locality[idx]
                + 0.20 * (1.0 - net[idx])
                + 0.09 * deadline[idx]
                + 0.10 * (1.0 - render_fit[idx])
            )
            if best is None or score > best[0]:
                best = (score, node.node_id)
        return best[1]


class TwoStageGeneticScheduler(Scheduler):
    name = "2SGA-Proxy"

    def __init__(self, cfg: dict, seed: int = 0):
        super().__init__(cfg, seed)
        params = cfg.get("scheduler", {}).get("search", {})
        self.population = int(params.get("population", 32))
        self.iterations = int(params.get("iterations", 24))

    def select(self, task, state) -> str:
        nodes = self.candidate_nodes(task, state, enforce_deadline=True)
        if not nodes:
            return None
        if len(nodes) == 1:
            return nodes[0].node_id
        cost_by_node = {}
        for node in nodes:
            c = state.predicted_components(task, node.node_id, task.arrival_ms)
            skew = state.post_load_skew(node.node_id, task, c["sync_ms"] + c["render_ms"], task.arrival_ms)
            cost_by_node[node.node_id] = c["latency_ms"] + 260.0 * skew + 0.006 * c["missing_mb"]
        candidates = [self.rng.choice(nodes).node_id for _ in range(self.population)]
        candidates.extend(n.node_id for n in nodes)
        best = None
        for _ in range(self.iterations):
            scored = []
            for node_id in candidates:
                scored.append((cost_by_node[node_id], node_id))
            scored.sort()
            if best is None or scored[0][0] < best[0]:
                best = scored[0]
            elites = [node_id for _, node_id in scored[: max(2, len(scored) // 4)]]
            candidates = list(elites)
            while len(candidates) < self.population:
                base = self.rng.choice(elites)
                if self.rng.random() < 0.35:
                    base = self.rng.choice(nodes).node_id
                candidates.append(base)
        return best[1]


REGISTRY = {
    "FIFO": FifoScheduler, "SPT": SptScheduler, "RR": RoundRobinScheduler,
    "SAG": SagScheduler, "A-LALL": AdaptiveLallScheduler, "K8s-FS": K8sFilterScoreScheduler,
    "2SGA-Proxy": TwoStageGeneticScheduler,
}


class ReplayScheduler(Scheduler):
    name = "RR"
    def __init__(self, cfg, seed=0, hotspot=False):
        super().__init__(cfg, seed)
        self.i = 0
        self.hotspot = hotspot
    def select(self, task, state):
        i = self.i
        self.i += 1
        if self.hotspot:
            return state.nodes[0 if i % 5 < 4 else 1+(i//5)%(len(state.nodes)-1)].node_id
        return state.nodes[i % len(state.nodes)].node_id


def make_scheduler(name, cfg, seed=0):
    from .learning import LEARNED, Policy, LearnedScheduler
    if name in LEARNED:
        path = cfg.get('checkpoints', {}).get(name)
        if not path:
            raise ValueError(f'{name} requires a trained checkpoint. Run scripts/train.py first and set checkpoints.{name}.')
        policy = Policy.load(path)
        if policy.method != name or policy.nodes != len(cfg['cluster']['nodes']):
            raise ValueError('Checkpoint method or worker count does not match the experiment')
        return LearnedScheduler(policy, fallback=AdaptiveLallScheduler(cfg,seed))
    if name in ('MADDPG','DANIEL','2SGA'):
        raise NotImplementedError(name + ' is not implemented. See README.md for supported baselines.')
    if cfg.get('controlled_replay',{}).get('enabled') and name in ('RR','Hotspot'):
        return ReplayScheduler(cfg,seed,hotspot=name == 'Hotspot')
    if name not in REGISTRY:
        raise KeyError('Unknown scheduler: '+name)
    return REGISTRY[name](cfg,seed)
