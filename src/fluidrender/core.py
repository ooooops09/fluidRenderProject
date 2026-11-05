from __future__ import annotations

import copy
import heapq
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def load_config(path: Optional[str] = None) -> dict:
    if path is None:
        path = str(Path(__file__).resolve().parents[2] / "configs" / "default.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_config(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_config(out[key], value)
        else:
            out[key] = value
    return out


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def weighted_choice(rng: random.Random, items: Sequence, weights: Sequence[float]):
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for item, weight in zip(items, weights):
        acc += weight
        if r <= acc:
            return item
    return items[-1]


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def minmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def capacity_weighted_load_skew(util: Sequence[float], caps: Sequence[float]) -> float:
    total_cap = sum(caps)
    if total_cap <= 0:
        return 0.0
    avg = sum(c * p for c, p in zip(caps, util)) / total_cap
    return sum(c * (p - avg) ** 2 for c, p in zip(caps, util)) / total_cap


def capacity_weighted_jain(util: Sequence[float], caps: Sequence[float]) -> float:
    numerator = sum(c * p for c, p in zip(caps, util)) ** 2
    denominator = sum(caps) * sum(c * p * p for c, p in zip(caps, util))
    if denominator <= 1e-12:
        return 1.0
    return max(0.0, min(1.0, numerator / denominator))


def gbps_to_mb_per_ms(gbps: float) -> float:
    return gbps * 1000.0 / 8.0 / 1000.0


@dataclass(frozen=True)
class Asset:
    asset_id: str
    size_mb: float
    app: int
    scene: Optional[int]
    kind: str


@dataclass
class Task:
    task_id: int
    arrival_ms: float
    session_id: int
    app: int
    scene: int
    working_set: Tuple[str, ...]
    demand_ms: float
    deadline_ms: float
    priority: float
    complexity: str

    @property
    def key(self) -> Tuple[int, int]:
        return self.app, self.scene


@dataclass
class CacheEntry:
    size_mb: float
    last_touch_ms: float
    pinned_until_ms: float = 0.0


class NodeCache:
    def __init__(self, capacity_mb: float):
        self.capacity_mb = capacity_mb
        self.items: Dict[str, CacheEntry] = {}
        self.used_mb = 0.0

    def contains(self, asset_id: str, now_ms: float) -> bool:
        entry = self.items.get(asset_id)
        if entry is None:
            return False
        entry.last_touch_ms = now_ms
        return True

    def snapshot(self) -> Dict[str, float]:
        return {asset_id: entry.size_mb for asset_id, entry in self.items.items()}

    def free_mb(self) -> float:
        return max(0.0, self.capacity_mb - self.used_mb)

    def insert(self, asset: Asset, now_ms: float, pinned_until_ms: float = 0.0) -> List[str]:
        evicted: List[str] = []
        old = self.items.get(asset.asset_id)
        if old is not None:
            old.last_touch_ms = now_ms
            old.pinned_until_ms = max(old.pinned_until_ms, pinned_until_ms)
            return evicted
        if asset.size_mb > self.capacity_mb:
            return evicted
        while self.used_mb + asset.size_mb > self.capacity_mb:
            candidates = [
                (entry.last_touch_ms / max(1.0, entry.size_mb), asset_id)
                for asset_id, entry in self.items.items()
                if entry.pinned_until_ms <= now_ms
            ]
            if not candidates:
                break
            _, victim = min(candidates)
            victim_entry = self.items.pop(victim)
            self.used_mb -= victim_entry.size_mb
            evicted.append(victim)
        if self.used_mb + asset.size_mb <= self.capacity_mb:
            self.items[asset.asset_id] = CacheEntry(asset.size_mb, now_ms, pinned_until_ms)
            self.used_mb += asset.size_mb
        return evicted

    def pin_many(self, asset_ids: Iterable[str], until_ms: float) -> None:
        for asset_id in asset_ids:
            entry = self.items.get(asset_id)
            if entry is not None:
                entry.pinned_until_ms = max(entry.pinned_until_ms, until_ms)


class NodeRuntime:
    def __init__(self, spec: dict, cache_mb: float, util_window_ms: float):
        self.node_id = spec["id"]
        self.slots = int(spec["slots"])
        self.rate = float(spec["rate"])
        self.rack = int(spec["rack"])
        self.cache = NodeCache(cache_mb)
        self.slot_ready_ms = [0.0 for _ in range(self.slots)]
        self.util_window_ms = util_window_ms
        self.alive = True
        self.failed_until_ms = 0.0
        self.load_ewma = 0.0
        self.load_last_ms = 0.0
        self.load_tau_ms = 1800.0
        self.reported_util_bias = 0.0

    @property
    def capacity(self) -> float:
        return self.slots * self.rate

    def update_liveness(self, now_ms: float) -> None:
        if not self.alive and now_ms >= self.failed_until_ms:
            self.alive = True

    def active_slots(self, now_ms: float) -> int:
        return sum(1 for t in self.slot_ready_ms if t > now_ms)

    def queue_ms(self, now_ms: float) -> float:
        if not self.alive:
            return float("inf")
        return max(0.0, min(self.slot_ready_ms) - now_ms)

    def utilization(self, now_ms: float) -> float:
        physical = self.physical_utilization(now_ms)
        if not self.alive:
            return physical
        return max(0.0, min(0.99, physical + self.reported_util_bias))

    def physical_utilization(self, now_ms: float) -> float:
        if not self.alive:
            return 1.0
        self._decay_load(now_ms)
        backlog = sum(max(0.0, t - now_ms) for t in self.slot_ready_ms)
        denom = max(1.0, self.slots * self.util_window_ms)
        instant = backlog / denom
        return min(0.99, max(instant, self.load_ewma))

    def projected_utilization(self, now_ms: float, service_ms: float) -> float:
        if not self.alive:
            return 1.0
        slot_times = list(self.slot_ready_ms)
        idx = min(range(len(slot_times)), key=slot_times.__getitem__)
        start = max(now_ms, slot_times[idx])
        slot_times[idx] = start + service_ms
        backlog = sum(max(0.0, t - now_ms) for t in slot_times)
        denom = max(1.0, self.slots * self.util_window_ms)
        projected = max(self._peek_load(now_ms), backlog / denom)
        projected += service_ms / max(1.0, self.slots * self.load_tau_ms)
        return min(0.99, projected)

    def assign(self, now_ms: float, service_ms: float) -> Tuple[float, float, float]:
        self._decay_load(now_ms)
        idx = min(range(len(self.slot_ready_ms)), key=self.slot_ready_ms.__getitem__)
        start = max(now_ms, self.slot_ready_ms[idx])
        queue = max(0.0, start - now_ms)
        finish = start + service_ms
        self.slot_ready_ms[idx] = finish
        self.load_ewma = min(0.99, self.load_ewma + service_ms / max(1.0, self.slots * self.load_tau_ms))
        return queue, start, finish

    def fail(self, now_ms: float, duration_ms: float) -> None:
        self.alive = False
        self.failed_until_ms = now_ms + duration_ms

    def _peek_load(self, now_ms: float) -> float:
        if now_ms <= self.load_last_ms:
            return self.load_ewma
        return self.load_ewma * math.exp(-(now_ms - self.load_last_ms) / self.load_tau_ms)

    def _decay_load(self, now_ms: float) -> None:
        self.load_ewma = self._peek_load(now_ms)
        self.load_last_ms = now_ms


class WorkloadGenerator:
    def __init__(self, cfg: dict, seed: int):
        self.cfg = cfg
        self.workload_cfg = cfg["workload"]
        self.rng = random.Random(seed)
        self.assets: Dict[str, Asset] = {}
        self.base_by_app: Dict[int, List[str]] = {}
        self.scene_by_key: Dict[Tuple[int, int], List[str]] = {}
        self.user_by_app: Dict[int, List[str]] = {}
        self.sessions: List[dict] = []
        self._build_catalog()
        self._build_sessions()

    def _asset_size(self) -> float:
        lo = float(self.workload_cfg["asset_size_min_mb"])
        hi = float(self.workload_cfg["asset_size_max_mb"])
        return round(math.exp(self.rng.uniform(math.log(lo), math.log(hi))), 3)

    def _add_asset(self, asset_id: str, app: int, scene: Optional[int], kind: str) -> str:
        self.assets[asset_id] = Asset(asset_id, self._asset_size(), app, scene, kind)
        return asset_id

    def _build_catalog(self) -> None:
        apps = int(self.workload_cfg["applications"])
        scenes_per_app = int(self.workload_cfg["scenes_per_app"])
        for app in range(apps):
            base = []
            for i in range(int(self.workload_cfg["base_assets_per_app"])):
                base.append(self._add_asset(f"a{app}_b{i}", app, None, "base"))
            self.base_by_app[app] = base
            user_assets = []
            for i in range(int(self.workload_cfg["user_assets_per_app"])):
                user_assets.append(self._add_asset(f"a{app}_u{i}", app, None, "user"))
            self.user_by_app[app] = user_assets
            for scene in range(scenes_per_app):
                scene_assets = []
                for i in range(int(self.workload_cfg["scene_assets_per_scene"])):
                    scene_assets.append(self._add_asset(f"a{app}_s{scene}_{i}", app, scene, "scene"))
                self.scene_by_key[(app, scene)] = scene_assets

    def _zipf_weights(self, n: int) -> List[float]:
        alpha = float(self.workload_cfg["zipf_alpha"])
        return [1.0 / ((i + 1) ** alpha) for i in range(n)]

    def _build_sessions(self) -> None:
        apps = list(range(int(self.workload_cfg["applications"])))
        scenes = list(range(int(self.workload_cfg["scenes_per_app"])))
        app_weights = self._zipf_weights(len(apps))
        scene_weights = self._zipf_weights(len(scenes))
        for session_id in range(int(self.workload_cfg["active_sessions"])):
            app = weighted_choice(self.rng, apps, app_weights)
            scene = weighted_choice(self.rng, scenes, scene_weights)
            complexity = "high" if self.rng.random() < float(self.workload_cfg["high_complexity_share"]) else "low"
            self.sessions.append({"session_id": session_id, "app": app, "scene": scene, "complexity": complexity})

    def _sample_assets(self, pool: Sequence[str], target_mb: float, used: set) -> float:
        added = 0.0
        local = list(pool)
        self.rng.shuffle(local)
        for asset_id in local:
            if asset_id in used:
                continue
            used.add(asset_id)
            added += self.assets[asset_id].size_mb
            if added >= target_mb:
                break
        return added

    def _working_set(self, session: dict) -> Tuple[str, ...]:
        complexity = session["complexity"]
        key = "high_working_set_mb" if complexity == "high" else "low_working_set_mb"
        lo, hi = self.workload_cfg[key]
        target = self.rng.uniform(float(lo), float(hi))
        app = session["app"]
        scene = session["scene"]
        used: set = set()
        self._sample_assets(self.base_by_app[app], target * 0.35, used)
        self._sample_assets(self.scene_by_key[(app, scene)], target * 0.45, used)
        self._sample_assets(self.user_by_app[app], target * 0.20, used)
        return tuple(sorted(used))

    def _maybe_scene_transition(self, session: dict) -> None:
        if self.rng.random() >= float(self.workload_cfg["scene_transition_prob"]):
            return
        scenes = list(range(int(self.workload_cfg["scenes_per_app"])))
        session["scene"] = self.rng.choice(scenes)

    def generate_tasks(self, duration_ms: float) -> List[Task]:
        rate_per_s = float(self.workload_cfg["active_sessions"]) * float(self.workload_cfg["lambda_per_session"])
        rate_per_ms = rate_per_s / 1000.0
        deadlines = [float(x) for x in self.workload_cfg["deadlines_ms"]]
        deadline_weights = [float(x) for x in self.workload_cfg["deadline_shares"]]
        tasks: List[Task] = []
        now = 0.0
        task_id = 0
        while now < duration_ms:
            now += self.rng.expovariate(rate_per_ms)
            if now > duration_ms:
                break
            session = self.rng.choice(self.sessions)
            self._maybe_scene_transition(session)
            working_set = self._working_set(session)
            volume = sum(self.assets[a].size_mb for a in working_set)
            demand = 7.5 + 0.018 * volume + self.rng.uniform(0.0, 7.0)
            if session["complexity"] == "high":
                demand += self.rng.uniform(3.0, 8.0)
            deadline = weighted_choice(self.rng, deadlines, deadline_weights)
            priority = 1.0
            tasks.append(
                Task(
                    task_id=task_id,
                    arrival_ms=now,
                    session_id=session["session_id"],
                    app=session["app"],
                    scene=session["scene"],
                    working_set=working_set,
                    demand_ms=demand,
                    deadline_ms=deadline,
                    priority=priority,
                    complexity=session["complexity"],
                )
            )
            task_id += 1
        return tasks


def default_method_names() -> List[str]:
    return ["FIFO", "SPT", "RR", "SAG", "A-LALL", "K8s-FS"]
