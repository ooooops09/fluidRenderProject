"""Directed, weighted max-min fluid transport. Units: MB, ms, Gbps (decimal)."""
from dataclasses import dataclass
import math


def progressive_fill(capacities, paths, weights=None):
    """Weighted max-min rates subject to EVERY traversed link's constraint."""
    weights = weights or {f: 1.0 for f in paths}
    if any(c < 0 for c in capacities.values()):
        raise ValueError('Negative capacity')
    if any(not p or any(l not in capacities for l in p) for p in paths.values()):
        raise ValueError('Each flow needs a nonempty, valid directed path')
    if any(weights[f] <= 0 for f in paths):
        raise ValueError('Weights must be positive')
    rates = dict.fromkeys(paths, 0.0)
    active = set(paths)
    while active:
        residual = {l: max(0., c - sum(rates[f] for f in paths if l in paths[f]))
                    for l, c in capacities.items()}
        slopes = {l: sum(weights[f] for f in active if l in paths[f]) for l in capacities}
        limits = {l: residual[l] / v for l, v in slopes.items() if v > 0}
        step = min(limits.values())
        for f in active:
            rates[f] += step * weights[f]
        tight = {l for l, v in limits.items() if v <= step + 1e-10}
        frozen = {f for f in active if tight.intersection(paths[f])}
        if not frozen:
            raise RuntimeError('Progressive filling made no progress')
        active -= frozen
    return rates


@dataclass
class Flow:
    flow_id: int
    task_id: int
    asset_id: str
    source: str | None
    destination: str
    path: tuple
    remaining_mb: float
    delay_ms: float
    sent_mb: float = 0.


class FluidNetwork:
    def __init__(self, cfg, nodes):
        self.cfg = cfg['network']
        self.racks = {n.node_id: n.rack for n in nodes}
        self.base = {}
        access = self.cfg.get('worker_access_gbps', self.cfg['same_rack_gbps']) / 8.
        rack = self.cfg.get('rack_uplink_gbps', self.cfg['cross_rack_gbps']) / 8.
        for n in nodes:
            self.base['in:' + n.node_id] = access
            self.base['out:' + n.node_id] = access
        for r in set(self.racks.values()):
            self.base['rack_up:' + str(r)] = rack
            self.base['rack_down:' + str(r)] = rack
        self.base['store'] = self.cfg['object_store_gbps'] / 8.
        self.flows = {}
        self.next_id = 0
        self.window_mb = self.window_store_mb = 0.
        self.measurement = tuple(x * 1000 for x in cfg.get('measurement_window_s',
                            [cfg['simulation']['warmup_s'], cfg['simulation']['duration_s']]))

    def path(self, source, destination):
        dr = self.racks[destination]
        if source is None:
            return ('store', 'rack_down:' + str(dr), 'in:' + destination), self.cfg['object_store_latency_ms']
        sr = self.racks[source]
        middle = () if sr == dr else ('rack_up:' + str(sr), 'rack_down:' + str(dr))
        delay = self.cfg['same_rack_latency_ms'] if sr == dr else self.cfg['cross_rack_latency_ms']
        return ('out:' + source,) + middle + ('in:' + destination,), delay

    def capacities(self, now):
        result = dict(self.base)
        for event in self.cfg.get('degradations', []):
            if event['start_s'] * 1000 <= now < event['end_s'] * 1000:
                link = event['link']
                if link not in result:
                    raise ValueError('Unknown degraded link: ' + link)
                result[link] *= event['factor']
        # Reservations represent only traffic NOT present as explicit flows.
        for link, fraction in self.cfg.get('external_reservation_fraction', {}).items():
            if not 0 <= fraction <= 1:
                raise ValueError('Reservation fraction outside [0, 1]')
            result[link] *= 1 - fraction
        return result

    def rates(self, now):
        paths = {f: flow.path for f, flow in self.flows.items()}
        weights = dict.fromkeys(paths, 1.)
        rho = self.cfg.get('background_weight_rho', 0.)
        n_store = sum('store' in p for p in paths.values())
        if rho > 0 and n_store:
            paths[-1] = ('store',)
            weights[-1] = rho * n_store
        return progressive_fill(self.capacities(now), paths, weights) if paths else {}

    def estimate(self, source, destination, size_mb, now):
        path, delay = self.path(source, destination)
        paths = {f: flow.path for f, flow in self.flows.items()}
        paths[-2] = path
        weights = dict.fromkeys(paths, 1.)
        rho = self.cfg.get('background_weight_rho', 0.)
        n = sum('store' in p for p in paths.values())
        if rho > 0 and n:
            paths[-1] = ('store',)
            weights[-1] = rho * n
        rate = progressive_fill(self.capacities(now), paths, weights)[-2]
        return size_mb / rate + delay if rate > 0 else math.inf

    def add(self, task_id, asset, source, destination):
        self.next_id += 1
        path, delay = self.path(source, destination)
        flow = Flow(self.next_id, task_id, asset.asset_id, source, destination,
                    path, asset.size_mb, delay)
        self.flows[flow.flow_id] = flow
        return flow

    def next_completion(self, now):
        rates = self.rates(now)
        return min((now + f.remaining_mb / rates[i] for i, f in self.flows.items()
                    if rates[i] > 0), default=math.inf)

    def advance(self, start, finish):
        rates = self.rates(start)
        measured = max(0., min(finish, self.measurement[1]) - max(start, self.measurement[0]))
        for i, f in self.flows.items():
            sent = min(f.remaining_mb, rates[i] * (finish - start))
            f.remaining_mb = max(0., f.remaining_mb - sent)
            f.sent_mb += sent
            count = rates[i] * measured
            self.window_mb += count
            if f.source is None:
                self.window_store_mb += count
        completed = [f for f in self.flows.values() if f.remaining_mb < 1e-8]
        for f in completed:
            del self.flows[f.flow_id]
        return completed

    def cancel(self, task_id):
        cancelled = [f for f in self.flows.values() if f.task_id == task_id]
        for f in cancelled:
            del self.flows[f.flow_id]
        return cancelled
