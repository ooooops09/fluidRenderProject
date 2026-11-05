"""Authenticated cache reports and freshness; no heartbeat-based claim renewal."""
from dataclasses import dataclass, asdict
import hashlib
import hmac
import json


@dataclass(frozen=True)
class Report:
    worker: str
    boot: int
    sequence: int
    generated_ms: float
    version: int
    full: bool
    present: tuple
    removed: tuple = ()

    def payload(self):
        return json.dumps(asdict(self), sort_keys=True, separators=(',', ':')).encode()


def sign(report, key):
    if len(key) != 32:
        raise ValueError('A worker key must contain 256 bits')
    return hmac.digest(key, report.payload(), 'sha256')


def verify_chunk(payload, version, trusted_version, trusted_digest):
    return version == trusted_version and hmac.compare_digest(hashlib.sha256(payload).hexdigest(), trusted_digest)


class ValidatedDirectory:
    def __init__(self, keys, stale_ms=500., heartbeat_timeout_ms=500.):
        self.keys = dict(keys)
        self.stale_ms = stale_ms
        self.heartbeat_timeout_ms = heartbeat_timeout_ms
        self.entries = {n: {} for n in keys}
        self.last = {}  # (boot, sequence, directory version)
        self.heartbeat_ms = {}
        self.needs_refresh = set()
        self.accepted = self.rejected = 0
        self.last_full_generated = {}
        self.observation_missing = set()

    def receive(self, report, tag, now):
        key = self.keys.get(report.worker)
        if key is None or not hmac.compare_digest(sign(report, key), tag) or report.generated_ms > now:
            self.rejected += 1
            return False
        previous = self.last.get(report.worker)
        if previous and (report.boot < previous[0] or
                         (report.boot == previous[0] and (report.sequence <= previous[1] or report.version < previous[2]))):
            self.rejected += 1
            return False
        new_boot = not previous or report.boot != previous[0]
        gap = previous and report.boot == previous[0] and report.version != previous[2] + 1
        self.last[report.worker] = (report.boot, report.sequence, report.version)
        if not report.full and (new_boot or gap or report.worker in self.needs_refresh):
            self.entries[report.worker].clear()
            self.needs_refresh.add(report.worker)
            self.rejected += 1
            return False
        if report.full:
            self.entries[report.worker] = {}
            self.needs_refresh.discard(report.worker)
            self.last_full_generated[report.worker] = report.generated_ms
        for chunk in report.removed:
            self.entries[report.worker].pop(chunk, None)
        for chunk in report.present:
            self.entries[report.worker][chunk] = report.generated_ms
        self.accepted += 1
        return True

    def heartbeat(self, worker, now):
        # This is a trusted control-plane liveness notification, not a cache update.
        self.heartbeat_ms[worker] = now

    def observed_has(self, worker, chunk, now):
        stamp = self.entries.get(worker, {}).get(chunk)
        alive = now - self.heartbeat_ms.get(worker, float('-inf')) <= self.heartbeat_timeout_ms
        return (alive and worker not in self.observation_missing and worker not in self.needs_refresh and stamp is not None and
                (self.stale_ms is None or now - stamp <= self.stale_ms))

    def invalidate(self, worker, chunk):
        self.entries.get(worker, {}).pop(chunk, None)

    def invalidate_node(self, worker):
        self.entries[worker] = {}
        self.needs_refresh.add(worker)
