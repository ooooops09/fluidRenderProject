"""Completed-task latency, offered-request misses and paired-seed statistics."""
import math
import statistics
from .core import mean, percentile


def summarize(records, duration_s=None, traffic_mb=None, store_mb=None):
    cohort = [r for r in records if not r.warmup]
    done = [r for r in cohort if r.status == 'completed']
    n = len(cohort)
    lat = [r.latency_ms for r in done]
    row = {
        'tasks': n, 'completed_tasks': len(done),
        'rejected_tasks': sum(r.status == 'rejected' for r in cohort),
        'failed_tasks': sum(r.status == 'failed' for r in cohort),
        'mean_latency_ms': mean(lat) if lat else None,
        'p99_latency_ms': percentile(lat, .99) if lat else None,
        'slo_miss_rate': sum(r.slo_miss for r in cohort) / n if n else None,
        'on_time_goodput_per_s': sum(not r.slo_miss for r in cohort) / duration_s if duration_s else None,
        'sync_overhead_gb': traffic_mb / 1000 if traffic_mb is not None else None,
        'object_store_gb': store_mb / 1000 if store_mb is not None else None,
        'mean_sync_ms': mean([r.sync_ms for r in done]) if done else None,
        'mean_queue_ms': mean([r.queue_ms for r in done]) if done else None,
        'mean_render_ms': mean([r.render_ms for r in done]) if done else None,
        'load_skew': mean([r.load_skew for r in cohort]) if cohort else None,
        'jain_util': mean([r.jain_util for r in cohort]) if cohort else None,
        'inference_ms': mean([r.inference_ms for r in cohort]) if cohort else None,
        'failed_peer_lookup_rate': sum(r.peer_failures > 0 for r in cohort) / n if n else None,
        'validation_cost': -mean([r.team_reward for r in cohort]) if cohort else None,
    }
    return row


def aggregate(rows):
    out = {'runs': len(rows)}
    if not rows:
        return out
    for key in rows[0]:
        if key in ('seed', 'scheduler', 'training_seed', 'scenario'):
            continue
        values = [r[key] for r in rows if r.get(key) is not None]
        out[key] = mean(values) if values else None
        out[key + '_sd'] = statistics.stdev(values) if len(values) > 1 else 0. if values else None
        out[key + '_n'] = len(values)
    return out


def hierarchical_aggregate(rows):
    """First average workloads per trained policy, then weight policies equally."""
    groups = {}
    for row in rows:
        groups.setdefault(row['training_seed'], []).append(row)
    per_policy = [{k: v for k, v in aggregate(group).items()
                   if not k.endswith(('_sd', '_n')) and k != 'runs'} for group in groups.values()]
    result = aggregate(per_policy)
    # Exact two-sided 95% t critical value for five independent policies, df=4.
    if len(per_policy) == 5:
        for k in per_policy[0]:
            if result[k + '_n'] == 5:
                result[k + '_ci95_half'] = 2.776445105 * result[k + '_sd'] / math.sqrt(5)
    return result


def tail_composition(records, start_ms, end_ms):
    done = [r for r in records if r.status == 'completed' and start_ms <= r.arrival_ms < end_ms]
    if not done:
        return {'completed': 0}
    tail = sorted(done, key=lambda r: r.latency_ms)[-max(1, math.ceil(len(done) * .01)):]
    avg = mean([r.latency_ms for r in done])
    groups = {}
    for r in done:
        groups.setdefault(r.node_id, []).append(r.latency_ms)
    within = sum(sum((x - mean(xs)) ** 2 for x in xs) for xs in groups.values()) / len(done)
    between = sum(len(xs) * (mean(xs) - avg) ** 2 for xs in groups.values()) / len(done)
    return {'completed': len(done), 'tail_count': len(tail),
            'tail_mean_ms': mean([r.latency_ms for r in tail]),
            'tail_sync_ms': mean([r.sync_ms for r in tail]),
            'tail_queue_ms': mean([r.queue_ms for r in tail]),
            'tail_render_ms': mean([r.render_ms for r in tail]),
            'within_worker_variance_ms2': within, 'between_worker_variance_ms2': between,
            'total_variance_ms2': within + between}
