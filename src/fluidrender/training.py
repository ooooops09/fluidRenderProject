"""Masked policy fusion, expert baselines and fixed-reference rewards."""
import math
from .core import percentile


def masked_softmax(logits, mask):
    if len(logits) != len(mask) or not logits:
        raise ValueError('Invalid policy dimensions')
    active = [x for x, keep in zip(logits, mask) if keep]
    if not active:
        return [0.] * len(logits)  # no feasible placement; never unmask a worker
    offset = max(active)
    w = [math.exp(x-offset) if keep else 0. for x, keep in zip(logits, mask)]
    return [x/sum(w) for x in w]


def product_of_experts(latency_logits, load_logits, mask):
    return masked_softmax([a+b for a,b in zip(latency_logits, load_logits)], mask)


def counterfactual_baseline(policy_probs, q_values, top_b=None):
    pairs = [(p, q) for p,q in zip(policy_probs, q_values) if p > 0]
    if top_b:
        pairs = sorted(pairs, reverse=True)[:top_b]
    total = sum(p for p,q in pairs)
    return sum(p*q for p,q in pairs)/total if total else 0.


def counterfactual_advantage(q_tot, policy_probs, q_values, top_b=None):
    return q_tot-counterfactual_baseline(policy_probs, q_values, top_b)


def team_reward(latency_ms, deadline_ms, load_skew, lambda_load=1., eta=2., t_ref_ms=100., completed=True):
    if t_ref_ms <= 0:
        raise ValueError('T_ref must be positive')
    cost = min(latency_ms/t_ref_ms, 1.) if completed else 1.
    missed = not completed or latency_ms > deadline_ms
    return -cost-lambda_load*load_skew-eta*int(missed)


def latency_reward(latency_ms, deadline_ms, completed=True, t_ref_ms=100., w_fail=-1.):
    return (deadline_ms-latency_ms)/t_ref_ms if completed and latency_ms <= deadline_ms else w_fail
