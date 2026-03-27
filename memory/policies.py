from dataclasses import dataclass

from .utils import clamp, parse_timestamp, utc_now


@dataclass
class DecayConfig:
    decay_base: float = 0.992
    usage_weight: float = 0.04
    hard_ttl_days: int = 120
    keep_if_importance_at_least: float = 0.8


def decay_and_prune(mem_items, now_ts=None, cfg=None):
    cfg = cfg or DecayConfig()
    now_ts = now_ts or utc_now()
    updated = []
    for item in mem_items:
        age_days = max(0, (now_ts - parse_timestamp(item.last_accessed_at)).days)
        freshness = cfg.decay_base ** age_days
        retrieval_bonus = cfg.usage_weight * item.times_retrieved
        item.decay_score = clamp((item.importance * freshness) + retrieval_bonus, 0.0, 1.0)
        if age_days > cfg.hard_ttl_days and item.importance < cfg.keep_if_importance_at_least:
            item.status = "archived"
        updated.append(item)
    return updated
