from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Tuple


@dataclass
class AggStats:
    count: int
    success_rate: float
    avg_time_ms: float
    avg_tokens_total: float


def aggregate(records: Iterable[Dict]) -> AggStats:
    rows = list(records)
    if not rows:
        return AggStats(count=0, success_rate=0.0, avg_time_ms=0.0, avg_tokens_total=0.0)
    cnt = len(rows)
    succ = sum(1 for r in rows if r.get("success")) / cnt
    avg_t = mean([r.get("time_ms", 0) for r in rows])
    avg_tok = mean([r.get("tokens_total", 0) for r in rows])
    return AggStats(count=cnt, success_rate=succ, avg_time_ms=avg_t, avg_tokens_total=avg_tok)


def group_by(records: Iterable[Dict], *keys: str) -> Dict[Tuple, List[Dict]]:
    groups: Dict[Tuple, List[Dict]] = {}
    for r in records:
        k = tuple(r.get(key) for key in keys)
        groups.setdefault(k, []).append(r)
    return groups


__all__ = ["AggStats", "aggregate", "group_by"]


