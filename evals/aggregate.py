"""Roll per-question rows up: repeats reported as ranges, every metric per slice, cost and latency percentiles."""

from collections import defaultdict
from collections.abc import Callable, Iterable


def percentile(values: list[float], q: float) -> float | None:
    """Linear interpolation between closest ranks."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def repeat_range(rows: Iterable[dict], metric: Callable[[list[dict]], float | None]) -> dict:
    """Compute a metric once per repeat, then report mean and range. A single run is not a result."""
    by_repeat = defaultdict(list)
    for row in rows:
        by_repeat[row["repeat"]].append(row)
    values = [v for v in (metric(rs) for _, rs in sorted(by_repeat.items())) if v is not None]
    if not values:
        return {"mean": None, "min": None, "max": None, "repeats": 0}
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values), "repeats": len(values)}


def by_slice(rows: Iterable[dict], metric: Callable[[list[dict]], float | None], key: str = "slices") -> dict:
    """Group by a slice field and apply the metric with repeat ranges. A row can sit in several slices."""
    groups = defaultdict(list)
    for row in rows:
        labels = row[key] if isinstance(row[key], (list, tuple, set)) else [row[key]]
        for label in labels:
            groups[label].append(row)
    return {label: {**repeat_range(rs, metric), "n_rows": len(rs)} for label, rs in sorted(groups.items())}


def pass_rate(rows: list[dict]) -> float | None:
    scored = [r["passed"] for r in rows if r.get("passed") is not None]
    return sum(scored) / len(scored) if scored else None


def cost_latency(rows: list[dict]) -> dict:
    latency = [r["latency_s"] for r in rows]
    tokens = [r["prompt_tokens"] + r["completion_tokens"] for r in rows]
    requests = [r["requests"] for r in rows]
    return {
        "n": len(rows),
        "latency_p50_s": percentile(latency, 0.5),
        "latency_p95_s": percentile(latency, 0.95),
        "tokens_mean": sum(tokens) / len(tokens) if tokens else None,
        "requests_mean": sum(requests) / len(requests) if requests else None,
    }
