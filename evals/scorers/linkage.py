"""Pair, cluster, blocking and calibration metrics, all measured against hidden-NPI truth.

Unlabeled records (no NPI in the source, none recoverable) are excluded from every metric. They are unknown, not
non-matches: counting them as non-matches would punish a correct link for a gap in the gold.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from itertools import combinations

from evals.contract import Pair, canonical_pair


def true_pairs(entity_of: Mapping[str, str]) -> set[Pair]:
    """All within-entity record pairs. Grows quadratically with cluster size, hence per-source-pair helpers later."""
    members = defaultdict(list)
    for record, entity in entity_of.items():
        members[entity].append(record)
    return {canonical_pair(a, b) for records in members.values() for a, b in combinations(sorted(records), 2)}


def _labeled_pairs(pairs: Iterable[Pair], entity_of: Mapping[str, str]) -> set[Pair]:
    canonical = {canonical_pair(*p) for p in pairs}
    return {p for p in canonical if p[0] in entity_of and p[1] in entity_of}


def score_links(predicted: Iterable[Pair], entity_of: Mapping[str, str], gold_pairs: set[Pair] | None = None) -> dict:
    """Pair precision and recall. `gold_pairs` narrows recall to a subset, e.g. one source pair or slice."""
    gold = gold_pairs if gold_pairs is not None else true_pairs(entity_of)
    pred = _labeled_pairs(predicted, entity_of)
    if gold_pairs is not None:
        allowed = {r for p in gold_pairs for r in p}
        pred = {p for p in pred if p[0] in allowed and p[1] in allowed}
    tp = sum(entity_of[a] == entity_of[b] for a, b in pred)
    fp = len(pred) - tp
    fn = len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def blocking_metrics(candidates: Iterable[Pair], entity_of: Mapping[str, str], total_possible: int) -> dict:
    """Recall caps everything after blocking: a true link never proposed can't be recovered by any later step."""
    gold = true_pairs(entity_of)
    all_candidates = {canonical_pair(*p) for p in candidates}
    found = len(gold & all_candidates)
    return {
        "recall": found / len(gold) if gold else None,
        "n_candidates": len(all_candidates),
        "reduction_ratio": 1 - len(all_candidates) / total_possible if total_possible else None,
    }


def clusters_from_links(records: Iterable[str], links: Iterable[Pair]) -> dict[str, str]:
    """Transitive closure over accepted links. This is how one bad link chains two providers into one cluster."""
    parent = {r: r for r in records}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in links:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {r: find(r) for r in parent}


def score_clusters(cluster_of: Mapping[str, str], entity_of: Mapping[str, str]) -> dict:
    """B-cubed precision and recall, purity, and the count of clusters that merge different providers."""
    labeled = [r for r in cluster_of if r in entity_of]
    if not labeled:
        return {"bcubed_precision": None, "bcubed_recall": None, "bcubed_f1": None, "purity": None,
                "overmerged_clusters": 0, "n_clusters": 0}
    by_cluster, by_entity = defaultdict(list), defaultdict(list)
    for r in labeled:
        by_cluster[cluster_of[r]].append(r)
        by_entity[entity_of[r]].append(r)
    # Compare against the entity's labeled records in `cluster_of`, so a record the matcher never saw isn't counted.
    pair_counts = Counter((cluster_of[r], entity_of[r]) for r in labeled)
    precision = sum(pair_counts[(cluster_of[r], entity_of[r])] / len(by_cluster[cluster_of[r]]) for r in labeled) / len(labeled)
    recall = sum(pair_counts[(cluster_of[r], entity_of[r])] / len(by_entity[entity_of[r]]) for r in labeled) / len(labeled)
    majority = sum(Counter(entity_of[r] for r in rs).most_common(1)[0][1] for rs in by_cluster.values())
    overmerged = sum(len({entity_of[r] for r in rs}) > 1 for rs in by_cluster.values())
    return {
        "bcubed_precision": precision,
        "bcubed_recall": recall,
        "bcubed_f1": 2 * precision * recall / (precision + recall),
        "purity": majority / len(labeled),
        "overmerged_clusters": overmerged,
        "n_clusters": len(by_cluster),
    }


def expected_calibration_error(confidences: Iterable[float], labels: Iterable[bool], n_bins: int = 10) -> dict:
    """Equal-width ECE plus the per-bin table behind it, so one summary number doesn't hide where it is off."""
    data = list(zip(confidences, labels, strict=True))
    if not data:
        return {"ece": None, "n": 0, "bins": []}
    bins = defaultdict(list)
    for confidence, label in data:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence outside [0, 1]: {confidence}")
        bins[min(int(confidence * n_bins), n_bins - 1)].append((confidence, label))
    table, ece = [], 0.0
    for index in sorted(bins):
        members = bins[index]
        mean_confidence = sum(c for c, _ in members) / len(members)
        accuracy = sum(bool(l) for _, l in members) / len(members)
        ece += len(members) / len(data) * abs(accuracy - mean_confidence)
        table.append({"low": index / n_bins, "high": (index + 1) / n_bins, "n": len(members),
                      "mean_confidence": mean_confidence, "accuracy": accuracy})
    return {"ece": ece, "n": len(data), "bins": table}
