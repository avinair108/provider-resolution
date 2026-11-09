"""Question-level answer scorers: numeric, set, conflict, abstention."""

from collections.abc import Iterable, Mapping

from evals.numbers import Number, extract_numbers

# Rounding is accepted only when the display is still informative: "$1M" must not pass for $1.4M because
# "1" could round from anything in 0.5-1.5M.
MAX_DISPLAY_REL = 0.05


def _money_matches(n: Number, gold: float, rel_tol: float) -> bool:
    if n.kind == "percent":
        return False
    if abs(gold) > 0 and n.precision / abs(gold) > MAX_DISPLAY_REL:
        return False
    return abs(n.value - gold) <= max(rel_tol * abs(gold), n.precision * 1.02)


def _percent_matches(n: Number, gold: float, abs_tol: float) -> bool:
    return n.kind == "percent" and abs(n.value - gold) <= max(abs_tol, n.precision * 1.02)


def _count_matches(n: Number, gold: float) -> bool:
    return n.kind == "plain" and n.value == gold


def score_numeric(value: str | None, gold: float, kind: str, rel_tol: float = 0.005, pct_abs_tol: float = 0.1) -> dict:
    """Score the asserted `value` field. Exactly one number may be asserted: a hedge such as "$10K or $12K" fails
    even though one of them is right, which is the leniency this contract exists to remove."""
    numbers = extract_numbers(value or "")
    match_fn = {
        "money": lambda n: _money_matches(n, gold, rel_tol),
        "count": lambda n: _count_matches(n, gold),
        "percent": lambda n: _percent_matches(n, gold, pct_abs_tol),
    }[kind]
    if not numbers:
        return {"passed": False, "reason": "no_number", "gold": gold}
    if len(numbers) > 1:
        return {"passed": False, "reason": "multiple_values", "gold": gold, "found": [n.text for n in numbers]}
    passed = match_fn(numbers[0])
    return {"passed": passed, "reason": "match" if passed else "wrong_value", "gold": gold, "found": [numbers[0].text]}


def to_entities(record_ids: Iterable[str], entity_of: Mapping[str, str]) -> tuple[set[str], list[str]]:
    """Map record IDs to gold entities. An ID the gold doesn't know is returned separately and counts as wrong."""
    entities, unknown = set(), []
    for r in record_ids:
        if r in entity_of:
            entities.add(entity_of[r])
        else:
            unknown.append(r)
    return entities, unknown


def score_set(providers: Iterable[str], gold_entities: Iterable[str], entity_of: Mapping[str, str]) -> dict:
    """Set overlap at the entity level, so naming any of a provider's records counts once.

    Unknown IDs add false positives. Listing the same provider through two records is not rewarded or penalized."""
    gold = set(gold_entities)
    predicted, unknown = to_entities(providers, entity_of)
    tp = len(predicted & gold)
    fp = len(predicted - gold) + len(unknown)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else (1.0 if not gold else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"passed": fp == 0 and fn == 0, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "unknown_ids": unknown}


def score_conflict(conflict_records: Iterable[str], gold_entity: str, disagreeing_records: Iterable[str],
                   entity_of: Mapping[str, str]) -> dict:
    """A conflict answer must surface at least two of the disagreeing records, all belonging to the right provider.
    Picking one side silently, or setting another provider's record against this one, fails."""
    reported = set(conflict_records)
    disagreeing = set(disagreeing_records)
    wrong_entity = sorted(r for r in reported if entity_of.get(r) != gold_entity)
    surfaced = reported & disagreeing
    passed = not wrong_entity and len(surfaced) >= 2
    return {"passed": passed, "surfaced": len(surfaced), "wrong_entity_records": wrong_entity}


def score_abstention(abstained: bool, should_abstain: bool) -> dict:
    return {"passed": abstained == should_abstain, "abstained": abstained, "should_abstain": should_abstain}


def abstention_rates(rows: Iterable[tuple[bool, bool]]) -> dict:
    """Rates over (abstained, should_abstain) rows. None means no question of that kind, not zero."""
    rows = list(rows)
    unanswerable = [a for a, s in rows if s]
    answerable = [a for a, s in rows if not s]
    return {
        "abstention_rate": sum(unanswerable) / len(unanswerable) if unanswerable else None,
        "false_abstention_rate": sum(answerable) / len(answerable) if answerable else None,
        "n_unanswerable": len(unanswerable),
        "n_answerable": len(answerable),
    }
