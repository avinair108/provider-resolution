"""Dispatch a structured agent output to the right scorer for its gold item."""

from collections.abc import Iterable, Mapping

from evals.contract import AgentOutput, GoldItem
from evals.scorers.answers import score_abstention, score_conflict, score_numeric, score_set


def score_answer(item: GoldItem, output: AgentOutput, entity_of: Mapping[str, str],
                 disagreeing_records: Iterable[str] = ()) -> dict:
    """An abstention is judged first: declining an answerable question fails however good the rest looks, and
    a confident value on an unanswerable question fails even if abstained is also set."""
    if item.answer_kind == "abstain":
        result = score_abstention(output.abstained, True)
        has_content = output.value is not None or bool(output.providers)
        return {**result, "passed": result["passed"] and not has_content, "answered_anyway": has_content}
    if output.abstained:
        return {"passed": False, "reason": "false_abstention"}
    if item.answer_kind in ("money", "count", "percent"):
        return score_numeric(output.value, item.gold_value, item.answer_kind)
    if item.answer_kind == "set":
        return score_set(output.providers, item.gold_entities, entity_of)
    if item.answer_kind == "conflict":
        return score_conflict(output.conflict_records, item.gold_conflict_entity, disagreeing_records, entity_of)
    raise ValueError(f"Unknown answer_kind: {item.answer_kind}")
