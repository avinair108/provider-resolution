"""Scorers tested on hand-made right and wrong answers, before any LLM exists. All providers here are synthetic."""

import pytest

from evals.aggregate import by_slice, cost_latency, pass_rate, percentile, repeat_range
from evals.contract import AgentOutput, Claim, GoldItem, canonical_pair
from evals.scorers import score_answer
from evals.scorers.answers import abstention_rates, score_conflict, score_numeric, score_set
from evals.scorers.linkage import (
    blocking_metrics,
    clusters_from_links,
    expected_calibration_error,
    score_clusters,
    score_links,
    true_pairs,
)
from evals.scorers.process import score_citations, score_routing

# Entity E1 appears in three sources; E2 in two; E3 only in the registry.
ENTITY_OF = {"np1": "E1", "op1": "E1", "pd1": "E1", "np2": "E2", "op2": "E2", "np3": "E3"}


@pytest.mark.parametrize("value, gold, kind, expected", [
    ("$12,345.67", 12345.67, "money", True),
    ("$12,346", 12345.67, "money", True),           # whole-dollar rounding
    ("$12.3K", 12345.67, "money", True),            # scaled rounding
    ("$12.35 thousand", 12345.67, "money", True),
    ("$1M", 1_400_000, "money", False),             # "1" could round from anything: not informative
    ("$0B", 12345, "money", False),
    ("$12,000", 12345.67, "money", False),          # 2.8% off and not a rounding of it
    ("$1,234.56", 12345.67, "money", False),        # decimal slip
    ("$10K or $12.3K", 12345.67, "money", False),   # hedge: one is right, but two are asserted
    ("about 45%", 45.0, "percent", True),
    ("45.04%", 45.0, "percent", True),
    ("45", 45.0, "percent", False),                 # unit dropped
    ("54%", 45.0, "percent", False),                # digit swap
    ("7", 7, "count", True),
    ("7.5", 7, "count", False),
    ("8", 7, "count", False),
    ("7%", 7, "count", False),
    ("1,201", 1200, "count", False),                # counts are exact: no relative tolerance
    ("no idea", 7, "count", False),
    ("", 7, "count", False),
    (None, 7, "count", False),
])
def test_numeric(value, gold, kind, expected):
    assert score_numeric(value, gold, kind)["passed"] is expected


def test_numeric_hedge_reports_reason():
    assert score_numeric("$10K or $12K", 12000, "money")["reason"] == "multiple_values"


def test_set_counts_entities_not_records():
    # Two records of E1 are one correct provider, not a duplicate or an extra.
    result = score_set(["np1", "op1", "np2"], ["E1", "E2"], ENTITY_OF)
    assert result["passed"] and result["f1"] == 1.0


def test_set_wrong_answers():
    assert score_set(["np1"], ["E1", "E2"], ENTITY_OF)["recall"] == 0.5
    extra = score_set(["np1", "np2"], ["E1"], ENTITY_OF)
    assert not extra["passed"] and extra["precision"] == 0.5


def test_set_invented_id_is_a_false_positive():
    result = score_set(["np1", "np999"], ["E1"], ENTITY_OF)
    assert not result["passed"] and result["fp"] == 1 and result["unknown_ids"] == ["np999"]


def test_empty_set_answers():
    assert score_set([], [], ENTITY_OF)["passed"]
    assert not score_set([], ["E1"], ENTITY_OF)["passed"]


def test_conflict_needs_two_real_sides():
    assert score_conflict(["np1", "op1"], "E1", ["np1", "op1"], ENTITY_OF)["passed"]
    assert not score_conflict(["np1"], "E1", ["np1", "op1"], ENTITY_OF)["passed"]  # picked one side silently


def test_conflict_with_another_provider_fails():
    result = score_conflict(["np1", "op1", "np2"], "E1", ["np1", "op1"], ENTITY_OF)
    assert not result["passed"] and result["wrong_entity_records"] == ["np2"]


def _item(kind, **kwargs):
    return GoldItem(id="q", question="?", question_type="x", answer_kind=kind, required_sources=[], **kwargs)


def test_dispatch_abstain():
    assert score_answer(_item("abstain"), AgentOutput(abstained=True), ENTITY_OF)["passed"]
    assert not score_answer(_item("abstain"), AgentOutput(value="$5"), ENTITY_OF)["passed"]
    # Hedged: says it abstains but still gives a figure.
    assert not score_answer(_item("abstain"), AgentOutput(abstained=True, value="$5"), ENTITY_OF)["passed"]


def test_dispatch_false_abstention():
    result = score_answer(_item("money", gold_value=5.0), AgentOutput(abstained=True, value="$5"), ENTITY_OF)
    assert not result["passed"] and result["reason"] == "false_abstention"


def test_dispatch_each_kind():
    assert score_answer(_item("count", gold_value=3), AgentOutput(value="3"), ENTITY_OF)["passed"]
    assert score_answer(_item("set", gold_entities=["E3"]), AgentOutput(providers=["np3"]), ENTITY_OF)["passed"]
    conflict = _item("conflict", gold_conflict_entity="E1")
    assert score_answer(conflict, AgentOutput(conflict_records=["np1", "op1"]), ENTITY_OF, ["np1", "op1"])["passed"]
    with pytest.raises(ValueError):
        score_answer(_item("mystery"), AgentOutput(), ENTITY_OF)


def test_abstention_rates_distinguish_none_from_zero():
    rates = abstention_rates([(True, True), (False, True), (True, False), (False, False), (False, False)])
    assert rates["abstention_rate"] == 0.5
    assert rates["false_abstention_rate"] == pytest.approx(1 / 3)
    assert abstention_rates([(False, False)])["abstention_rate"] is None


def test_routing():
    result = score_routing(["nppes", "open_payments", "part_d"], ["nppes", "open_payments"])
    assert result["recall"] == 1.0 and result["precision"] == pytest.approx(2 / 3) and result["extra"] == ["part_d"]
    missed = score_routing(["nppes"], ["nppes", "part_d"])
    assert missed["recall"] == 0.5 and missed["missed"] == ["part_d"]
    assert score_routing([], ["nppes"])["recall"] == 0.0


def test_canonical_pair_is_unordered_and_rejects_self():
    assert canonical_pair("b", "a") == canonical_pair("a", "b")
    with pytest.raises(ValueError):
        canonical_pair("a", "a")


def test_true_pairs():
    assert true_pairs(ENTITY_OF) == {("np1", "op1"), ("np1", "pd1"), ("op1", "pd1"), ("np2", "op2")}


def test_links_right_and_wrong():
    perfect = score_links(sorted(true_pairs(ENTITY_OF)), ENTITY_OF)
    assert perfect["precision"] == 1.0 and perfect["recall"] == 1.0
    wrong = score_links([("op1", "np1"), ("np1", "np2")], ENTITY_OF)  # one true (reversed order), one false
    assert wrong["tp"] == 1 and wrong["fp"] == 1 and wrong["fn"] == 3


def test_links_duplicates_count_once():
    assert score_links([("np1", "op1"), ("op1", "np1")], ENTITY_OF)["tp"] == 1


def test_links_on_unlabeled_records_are_not_errors():
    # "op9" has no recoverable NPI: a link to it is unknown, so it must not lower precision.
    result = score_links([("np1", "op1"), ("np1", "op9")], ENTITY_OF)
    assert result["precision"] == 1.0 and result["fp"] == 0


def test_link_to_a_decoy_counts_against_precision_with_a_gold_subset():
    entity_of = {**ENTITY_OF, "np9": "E9"}  # E9 has no true links: a namesake decoy
    result = score_links([("np1", "op1"), ("np9", "op1")], entity_of, gold_pairs={("np1", "op1")})
    assert result["fp"] == 1 and result["precision"] == 0.5


def test_no_predictions_gives_none_precision_not_one():
    result = score_links([], ENTITY_OF)
    assert result["precision"] is None and result["recall"] == 0.0


def test_blocking():
    result = blocking_metrics([("np1", "op1"), ("np2", "op2"), ("np1", "np3")], ENTITY_OF, total_possible=15)
    assert result["recall"] == 0.5 and result["reduction_ratio"] == pytest.approx(0.8)


def test_clusters_chain_transitively():
    clusters = clusters_from_links(ENTITY_OF, [("np1", "op1"), ("op1", "np2")])  # one bad link merges E1 and E2
    assert clusters["np1"] == clusters["np2"]
    result = score_clusters(clusters, ENTITY_OF)
    assert result["overmerged_clusters"] == 1 and result["bcubed_precision"] < 1.0


def test_perfect_clusters():
    clusters = clusters_from_links(ENTITY_OF, sorted(true_pairs(ENTITY_OF)))
    result = score_clusters(clusters, ENTITY_OF)
    assert result["bcubed_precision"] == result["bcubed_recall"] == result["purity"] == 1.0
    assert result["overmerged_clusters"] == 0 and result["n_clusters"] == 3


def test_all_singletons_have_perfect_precision_low_recall():
    result = score_clusters({r: r for r in ENTITY_OF}, ENTITY_OF)
    assert result["bcubed_precision"] == 1.0
    assert result["bcubed_recall"] == pytest.approx((3 * (1 / 3) + 2 * (1 / 2) + 1) / 6)


def test_one_big_cluster_has_perfect_recall_low_precision():
    result = score_clusters({r: "all" for r in ENTITY_OF}, ENTITY_OF)
    assert result["bcubed_recall"] == 1.0
    assert result["bcubed_precision"] == pytest.approx((3 * (3 / 6) + 2 * (2 / 6) + 1 / 6) / 6)
    assert result["purity"] == 0.5


def test_clusters_ignore_unlabeled():
    result = score_clusters({"np1": "c", "op1": "c", "op9": "c"}, ENTITY_OF)
    assert result["bcubed_precision"] == 1.0 and result["overmerged_clusters"] == 0


def test_citations():
    pairs = true_pairs(ENTITY_OF)
    good = [Claim("paid $5", [("op1", "np1")])]
    result = score_citations(good, [("np1", "op1")], pairs)
    assert result["precision"] == 1.0 and result["recall"] == 1.0


def test_citations_wrong_missing_and_uncited():
    pairs = true_pairs(ENTITY_OF)
    bad = [Claim("paid $5", [("op1", "np2")]), Claim("is a cardiologist")]
    result = score_citations(bad, [("np1", "op1")], pairs)
    assert result["precision"] == 0.0 and result["recall"] == 0.0
    assert result["unsupported"] == [("np2", "op1")] and result["uncited_claims"] == 1


def test_citations_on_unlabeled_links_are_reported_not_scored():
    result = score_citations([Claim("x", [("np1", "op9")])], [], true_pairs(ENTITY_OF), frozenset({"op9"}))
    assert result["precision"] is None and result["unscored"] == [("np1", "op9")]


def test_ece_perfect_and_overconfident():
    assert expected_calibration_error([1.0, 1.0, 0.0, 0.0], [True, True, False, False])["ece"] == 0.0
    # Says 0.95 but is right half the time.
    assert expected_calibration_error([0.95] * 4, [True, True, False, False])["ece"] == pytest.approx(0.45)


def test_ece_edges():
    assert expected_calibration_error([], [])["ece"] is None
    assert expected_calibration_error([1.0], [True])["bins"][0]["high"] == 1.0  # 1.0 lands in the last bin
    with pytest.raises(ValueError):
        expected_calibration_error([1.2], [True])
    with pytest.raises(ValueError):
        expected_calibration_error([0.5, 0.5], [True])  # lengths must agree


def test_percentile():
    assert percentile([], 0.5) is None
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert percentile([0, 10], 0.95) == pytest.approx(9.5)


def test_repeat_range_reports_spread():
    rows = [{"repeat": 0, "passed": True}, {"repeat": 0, "passed": True},
            {"repeat": 1, "passed": True}, {"repeat": 1, "passed": False}]
    assert repeat_range(rows, pass_rate) == {"mean": 0.75, "min": 0.5, "max": 1.0, "repeats": 2}


def test_by_slice_row_in_several_slices():
    rows = [{"repeat": 0, "passed": True, "slices": ["typo", "nickname"]},
            {"repeat": 0, "passed": False, "slices": ["typo"]}]
    result = by_slice(rows, pass_rate)
    assert result["typo"]["mean"] == 0.5 and result["nickname"]["mean"] == 1.0 and result["typo"]["n_rows"] == 2


def test_pass_rate_skips_unscored():
    assert pass_rate([{"passed": None}]) is None
    assert pass_rate([{"passed": True}, {"passed": None}]) == 1.0


def test_cost_latency():
    rows = [{"latency_s": s, "prompt_tokens": 10, "completion_tokens": 5, "requests": 2} for s in (1.0, 3.0)]
    summary = cost_latency(rows)
    assert summary["latency_p50_s"] == 2.0 and summary["tokens_mean"] == 15 and summary["requests_mean"] == 2
