"""Matching evaluation on a tiny synthetic gold, plus a guard that the matcher never reads gold."""

import ast
from pathlib import Path

import duckdb
import pytest

from evals.scorers.linkage import score_links
from matching import evaluate

ROOT = Path(__file__).resolve().parent.parent

# E1: registry np_1, Open Payments op_1, Part D pd_1. E2: np_2 and op_2. E3: op_3 only (no registry record).
# E4: np_4 alone, a namesake decoy. op_9 is unlabeled.
MEMBERSHIP = [("np_1", "nppes", "E1"), ("op_1", "open_payments", "E1"), ("pd_1", "part_d", "E1"),
              ("np_2", "nppes", "E2"), ("op_2", "open_payments", "E2"), ("op_3", "open_payments", "E3"),
              ("np_4", "nppes", "E4")]


@pytest.fixture()
def gold(tmp_path):
    con = duckdb.connect()
    con.execute("CREATE TABLE m (record_id VARCHAR, source VARCHAR, entity_id VARCHAR)")
    con.executemany("INSERT INTO m VALUES (?, ?, ?)", MEMBERSHIP)
    con.execute(f"COPY m TO '{tmp_path / 'membership.parquet'}' (FORMAT parquet)")
    con.execute(f"COPY (SELECT * FROM (VALUES ('E1'), ('E3'), ('E4')) t(entity_id)) TO '{tmp_path / 'focus_entities.parquet'}' (FORMAT parquet)")
    con.execute(f"COPY (SELECT 'op_9' AS record_id, 'open_payments' AS source) TO '{tmp_path / 'unlabeled.parquet'}' (FORMAT parquet)")
    con.execute(f"COPY (SELECT 'op_1' AS record_id, 'open_payments' AS source, 'namesake_same_city' AS slice) TO '{tmp_path / 'slices.parquet'}' (FORMAT parquet)")
    con.execute(f"""COPY (SELECT a.record_id AS left_record, a.source AS left_source, b.record_id AS right_record,
                                 b.source AS right_source, a.entity_id FROM m a JOIN m b ON a.entity_id = b.entity_id AND a.record_id < b.record_id)
                    TO '{tmp_path / 'links.parquet'}' (FORMAT parquet)""")
    evaluate.load_gold(con, tmp_path)
    return con


def candidates(con, rows):
    con.execute("CREATE OR REPLACE TABLE cand (a VARCHAR, b VARCHAR, p DOUBLE)")
    con.executemany("INSERT INTO cand VALUES (least(?, ?), greatest(?, ?), ?)", [(a, b, a, b, p) for a, b, p in rows])


def test_pair_metrics_match_the_tested_python_scorer(gold):
    rows = [("np_1", "op_1", 0.99), ("np_1", "pd_1", 0.6), ("np_2", "op_2", 0.3), ("np_4", "op_1", 0.95),
            ("np_1", "op_9", 0.99)]  # the unlabeled pair must not count either way
    candidates(gold, rows)
    curve = evaluate.pair_curve(gold, "cand")
    for t in (0.2, 0.5, 0.9):
        sql = evaluate.pair_metrics(curve, t)
        entity_of = {r: e for r, _, e in MEMBERSHIP}
        cross = {tuple(sorted((a, b))) for a, sa, ea in MEMBERSHIP for b, sb, eb in MEMBERSHIP if ea == eb and sa != sb and a < b}
        py = score_links([(a, b) for a, b, p in rows if p >= t], entity_of, gold_pairs=cross)
        assert (sql["tp"], sql["fp"]) == (py["tp"], py["fp"]), t
        assert sql["recall"] == pytest.approx(py["recall"])


def test_blocking_recall_counts_true_pairs_ever_proposed(gold):
    candidates(gold, [("np_1", "op_1", 0.01)])
    assert evaluate.pair_metrics(evaluate.pair_curve(gold, "cand"), 0.5)["blocking_recall"] == pytest.approx(1 / 4)


def test_record_resolution_outcomes(gold):
    candidates(gold, [
        ("np_1", "op_1", 0.9), ("np_4", "op_1", 0.4),   # op_1: right registry record on top -> correct
        ("np_1", "pd_1", 0.7), ("np_4", "pd_1", 0.7),   # pd_1: tie at the top -> can't resolve -> missed
        ("np_4", "op_2", 0.95),                          # op_2: wrong registry record on top -> wrong
        ("np_4", "op_3", 0.3),                           # op_3: has no registry record; below threshold -> none, correct
    ])
    evaluate.record_resolution(gold, "cand")
    curve = evaluate.record_curve(gold)
    m = evaluate.record_metrics(curve, 0.5)
    assert (m["correct"], m["wrong"], m["missed"], m["none_correct"], m["linked_without_counterpart"]) == (1, 1, 1, 1, 0)
    low = evaluate.record_metrics(curve, 0.2)
    assert low["linked_without_counterpart"] == 1  # op_3 now links to a decoy
    assert evaluate.record_metrics(curve, 0.5, scope="focus")["n"] == 3  # op_2 belongs to E2, not a focus provider
    assert evaluate.record_metrics(curve, 0.5, group="slice:namesake_same_city")["correct"] == 1


def test_records_with_no_candidates_still_count(gold):
    candidates(gold, [("np_1", "op_1", 0.9)])
    evaluate.record_resolution(gold, "cand")
    m = evaluate.record_metrics(evaluate.record_curve(gold), 0.5)
    assert m["n"] == 4 and m["missed"] == 2 and m["none_correct"] == 1


def test_calibration_and_clusters(gold):
    candidates(gold, [("np_1", "op_1", 0.9), ("np_4", "op_2", 0.9), ("np_1", "pd_1", 0.9)])
    evaluate.record_resolution(gold, "cand")
    cal = evaluate.top_candidate_calibration(gold)
    assert cal["n"] == 3 and cal["ece"] == pytest.approx(abs(2 / 3 - 0.9))
    clusters = evaluate.cluster_metrics(gold, "cand", 0.5)
    assert clusters["overmerged_clusters"] == 1  # np_4 and op_2 joined across two providers


def test_best_threshold_prefers_higher_on_ties():
    assert evaluate.best_threshold(lambda t: 1.0, [0.1, 0.5]) == 0.5


GOLD_TOKENS = ("gold", "membership", "npi_map", "focus_entities", "links.parquet", "slices", "unlabeled", "g_truth")


@pytest.mark.parametrize("module", ["inputs.py", "candidates.py"])
def test_matcher_never_reads_gold(module):
    tree = ast.parse((ROOT / "matching" / module).read_text())
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
                  and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    code_strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                    and id(n) not in docstrings]
    assert not [s for s in code_strings if any(t in s.lower() for t in GOLD_TOKENS)]
    imported = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "evaluate" not in imported and "matching.evaluate" not in modules and "pipeline.build" not in modules
