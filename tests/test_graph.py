"""Identity graph on a tiny synthetic state: every build rule, the lookups, the gold scoring, and the no-gold guard."""

import ast
import re
from pathlib import Path

import duckdb
import pytest

from graph import build, evaluate, views

ROOT = Path(__file__).resolve().parent.parent

# Candidate scores. op_1: Splink prefers np_1, the adjudicator picks np_2. pd_1: Splink alone, np_1.
# op_1~pd_1 are mutual best but point at different registry records -> rejected. op_2~pd_2: no registry records,
# mutual best -> their own entity. op_3: adjudicator said "none"; op_3~pd_3 would pull it under np_2 -> rejected.
SCORES = [("np_1", "op_1", 0.6), ("np_2", "op_1", 0.3), ("np_1", "pd_1", 0.9), ("op_1", "pd_1", 0.95),
          ("op_2", "pd_2", 0.9), ("np_1", "op_3", 0.99), ("np_2", "pd_3", 0.95), ("op_3", "pd_3", 0.9)]
DECISIONS = [("op_1", "decided", "np_2", 0.8, False, "run_x"), ("op_3", "decided", None, 0.7, True, "run_x")]


@pytest.fixture()
def state(tmp_path):
    con = duckdb.connect()
    s = tmp_path / "stripped"
    s.mkdir()
    con.execute(f"""COPY (SELECT * FROM (VALUES
        ('np_1', 'SMITH', 'JOHN', 'A', 'MD', '1 MAIN ST', 'AUSTIN', '78701', ['207RC0000X']),
        ('np_2', 'SMITH', 'JANE', NULL, 'MD', '9 OAK AVE', 'AUSTIN', '78702', ['207R00000X'])
        ) t(record_id, last_name, first_name, middle_name, credential, practice_address_1, practice_city, practice_zip, taxonomy_codes))
        TO '{s / 'nppes.parquet'}' (FORMAT parquet)""")
    con.execute(f"""COPY (SELECT * FROM (VALUES
        ('op_1', 'SMITH', 'JANE', NULL, '9 OAK AVE', 'AUSTIN', '78702', ['Allopathic|Internal Medicine']),
        ('op_2', 'DOE', 'ANN', NULL, '5 ELM', 'DALLAS', '75201', ['Allopathic|Internal Medicine']),
        ('op_3', 'SMITH', 'JOHN', NULL, '1 MAIN ST', 'AUSTIN', '78701', ['Allopathic|Cardiovascular Disease'])
        ) t(record_id, last_name, first_name, middle_name, address_1, city, zip, specialties))
        TO '{s / 'open_payments_recipients.parquet'}' (FORMAT parquet)""")
    con.execute(f"""COPY (SELECT * FROM (VALUES
        ('pd_1', 'SMITH', 'JOHN', 'A', 'MD', '1 MAIN ST', 'AUSTIN', '78701', 'Cardiology'),
        ('pd_2', 'DOE', 'ANN', NULL, 'MD', '5 ELM', 'DALLAS', '75201', 'Internal Medicine'),
        ('pd_3', 'SMITH', 'JANE', NULL, 'MD', '9 OAK AVE', 'AUSTIN', '78702', 'Internal Medicine')
        ) t(record_id, last_name, first_name, middle_initial, credential, address_1, city, zip, prescriber_type))
        TO '{s / 'part_d_prescribers.parquet'}' (FORMAT parquet)""")
    con.execute(f"""COPY (SELECT * FROM (VALUES ('op_1', 'ACME', 100.0), ('op_1', 'ACME', 50.0), ('op_2', 'BETA', 10.0)) t(record_id, payer, amount_usd))
        TO '{s / 'open_payments_payments.parquet'}' (FORMAT parquet)""")
    con.execute(f"""COPY (SELECT * FROM (VALUES ('pd_3', 'X', 'xamine', 30.0, 300.0)) t(record_id, brand_name, generic_name, total_claims, total_drug_cost))
        TO '{s / 'part_d_drugs.parquet'}' (FORMAT parquet)""")
    con.execute("CREATE TABLE splink (a VARCHAR, b VARCHAR, p DOUBLE)")
    con.executemany("INSERT INTO splink VALUES (least(?, ?), greatest(?, ?), ?)", [(a, b, a, b, p) for a, b, p in SCORES])
    con.execute("CREATE TABLE dec (record_id VARCHAR, status VARCHAR, registry VARCHAR, confidence DOUBLE, flagged BOOLEAN, run_id VARCHAR)")
    con.executemany("INSERT INTO dec VALUES (?, ?, ?, ?, ?, ?)", DECISIONS)
    stats = build.build(con, s, "splink", record_t=0.05, pair_t=0.8)
    out = tmp_path / "graph"
    build.write(con, out, {"stats": stats})
    return con, tmp_path, stats


def groups(con) -> set[frozenset]:
    rows = con.execute("SELECT entity_id, list(record_id ORDER BY record_id) FROM g_entities GROUP BY 1").fetchall()
    return {frozenset(members) for _, members in rows}


def test_build_rules(state):
    con, _, stats = state
    assert groups(con) == {frozenset({"np_1", "pd_1"}), frozenset({"np_2", "op_1", "pd_3"}),
                           frozenset({"op_2", "pd_2"}), frozenset({"op_3"})}
    outcomes = dict(con.execute("SELECT a || '~' || b, outcome FROM g_links WHERE kind = 'cross_source'").fetchall())
    assert outcomes == {"op_1~pd_1": "rejected: different registry records", "op_2~pd_2": "accepted",
                        "op_3~pd_3": "rejected: adjudicator placed one with no registry record"}
    by = dict(con.execute("SELECT a, decided_by FROM g_links WHERE kind = 'registry'").fetchall())
    assert by == {"op_1": "llm", "pd_1": "splink", "pd_3": "splink"}
    assert stats["largest_entity"] == 3 and stats["registry_links_by"] == {"llm": 1, "splink": 2}


def test_without_llm_decisions_splink_decides(state):
    con, tmp_path, _ = state
    con.execute("DELETE FROM dec")
    build.build(con, tmp_path / "stripped", "splink", record_t=0.05, pair_t=0.8)
    assert frozenset({"np_1", "op_1", "pd_1", "op_3"}) in groups(con)  # Splink's choices, and the cross links follow


def test_lookups(state):
    con, tmp_path, _ = state
    q = duckdb.connect()
    views.register(q, tmp_path / "graph", tmp_path / "stripped")
    found = views.find_entities(q, "smith", first_name="jane")
    assert len(found) == 1 and found[0]["n_records"] == 3 and found[0]["in_registry"]
    assert found[0]["paid_usd"] == 150.0 and found[0]["part_d_claims"] == 30.0
    profile = views.entity_profile(q, found[0]["entity_id"])
    assert profile["top_payers"][0]["payer"] == "ACME" and profile["top_drugs"][0]["drug"] == "xamine"
    assert {r["source"] for r in profile["records"]} == {"nppes", "open_payments", "part_d"}
    assert views.entity_profile(q, "Enope") is None


def test_graph_scoring_against_gold(state):
    con, tmp_path, _ = state
    gold = tmp_path / "gold"
    gold.mkdir()
    # Truth: op_3 is really np_1's provider (the adjudicator was wrong to say none); everything else as built.
    truth = [("np_1", "nppes", "T1"), ("pd_1", "part_d", "T1"), ("op_3", "open_payments", "T1"),
             ("np_2", "nppes", "T2"), ("op_1", "open_payments", "T2"), ("pd_3", "part_d", "T2"),
             ("op_2", "open_payments", "T3"), ("pd_2", "part_d", "T3")]
    g = duckdb.connect()
    g.execute("CREATE TABLE m (record_id VARCHAR, source VARCHAR, entity_id VARCHAR)")
    g.executemany("INSERT INTO m VALUES (?, ?, ?)", truth)
    g.execute(f"COPY m TO '{gold / 'membership.parquet'}' (FORMAT parquet)")
    g.execute(f"COPY (SELECT 'T1' AS entity_id) TO '{gold / 'focus_entities.parquet'}' (FORMAT parquet)")
    r = evaluate.score_graph(g, tmp_path / "graph", gold)
    assert r["all"]["overmerged_clusters"] == 0 and r["all"]["split_providers"] == 1
    assert r["all"]["registry_attachment"] == pytest.approx(5 / 6)
    assert r["focus"]["records"] == 3 and r["focus"]["registry_attachment"] == pytest.approx(1 / 2)


GOLD_TOKENS = ("gold", "membership", "npi_map", "focus_entities", "slices", "unlabeled", "g_truth", "g_records")
GOLD_LINKS = re.compile(r"(?<![a-z_])links\.parquet")  # gold's file, not graph_links.parquet


@pytest.mark.parametrize("module", ["build.py", "views.py", "query.py"])
def test_graph_build_and_views_never_read_gold(module):
    tree = ast.parse((ROOT / "graph" / module).read_text())
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
                  and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]
    assert not [s for s in strings if any(t in s.lower() for t in GOLD_TOKENS) or GOLD_LINKS.search(s)]
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not modules & {"matching.evaluate", "pipeline.build", "graph.evaluate"}
