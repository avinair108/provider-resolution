"""Adjudicator pieces offline: prompt rendering, the verifier, decision application, and the no-gold guard."""

import ast
import json
from pathlib import Path

import duckdb
import pytest

from adjudicate import prompt, select, verify
from adjudicate.run import adjudicate, apply_decisions
from llm import LLM
from llm.providers import FakeProvider

ROOT = Path(__file__).resolve().parent.parent

SOURCE = {"last_name": "GARCIA", "first_name": "BOB", "middle_name": "J", "address_1": "1 MAIN ST", "city": "AUSTIN",
          "state": "TX", "zip": "787011234", "specialties": ["Allopathic & Osteopathic Physicians|Internal Medicine|Cardiovascular Disease"]}
REG_A = {"last_name": "GARCIA-LOPEZ", "first_name": "ROBERT", "middle_name": "JAMES", "practice_address_1": "1 MAIN ST",
         "practice_city": "AUSTIN", "practice_state": "TX", "practice_zip": "78701", "mailing_zip": "78701",
         "mailing_city": "AUSTIN", "taxonomy_codes": ["207RC0000X"], "credential": "MD"}
REG_B = {"last_name": "GARCIA", "first_name": "BOB", "middle_name": "K", "practice_city": "HOUSTON", "practice_zip": "77001",
         "mailing_city": "HOUSTON", "mailing_zip": "77001", "taxonomy_codes": ["207Q00000X"]}
ITEM = {"record_id": "op_0000001", "p_top": 0.6, "candidates": [{"registry": "np_a", "p": 0.6}, {"registry": "np_b", "p": 0.3}]}
ROWS = {"np_a": REG_A, "np_b": REG_B}


def test_render_hides_ids_and_scores_and_shuffles_stably():
    text, mapping = prompt.render(ITEM, SOURCE, ROWS)
    assert "np_a" not in text and "op_0000001" not in text and "0.60" not in text
    assert "(cardiology)" in text and "Cardiovascular Disease" in text
    assert sorted(mapping.values()) == ["np_a", "np_b"]
    assert prompt.render(ITEM, SOURCE, ROWS) == (text, mapping)
    orders = {tuple(prompt.letter_order("op_x", 6, s)) for s in range(10)}
    assert len(orders) > 1  # samples reshuffle
    assert "similarity score 0.60" in prompt.render(ITEM, SOURCE, ROWS, show_scores=True)[0]


def test_actual_relations():
    rel = verify.actual_relations(SOURCE, REG_A)
    assert rel == {"last_name": "compatible", "first_name": "compatible", "middle_initial": "same", "zip": "same", "city": "same"}
    rel_b = verify.actual_relations(SOURCE, REG_B)
    assert rel_b["last_name"] == "same" and rel_b["middle_initial"] == "different" and rel_b["zip"] == "different"


def test_check_accepts_a_true_reply_and_catches_contradictions():
    mapping = {"A": "np_a", "B": "np_b"}
    good = {"choice": "a", "confidence": 0.9, "evidence": [{"field": "zip", "relation": "same"},
                                                            {"field": "last_name", "relation": "compatible"}], "reason": "r"}
    v = verify.check(good, mapping, SOURCE, ROWS)
    assert v.ok and v.registry == "np_a" and v.choice == "A"
    bad = {**good, "choice": "B", "evidence": [{"field": "zip", "relation": "same"}, {"field": "middle_initial", "relation": "same"}]}
    v = verify.check(bad, mapping, SOURCE, ROWS)
    assert not v.ok and len(v.contradictions) == 2
    wording = {**good, "evidence": [{"field": "last_name", "relation": "same"}, {"field": "middle_initial", "relation": "missing"}]}
    assert verify.check(wording, mapping, SOURCE, ROWS).ok  # compatible vs same is wording, not a contradiction
    nearby = {**good, "choice": "B", "evidence": [{"field": "zip", "relation": "compatible"}]}
    assert verify.check(nearby, mapping, SOURCE, ROWS).ok  # a nearby ZIP can't be refuted from the records
    assert verify.actual_relations({"first_name": "MUHAMMAD"}, {"first_name": "MOHAMMED"})["first_name"] == "compatible"
    assert verify.actual_relations({"first_name": "SARAH"}, {"first_name": "JOHN"})["first_name"] == "different"


def test_check_format_errors():
    mapping = {"A": "np_a"}
    assert verify.check(None, mapping, SOURCE, ROWS, "Expecting value").errors
    assert verify.check({"choice": "C", "confidence": 0.5}, mapping, SOURCE, ROWS).errors
    assert verify.check({"choice": "A", "confidence": 2}, mapping, SOURCE, ROWS).errors
    assert verify.check({"choice": "A", "confidence": 0.5, "evidence": [{"field": "shoe", "relation": "same"}]}, mapping, SOURCE, ROWS).errors
    none = verify.check({"choice": "None", "confidence": 0.7, "evidence": []}, mapping, SOURCE, ROWS)
    assert none.ok and none.choice == "none" and none.registry is None


def test_adjudicate_repairs_once_then_falls_back(tmp_path):
    replies = iter(["not json", json.dumps({"choice": "none", "confidence": 0.8, "evidence": []})])
    llm = LLM(FakeProvider(lambda r: next(replies)), model="m")
    d = adjudicate(llm, ITEM, SOURCE, ROWS, sample=0, show_scores=False)
    assert d["status"] == "decided" and d["attempts"] == 2 and not d["first_try_ok"] and d["registry"] is None
    llm = LLM(FakeProvider(lambda r: "still not json"), model="m")
    d = adjudicate(llm, ITEM, SOURCE, ROWS, sample=0, show_scores=False)
    assert d["status"] == "fallback" and d["attempts"] == 2


def test_apply_decisions():
    con = duckdb.connect()
    con.execute("CREATE TABLE s (a VARCHAR, b VARCHAR, p DOUBLE)")
    con.executemany("INSERT INTO s VALUES (?, ?, ?)", [
        ("np_a", "op_1", 0.6), ("np_b", "op_1", 0.3), ("op_1", "pd_1", 0.9),   # op_1 decided: np_b
        ("np_a", "op_2", 0.4),                                                    # op_2 decided: none
        ("np_a", "op_3", 0.5)])                                                   # op_3 fell back
    apply_decisions(con, "s", "c", [{"record_id": "op_1", "status": "decided", "registry": "np_b"},
                                    {"record_id": "op_2", "status": "decided", "registry": None},
                                    {"record_id": "op_3", "status": "fallback", "registry": None}])
    got = dict(((a, b), p) for a, b, p in con.execute("SELECT * FROM c").fetchall())
    assert got == {("np_a", "op_1"): 0.0, ("np_b", "op_1"): 1.0, ("op_1", "pd_1"): 0.9, ("np_a", "op_2"): 0.0, ("np_a", "op_3"): 0.5}


def test_band_selection_without_gold(tmp_path):
    con = duckdb.connect()
    con.execute("CREATE TABLE src_cardiology AS SELECT 'np_a' AS record_id")
    con.execute("CREATE TABLE s (a VARCHAR, b VARCHAR, p DOUBLE)")
    con.executemany("INSERT INTO s VALUES (?, ?, ?)", [
        ("np_a", "op_1", 0.5), ("np_b", "op_1", 0.2),   # in band, a candidate is a cardiologist
        ("np_b", "op_2", 0.5),                           # in band, no cardiology anywhere
        ("np_a", "op_3", 0.999)])                        # confident
    select.registry_candidates(con, "s")
    items = select.band(con, 0.01, 0.99)
    assert [i["record_id"] for i in items] == ["op_1"] and [c["registry"] for c in items[0]["candidates"]] == ["np_a", "np_b"]


GOLD_TOKENS = ("gold", "membership", "npi_map", "focus_entities", "links.parquet", "slices", "unlabeled", "g_truth", "g_records")


@pytest.mark.parametrize("module", ["select.py", "prompt.py", "verify.py"])
def test_agent_modules_never_read_gold(module):
    tree = ast.parse((ROOT / "adjudicate" / module).read_text())
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
                  and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]
    assert not [s for s in strings if any(t in s.lower() for t in GOLD_TOKENS)]
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not modules & {"matching.evaluate", "pipeline.build"}


def test_order_ablations():
    assert prompt.letter_order("op_x", 4, 0, "splink") == [0, 1, 2, 3]
    assert prompt.letter_order("op_x", 4, 0, "reversed") == [3, 2, 1, 0]
    _, mapping = prompt.render(ITEM, SOURCE, ROWS, order="reversed")
    assert mapping == {"A": "np_b", "B": "np_a"}


def test_no_repair_keeps_a_contradicted_first_answer():
    reply = json.dumps({"choice": "B", "confidence": 0.9, "evidence": [{"field": "zip", "relation": "same"}]})
    _, mapping = prompt.render(ITEM, SOURCE, ROWS, order="splink")  # B = np_b, whose ZIP differs
    llm = LLM(FakeProvider(lambda r: reply), model="m")
    d = adjudicate(llm, ITEM, SOURCE, ROWS, sample=0, show_scores=False, order="splink", repair=False)
    assert d["status"] == "decided" and d["registry"] == "np_b" and d["attempts"] == 1 and d["first_contradictions"]
    d = adjudicate(llm, ITEM, SOURCE, ROWS, sample=0, show_scores=False, order="splink", repair=True)
    assert d["status"] == "fallback" and d["attempts"] == 2


def test_finetune_targets_pass_the_verifier():
    import json as _json
    from adjudicate import verify
    from finetune.prepare import target
    src = {"last_name": "SMITH", "first_name": "BOB", "middle_name": "J", "city": "AUSTIN", "zip": "787011234"}
    cand = {"last_name": "SMITH", "first_name": "ROBERT", "middle_name": "K", "practice_city": "DALLAS", "practice_zip": "75201"}
    reply = _json.loads(target("B", src, cand))
    v = verify.check(reply, {"A": "np_1", "B": "np_2"}, src, {"np_1": {}, "np_2": cand})
    assert v.ok and v.choice == "B" and {e["field"] for e in reply["evidence"]} >= {"last_name", "city"}
    none = _json.loads(target("none", src, None))
    assert verify.check(none, {"A": "np_1"}, src, {"np_1": cand}).ok and none["evidence"] == []
