"""Question agent offline: tools over the synthetic graph, the JSON loop, and the grounding check, with a scripted model."""

import json

import duckdb
import pytest

from llm import LLM
from llm.providers import FakeProvider
from qa import agent
from qa.tools import Tools
from tests.test_graph import state  # noqa: F401  (pytest fixture: a built synthetic graph)


@pytest.fixture()
def tools(state):  # noqa: F811
    _, tmp_path, _ = state
    return Tools(duckdb.connect(), tmp_path / "graph", tmp_path / "stripped")


def scripted(*replies):
    """A fake model that gives the next reply each time it's asked, whatever the prompt."""
    it = iter(replies)
    return LLM(FakeProvider(lambda r: json.dumps(next(it))), model="m")


def call(tool, **args):
    return {"action": "call", "tool": tool, "args": args}


def answer(**fields):
    return {"action": "answer", "answer_text": "", "abstained": False, **fields}


def test_tools_over_the_graph(tools):
    found = tools.find_entities("SMITH", "JANE")
    assert len(found) == 1 and found[0]["paid_usd"] == 150.0
    eid = found[0]["entity_id"]
    assert tools.payer_total(eid, "acme")["usd"] == 150.0
    assert tools.top_prescribers("XAMINE")[0]["entity_id"] == eid
    assert [r["entity_id"] for r in tools.registry_providers("78702", cardiology_only=False)] == [eid]
    assert tools.registry_providers("78702") == []  # np_2 isn't a cardiologist
    with pytest.raises(ValueError):
        tools.call("find_entities", {"last_name": "SMITH", "npi": "x"})


def test_grounded_answer_is_accepted(tools):
    eid = tools.find_entities("SMITH", "JANE")[0]["entity_id"]
    llm = scripted(call("find_entities", last_name="SMITH", first_name="JANE"), answer(value="$150.00", providers=[eid]))
    out, trace = agent.run("How much did Jane Smith receive?", llm, tools)
    assert out.value == "$150.00" and not trace.repaired and not trace.final_problems
    assert "open_payments" in out.sources_queried and out.requests == 2


def test_ungrounded_value_gets_one_repair(tools):
    llm = scripted(call("find_entities", last_name="SMITH", first_name="JANE"), answer(value="$999.00"), answer(value="$150.00"))
    out, trace = agent.run("q", llm, tools)
    assert trace.repaired and "doesn't match" in trace.grounding_problems[0] and out.value == "$150.00" and not trace.final_problems


def test_still_ungrounded_after_repair_is_kept_and_flagged(tools):
    llm = scripted(answer(value="$999.00"), answer(value="$998.00"))
    out, trace = agent.run("q", llm, tools)
    assert out.value == "$998.00" and trace.final_problems


def test_unknown_ids_are_ungrounded(tools):
    problems = agent.grounding({"providers": ["Edeadbeef0000"], "conflict_records": ["op_8", "op_9"]}, [])
    assert len(problems) == 2 and "providers not seen" in problems[0] and "record_ids not seen" in problems[1]


def test_bad_tool_call_is_shown_to_the_model_and_the_loop_continues(tools):
    llm = scripted(call("no_such_tool"), answer(abstained=True, answer_text="no data"))
    out, trace = agent.run("q", llm, tools)
    assert trace.steps[0]["kind"] == "error" and out.abstained


def test_running_out_of_steps_abstains(tools):
    llm = scripted(*[call("find_entities", last_name="DOE")] * agent.MAX_STEPS)
    out, trace = agent.run("q", llm, tools)
    assert out.abstained and trace.stopped and len(trace.steps) == agent.MAX_STEPS


def test_a_lifted_native_tool_call_is_read_as_an_action(tools):
    from llm.types import Response
    r = Response(text="", provider="p", model="m", input_tokens=0, output_tokens=0, latency_s=0,
                 meta={"tool_calls": [{"name": "functions.find_entities", "arguments": '{"last_name": "SMITH"}'}]})
    assert agent.native_call(r) == {"action": "call", "tool": "find_entities", "args": {"last_name": "SMITH"}}
    assert agent.native_call(Response(text='{"a": 1}', provider="p", model="m", input_tokens=0, output_tokens=0, latency_s=0)) is None


def test_name_lookup_tolerates_two_word_first_names_and_punctuation(tools):
    assert len(tools.find_entities("smith", "jane marie")) == 1  # first word of the query vs first word of the record
    assert len(tools.find_entities("S-MITH", "JANE", city="aus tin")) == 1


def test_empty_answers_must_abstain_and_conflicts_need_both_sides():
    assert any("abstained" in p for p in agent.grounding({"answer_text": "nothing found", "abstained": False}, []))
    assert not agent.grounding({"answer_text": "nothing found", "abstained": True}, [])
    steps = [{"kind": "call", "result_text": '{"record_id": "op_1"} {"record_id": "np_2"}'}]
    assert any("each side" in p for p in agent.grounding({"conflict_records": ["op_1"]}, steps))
    assert not agent.grounding({"conflict_records": ["op_1", "np_2"]}, steps)
