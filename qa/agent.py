"""The question agent: the model chooses tools and proposes an answer; code checks the answer before accepting it.

Each step is one request through the llm layer (leak gate, cache, budgets): the system prompt, the question and the
transcript so far. The model replies with JSON, either {"action": "call", ...} or {"action": "answer", ...}. That keeps
the agent provider-neutral (no vendor tool-calling format) and every step cacheable.

Before an answer is accepted, code checks that it is grounded and complete:
- a numeric value must equal a number that appeared in a tool result (to the cent, or within 0.5%);
- every entity it lists and every record it cites must have appeared in a tool result;
- an answer with nothing in it must say it abstains, and a reported disagreement must cite both sides.
A failed check gets one repair, with the reason; a second failure is recorded as such and the answer is kept.
"""

import json
import re
from dataclasses import dataclass, field

from evals.contract import AgentOutput
from evals.numbers import extract_numbers
from qa.tools import SPEC

MAX_STEPS = 8
TOOL_SOURCES = {"payer_total": {"open_payments"}, "top_prescribers": {"part_d"}, "registry_providers": {"nppes"}}
MAX_RESULT_CHARS = 6000

SYSTEM = """You answer questions about US healthcare providers in one state, using three CMS datasets for 2024: the \
NPPES provider registry, Open Payments general payments (program year 2024), and Medicare Part D prescribing (2024). \
Nothing else is loaded: no other years, no Medicare Part B, no other states.

Records from the three datasets are already linked into providers ("entities"). An entity's totals cover all of its \
records, so use them rather than adding records up yourself. Always work from tool results, never from memory.

Tools:
{tools}

Rules:
- Identify the provider the question means before answering. Use every detail it gives (name, city, address, \
specialty, middle initial). If several providers fit and the question doesn't tell them apart, don't pick one: \
abstain and say the question is ambiguous.
- If no provider matches, or the question needs data that isn't loaded, abstain and say why.
- If asked whether sources agree and they don't, list in conflict_records the record_ids on every side of the \
disagreement: at least one record for each differing value (e.g. a record that says Tyler and one that says Crockett).
- For a list of providers, give their entity_ids, each once.
- Give money as dollars with cents, counts as whole numbers.

Reply with one JSON object and nothing else. Either call a tool:
{{"action": "call", "tool": "<name>", "args": {{...}}}}
or answer:
{{"action": "answer", "answer_text": "<one or two sentences>", "value": "<the single number asked for, or null>",
 "providers": [<entity_ids, for list questions>], "conflict_records": [<record_ids that disagree, for conflict questions>],
 "evidence_records": [<record_ids your answer relies on>], "abstained": <true or false>}}"""


def system_prompt() -> str:
    tools = "\n".join(f"- {name}({', '.join(f'{a}: {t}' for a, t in s['args'].items())}): {s['does']}" for name, s in SPEC.items())
    return SYSTEM.format(tools=tools)


@dataclass
class Trace:
    steps: list[dict] = field(default_factory=list)
    repaired: bool = False
    grounding_problems: list[str] = field(default_factory=list)
    stopped: str | None = None  # why the loop ended without an accepted answer
    final_problems: list[str] = field(default_factory=list)  # grounding problems left in the accepted answer


def _transcript(question: str, steps: list[dict], note: str | None) -> str:
    parts = [f"QUESTION: {question}"]
    for i, s in enumerate(steps, 1):
        if s["kind"] == "call":
            parts.append(f"STEP {i}: you called {s['tool']} {json.dumps(s['args'])}\nRESULT: {s['result_text']}")
        else:
            parts.append(f"STEP {i}: your reply couldn't be used: {s['problem']}")
    if note:
        parts.append(note)
    parts.append(f"You have {MAX_STEPS - len(steps)} step(s) left. Reply with your next action as JSON.")
    return "\n\n".join(parts)


def _seen(steps: list[dict]) -> tuple[set[str], set[str], list[float]]:
    """Entity ids, record ids and numbers that appeared in tool results so far."""
    text = " ".join(s.get("result_text", "") for s in steps if s["kind"] == "call")
    entities = set(re.findall(r"\bE[0-9a-f]{12}\b", text))
    records = set(re.findall(r"\b(?:np|op|pd)_[0-9]+\b", text))
    numbers = [n.value for n in extract_numbers(text.replace('"', " "))]
    return entities, records, numbers


def grounding(reply: dict, steps: list[dict]) -> list[str]:
    entities, records, numbers = _seen(steps)
    problems = []
    if reply.get("value") not in (None, "", "null"):
        asserted = extract_numbers(str(reply["value"]))
        if len(asserted) != 1:
            problems.append(f"value must be exactly one number, got {reply['value']!r}")
        elif not any(abs(asserted[0].value - n) <= max(0.005, 0.005 * abs(n)) for n in numbers):
            problems.append(f"value {reply['value']} doesn't match any number in the tool results")
    unknown = [p for p in reply.get("providers") or [] if p not in entities]
    if unknown:
        problems.append(f"providers not seen in any tool result: {unknown[:5]}")
    has_content = reply.get("value") not in (None, "", "null") or bool(reply.get("providers")) or bool(reply.get("conflict_records"))
    if not has_content and not reply.get("abstained"):
        problems.append("the answer has no value, providers or conflict_records; if you can't answer, set abstained to true")
    if len(reply.get("conflict_records") or []) == 1:
        problems.append("a disagreement needs records from each side: list at least one record for every differing value")
    cited = list(reply.get("conflict_records") or []) + list(reply.get("evidence_records") or [])
    unknown = [r for r in cited if r not in records]
    if unknown:
        problems.append(f"record_ids not seen in any tool result: {unknown[:5]}")
    return problems


def native_call(response) -> dict | None:
    """A host that lifted the model's native tool call out of the text (Groq with gpt-oss): read it as our action."""
    calls = response.meta.get("tool_calls") if not response.text.strip() else None
    if not calls:
        return None
    name, raw = calls[0].get("name") or "", calls[0].get("arguments") or "{}"
    name = name.split(".")[-1]  # "functions.find_entities" -> "find_entities"
    args = json.loads(raw) if isinstance(raw, str) else raw
    if name == "answer":
        return {"action": "answer", **args}
    return {"action": "call", "tool": name, "args": args}


def run(question: str, llm, tools, sample: int = 0) -> tuple[AgentOutput, Trace]:
    trace, note = Trace(), None
    system = system_prompt()
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0, "latency_s": 0.0}
    sources = set()
    while len(trace.steps) < MAX_STEPS:
        r = llm.complete(llm.request(user=_transcript(question, trace.steps, note), system=system, json=True, sample=sample))
        usage["prompt_tokens"] += r.input_tokens
        usage["completion_tokens"] += r.output_tokens
        usage["requests"] += 1
        usage["latency_s"] += r.latency_s
        note = None
        try:
            reply = native_call(r) or r.parsed()
        except ValueError as e:
            trace.steps.append({"kind": "error", "problem": f"not a JSON object ({str(e)[:80]})"})
            continue
        if reply.get("action") == "call":
            name, args = reply.get("tool"), reply.get("args") or {}
            try:
                result = tools.call(name, args)
                text = json.dumps(result, default=str)
            except Exception as e:  # a bad tool call is the model's mistake; show it and continue
                trace.steps.append({"kind": "error", "problem": f"tool call failed: {str(e)[:200]}"})
                continue
            if len(text) > MAX_RESULT_CHARS:
                text = text[:MAX_RESULT_CHARS] + " ...(truncated)"
            trace.steps.append({"kind": "call", "tool": name, "args": args, "result_text": text})
            sources |= set(re.findall(r'"(nppes|open_payments|part_d)"', text))
            sources |= TOOL_SOURCES.get(name, set())
            continue
        if reply.get("action") != "answer":
            trace.steps.append({"kind": "error", "problem": "action must be \"call\" or \"answer\""})
            continue
        problems = grounding(reply, trace.steps)
        if problems and not trace.repaired:
            trace.repaired = True
            trace.grounding_problems = problems
            note = ("Your answer was rejected by an automatic check: " + "; ".join(problems) +
                    ". Use only values and ids from tool results; call another tool if you need to.")
            continue
        trace.grounding_problems = problems if problems else trace.grounding_problems
        out = AgentOutput(answer_text=str(reply.get("answer_text", ""))[:1000],
                          value=None if reply.get("value") in (None, "", "null") else str(reply["value"]),
                          providers=[str(p) for p in reply.get("providers") or []],
                          abstained=bool(reply.get("abstained")),
                          conflict_records=[str(x) for x in reply.get("conflict_records") or []],
                          evidence_records=[str(x) for x in reply.get("evidence_records") or []],
                          sources_queried=sorted(sources), **usage)
        trace.final_problems = problems
        return out, trace
    trace.stopped = f"no accepted answer in {MAX_STEPS} steps"
    return AgentOutput(answer_text="", abstained=True, sources_queried=sorted(sources), **usage), trace
