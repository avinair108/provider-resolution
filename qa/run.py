"""Run the question agent over a state's question set and score it with the Phase 2 scorers.

  python -m qa.run --limit 10 --repeats 1        # smoke test
  python -m qa.run                               # Texas dev, 3 repeats, gpt-oss-120b on DeepInfra at medium effort

Answers come first, then gold is loaded for scoring (as in Phase 4). Writes runs/<run_id>/ with summary.json,
results_<sample>.jsonl and report.md; the report has aggregates only, never question text or names.
"""

import argparse
import json
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path

import duckdb

from evals.contract import GoldItem
from evals.scorers import score_answer
from evals.scorers.process import score_routing
from llm import LLM, BudgetExceeded, ProviderError, QuotaExhausted
from qa import agent
from qa.tools import Tools

ROOT = Path(__file__).resolve().parent.parent


def load_questions(path: Path) -> list[GoldItem]:
    names = {f.name for f in fields(GoldItem)}
    items = []
    for line in path.read_text().splitlines():
        d = {k: v for k, v in json.loads(line).items() if k in names}
        d["required_links"] = [tuple(p) for p in d.get("required_links", [])]
        items.append(GoldItem(**d))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--state", default="TX")
    parser.add_argument("--graph", default="graph", help="graph dir under data/<split>/<state>/")
    parser.add_argument("--profile", default="or-120b", help="gpt-oss on DeepInfra: Groq lifts gpt-oss's tool calls out of the text")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-cost", type=float, default=1.0)
    parser.add_argument("--set", default="eval", choices=["eval", "tune"], help="tune: iterate here; eval: the reported number")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--types", nargs="*", help="only these question types")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    if args.split != "dev":
        print("Note: running on a held-out split. Do this once, with the method frozen.")

    data = ROOT / "data" / args.split / args.state
    items = load_questions(data / "questions" / ("questions.jsonl" if args.set == "eval" else f"questions_{args.set}.jsonl"))
    if args.types:
        items = [i for i in items if i.question_type in args.types]
    if args.limit:
        by_type = defaultdict(list)
        for i in items:
            by_type[i.question_type].append(i)
        items = [i for group in zip(*by_type.values()) for i in group][:args.limit]  # spread the limit across types
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_qa_{args.split}_{args.state}_{args.set}" + (f"_{args.tag}" if args.tag else "")
    out = ROOT / "runs" / run_id
    out.mkdir(parents=True)
    llm = LLM.from_config(args.profile, max_cost_usd=args.max_cost, effort=args.effort)
    local = threading.local()

    def tools() -> Tools:
        if not hasattr(local, "tools"):
            local.tools = Tools(duckdb.connect(), data / args.graph, data / "stripped")
        return local.tools

    stop = threading.Event()
    stopped = [None]

    def ask(item: GoldItem, sample: int) -> dict:
        if stop.is_set():
            return {"id": item.id, "sample": sample, "status": "not_run"}
        try:
            output, trace = agent.run(item.question, llm, tools(), sample=sample)
        except (QuotaExhausted, BudgetExceeded) as e:
            stopped[0] = f"{type(e).__name__}: {e}"
            stop.set()
            return {"id": item.id, "sample": sample, "status": "not_run"}
        except ProviderError as e:  # one question's request failed for good: record it as a failure, keep going
            print(f"  {item.id}: {str(e)[:160]}", flush=True)
            return {"id": item.id, "sample": sample, "status": "error", "error": str(e)[:300]}
        return {"id": item.id, "sample": sample, "status": "answered", "output": asdict(output),
                "steps": [{k: v for k, v in s.items() if k != "result_text"} for s in trace.steps],
                "repaired": trace.repaired, "grounding_problems": trace.grounding_problems,
                "final_problems": trace.final_problems, "stopped": trace.stopped}

    t0 = time.time()
    answers = {}
    for sample in range(args.repeats):
        with ThreadPoolExecutor(args.workers) as pool:
            answers[sample] = list(pool.map(lambda i: ask(i, sample), items))
        print(f"sample {sample}: {sum(a['status'] == 'answered' for a in answers[sample])}/{len(items)} answered, "
              f"${llm.stats['cost_usd']:.3f}, {time.time() - t0:.0f}s", flush=True)

    # Scoring: the first point at which gold is read.
    con = duckdb.connect()
    entity_of = dict(con.execute(f"SELECT record_id, entity_id FROM read_parquet('{data / 'gold' / 'membership.parquet'}')").fetchall())
    members = defaultdict(list)
    for eid, rid, src in con.execute(f"SELECT entity_id, record_id, source FROM read_parquet('{data / args.graph / 'graph_entities.parquet'}') ORDER BY source, record_id").fetchall():
        members[eid].append((src, rid))
    representative = {eid: next((r for s, r in recs if s == "nppes"), recs[0][1]) for eid, recs in members.items()}
    by_id = {i.id: i for i in items}
    results = {}
    for sample, rows in answers.items():
        scored = []
        for a in rows:
            item = by_id[a["id"]]
            if a["status"] == "error":
                scored.append({"id": item.id, "type": item.question_type, "kind": item.answer_kind, "status": "answered",
                               "passed": False, "reason": "request_error", "routing_recall": 0.0, "abstained": False,
                               "should_abstain": item.answer_kind == "abstain", "steps": 0, "tool_calls": [], "errors": 1,
                               "repaired": False, "ungrounded": False, "stopped": "request error", "requests": 0, "tokens": 0})
                continue
            if a["status"] != "answered":
                scored.append({"id": item.id, "type": item.question_type, "kind": item.answer_kind, "status": a["status"]})
                continue
            o = a["output"]
            from evals.contract import AgentOutput
            output = AgentOutput(**{k: v for k, v in o.items() if k in {f.name for f in fields(AgentOutput)}})
            output.providers = [representative.get(p, p) for p in output.providers]  # graph entity -> one of its records
            s = score_answer(item, output, entity_of, item.disagreeing_records)
            routing = score_routing(output.sources_queried, item.required_sources)
            scored.append({"id": item.id, "type": item.question_type, "kind": item.answer_kind, "status": "answered",
                           "passed": bool(s.get("passed")), "reason": s.get("reason"), "score": s, "routing_recall": routing["recall"],
                           "abstained": output.abstained, "should_abstain": item.answer_kind == "abstain",
                           "steps": len(a["steps"]), "tool_calls": [st["tool"] for st in a["steps"] if st["kind"] == "call"],
                           "errors": sum(1 for st in a["steps"] if st["kind"] == "error"), "repaired": a["repaired"],
                           "ungrounded": bool(a["final_problems"]), "stopped": a["stopped"],
                           "requests": output.requests, "tokens": output.prompt_tokens + output.completion_tokens})
        results[sample] = scored
        with (out / f"results_{sample}.jsonl").open("w") as f:
            for r in scored:
                f.write(json.dumps(r, default=str) + "\n")
        with (out / f"answers_{sample}.jsonl").open("w") as f:  # git-ignored with runs/: may contain names
            for a in rows:
                f.write(json.dumps(a, default=str) + "\n")

    summary = {"run_id": run_id, "split": args.split, "state": args.state, "set": args.set, "graph": args.graph, "profile": args.profile,
               "effort": args.effort, "repeats": args.repeats, "n_questions": len(items), "stopped": stopped[0],
               "prompt_sha": __import__("hashlib").sha256(agent.system_prompt().encode()).hexdigest()[:12],
               "graph_meta": json.loads((data / args.graph / "meta.json").read_text()).get("llm"),
               "llm": llm.summary(), "wall_s": round(time.time() - t0, 1), "aggregate": aggregate(results)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "report.md").write_text(to_markdown(summary))
    print((out / "report.md").read_text())


def _rate(rows, pred):
    rows = list(rows)
    return sum(1 for r in rows if pred(r)) / len(rows) if rows else None


def aggregate(results: dict[int, list[dict]]) -> dict:
    per_sample = []
    for sample, rows in results.items():
        done = [r for r in rows if r["status"] == "answered"]
        types = sorted({r["type"] for r in rows})
        per_sample.append({
            "answered": len(done), "not_run": len(rows) - len(done),
            "pass": _rate(done, lambda r: r["passed"]),
            "by_type": {t: _rate([r for r in done if r["type"] == t], lambda r: r["passed"]) for t in types},
            "abstain_recall": _rate([r for r in done if r["should_abstain"]], lambda r: r["abstained"]),
            "false_abstention": _rate([r for r in done if not r["should_abstain"]], lambda r: r["abstained"]),
            "repaired": _rate(done, lambda r: r["repaired"]), "ungrounded": _rate(done, lambda r: r["ungrounded"]),
            "out_of_steps": _rate(done, lambda r: r["stopped"] is not None),
            "routing_recall": sum(r["routing_recall"] for r in done) / len(done) if done else None,
            "mean_steps": sum(r["steps"] for r in done) / len(done) if done else None,
            "mean_requests": sum(r["requests"] for r in done) / len(done) if done else None,
            "failure_reasons": dict(Counter(f"{r['type']}: {r.get('reason') or 'failed'}" for r in done if not r["passed"])),
        })
    return {"per_sample": per_sample}


def _rng(xs, fmt="{:.0%}"):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "–"
    a, b = fmt.format(min(xs)), fmt.format(max(xs))
    return a if a == b else f"{a}–{b}"


def to_markdown(s: dict) -> str:
    ps = s["aggregate"]["per_sample"]
    llm = s["llm"]
    lines = [f"# Question answering ({s['split']} {s['state']}, {s.get('set', 'eval')} set)", "",
             f"Run `{s['run_id']}`: {s['n_questions']} questions × {s['repeats']} repeat(s); model `{llm['model']}` "
             f"({llm['provider']}, effort {s['effort']}, agent prompt `{s['prompt_sha']}`) over graph `{s['graph']}`. "
             "Ranges over repeats. Aggregates only.", ""]
    if s["stopped"]:
        lines += [f"**Partial run:** {s['stopped']}", ""]
    types = sorted({t for p in ps for t in p["by_type"]})
    lines += ["| Question type | Pass rate |", "|---|---|"]
    lines += [f"| {t} | {_rng([p['by_type'].get(t) for p in ps])} |" for t in types]
    lines += [f"| **all** | **{_rng([p['pass'] for p in ps])}** |", "",
              "| Behaviour | Value |", "|---|---|",
              f"| Abstained when it should (unanswerable, ambiguous) | {_rng([p['abstain_recall'] for p in ps])} |",
              f"| Abstained when it shouldn't | {_rng([p['false_abstention'] for p in ps])} |",
              f"| Answer repaired after the grounding check | {_rng([p['repaired'] for p in ps])} |",
              f"| Accepted answer still ungrounded | {_rng([p['ungrounded'] for p in ps])} |",
              f"| Ran out of steps | {_rng([p['out_of_steps'] for p in ps])} |",
              f"| Required sources touched (routing recall) | {_rng([p['routing_recall'] for p in ps])} |",
              f"| Steps per question | {_rng([p['mean_steps'] for p in ps], '{:.1f}')} |", "",
              "Failures (repeat 0): " + (", ".join(f"{k} ×{v}" for k, v in sorted(ps[0]["failure_reasons"].items())) or "none"), "",
              f"Model calls {llm['calls']:,} ({llm['live_calls']:,} live, {llm['cache_hits']:,} cached); tokens "
              f"{llm['input_tokens']:,} in / {llm['output_tokens']:,} out; cost ${llm['cost_usd']:.2f}; wall {s['wall_s']:.0f}s."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
