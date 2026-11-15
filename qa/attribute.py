"""Attribute each failed answer to the graph or to the agent. Reads gold; run after qa.run.

  python -m qa.attribute runs/<qa run dir>

A failure is the graph's when the right answer wasn't reachable from the graph: the provider is split across graph
entities, or its one graph entity's total differs from gold (records missing or wrongly attached), or a set question
needs a provider the graph split. Otherwise it's the agent's: it abstained, picked the wrong entity or number, or
listed the wrong providers. Prints a markdown table (aggregates only).
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import duckdb

from qa.tools import Tools

ROOT = Path(__file__).resolve().parent.parent


def attribute(run: Path) -> Counter:
    summary = json.loads((run / "summary.json").read_text())
    data = ROOT / "data" / summary["split"] / summary["state"]
    name = "questions.jsonl" if summary.get("set", "eval") == "eval" else f"questions_{summary['set']}.jsonl"
    questions = {q["id"]: q for q in map(json.loads, (data / "questions" / name).read_text().splitlines())}
    con = duckdb.connect()
    gold = dict(con.execute(f"SELECT record_id, entity_id FROM read_parquet('{data / 'gold' / 'membership.parquet'}')").fetchall())
    graph_entities: dict[str, set] = {}
    for eid, rid in con.execute(f"SELECT entity_id, record_id FROM read_parquet('{data / summary['graph'] / 'graph_entities.parquet'}')").fetchall():
        if rid in gold:
            graph_entities.setdefault(gold[rid], set()).add(eid)
    tools = Tools(duckdb.connect(), data / summary["graph"], data / "stripped")
    causes = Counter()
    for sample in range(summary["repeats"]):
        for r in map(json.loads, (run / f"results_{sample}.jsonl").read_text().splitlines()):
            if r.get("status") != "answered" or r["passed"]:
                continue
            q = questions[r["id"]]
            subject, kind = q.get("subject_entity"), r["kind"]
            if r.get("reason") == "request_error":
                label, who = "request failed", "infrastructure"
            elif kind in ("money", "count") and subject:
                ents = graph_entities.get(subject, set())
                key = "paid_usd" if kind == "money" else "part_d_claims"
                totals = [p[key] for p in (tools.entity_profile(e) for e in ents) if p]
                if len(ents) > 1:
                    label, who = f"provider split across {len(ents)} graph entities", "graph"
                elif totals and abs(totals[0] - q["gold_value"]) > max(0.01, 0.005 * abs(q["gold_value"])):
                    label, who = "graph entity's total differs from gold", "graph"
                elif r.get("reason") == "false_abstention":
                    label, who = "abstained on an answerable question", "agent"
                else:
                    label, who = "wrong entity or number", "agent"
            elif kind == "set":
                split = any(len(graph_entities.get(e, ())) > 1 for e in q["gold_entities"])
                label, who = ("a listed provider is split in the graph", "graph") if split else ("wrong providers listed", "agent")
            elif kind == "abstain":
                label, who = "answered when it should have abstained", "agent"
            elif kind == "conflict":
                label, who = "conflict not surfaced", "agent"
            else:
                label, who = r.get("reason") or "failed", "agent"
            causes[(who, r["type"], label)] += 1
    return causes


def to_markdown(causes: Counter, repeats: int) -> str:
    total = sum(causes.values())
    by_who = Counter()
    for (who, _, _), n in causes.items():
        by_who[who] += n
    lines = [f"Failures over {repeats} repeat(s): {total}. " + ", ".join(f"{w} {n}" for w, n in sorted(by_who.items())) + ".", "",
             "| Cause | Question type | Reason | Failures |", "|---|---|---|---|"]
    lines += [f"| {who} | {typ} | {label} | {n} |" for (who, typ, label), n in sorted(causes.items())]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="a qa run directory")
    args = parser.parse_args()
    run = Path(args.run)
    summary = json.loads((run / "summary.json").read_text())
    print(to_markdown(attribute(run), summary["repeats"]))


if __name__ == "__main__":
    main()
