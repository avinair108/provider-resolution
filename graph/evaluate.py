"""Score built graphs against gold at the entity level. The only module in graph/ that reads gold.

  python -m graph.evaluate                       # graph_splink vs graph, Texas
  python -m graph.evaluate --out reports/phase5_graph_dev_TX.md

Metrics over labeled records (unlabeled ones have no known truth):
- B-cubed precision, recall and F1, and purity: how well graph entities match true providers.
- Over-merged entities: graph entities holding records of more than one provider.
- Split providers: true providers whose records landed in more than one graph entity.
- Registry attachment: share of Open Payments and Part D records whose entity holds their true registry record
  (or rightly holds none when the provider isn't in the registry).
Reported for all providers and for cardiologists.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import duckdb

from evals.scorers.linkage import score_clusters

ROOT = Path(__file__).resolve().parent.parent


def score_graph(con, graph: Path, gold: Path) -> dict:
    con.execute(f"CREATE OR REPLACE TABLE ge AS SELECT record_id, entity_id, source FROM read_parquet('{graph / 'graph_entities.parquet'}')")
    con.execute(f"CREATE OR REPLACE TABLE gm AS SELECT record_id, source, entity_id AS true_id FROM read_parquet('{gold / 'membership.parquet'}')")
    focus = {r for (r,) in con.execute(f"SELECT entity_id FROM read_parquet('{gold / 'focus_entities.parquet'}')").fetchall()}
    rows = con.execute("SELECT g.record_id, g.entity_id, m.true_id, g.source FROM ge g JOIN gm m USING (record_id)").fetchall()
    out = {}
    for scope in ("all", "focus"):
        keep = [r for r in rows if scope == "all" or r[2] in focus]
        cluster_of = {rid: eid for rid, eid, _, _ in keep}
        entity_of = {rid: tid for rid, _, tid, _ in keep}
        m = score_clusters(cluster_of, entity_of)
        graph_entities_of_true = defaultdict(set)
        for rid, eid, tid, _ in keep:
            graph_entities_of_true[tid].add(eid)
        m["split_providers"] = sum(1 for es in graph_entities_of_true.values() if len(es) > 1)
        m["providers"] = len(graph_entities_of_true)
        # Registry attachment: does each non-registry record's entity hold its provider's registry record?
        registry_entity = {tid: eid for rid, eid, tid, src in keep if src == "nppes"}
        entity_has_registry = {eid for _, eid, _, src in rows if src == "nppes"}
        attach = [(registry_entity[tid] == eid) if tid in registry_entity else (eid not in entity_has_registry)
                  for rid, eid, tid, src in keep if src != "nppes"]
        m["registry_attachment"] = sum(attach) / len(attach) if attach else None
        m["records"] = len(keep)
        out[scope] = m
    return out


def to_markdown(results: dict[str, dict], meta: dict[str, dict]) -> str:
    names = list(results)
    lines = ["# Phase 5: identity graph vs gold (dev TX)", "",
             "Entity-level metrics over labeled records. Aggregates only.", ""]
    for name in names:
        m = meta[name]
        llm = m.get("llm") or {}
        config = llm.get("config") or {}
        lines.append(f"- `{name}`: built {m['built']} from `{m['baseline']}`"
                     + (f", LLM decisions from {', '.join(r['run_id'] for r in llm['runs'])} "
                        f"({config.get('model')}, {config.get('provider')}, effort {config.get('effort')}, repeat {llm['sample']})"
                        if llm else ", Splink alone")
                     + f"; {m['stats']['entities']:,} entities over {m['stats']['records']:,} records.")
    for scope, label in (("focus", "Cardiologists"), ("all", "All providers")):
        lines += ["", f"## {label}", "", "| Graph | B-cubed precision | B-cubed recall | B-cubed F1 | Purity | Over-merged entities | Split providers | Registry attachment |",
                  "|---|---|---|---|---|---|---|---|"]
        for name in names:
            r = results[name][scope]
            lines.append(f"| {name} | {r['bcubed_precision']:.2%} | {r['bcubed_recall']:.2%} | {r['bcubed_f1']:.2%} | {r['purity']:.2%} | "
                         f"{r['overmerged_clusters']:,} | {r['split_providers']:,} of {r['providers']:,} | {r['registry_attachment']:.2%} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--state", default="TX")
    parser.add_argument("--graphs", nargs="*", default=["graph_splink", "graph"])
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.split != "dev" and not args.out:
        print("Note: scoring a held-out graph. Do this once, with the method frozen.")
    data = ROOT / "data" / args.split / args.state
    con = duckdb.connect()
    results, meta = {}, {}
    for name in args.graphs:
        results[name] = score_graph(con, data / name, data / "gold")
        meta[name] = json.loads((data / name / "meta.json").read_text())
    text = to_markdown(results, meta)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
