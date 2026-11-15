"""Build the identity graph for one state from Splink scores and adjudication decisions. No gold, no model calls.

  python -m graph.build                                   # Texas, LLM decisions from the default runs
  python -m graph.build --no-llm                          # Splink alone, for comparison
  python -m graph.build --decisions <run_dir> [<run_dir> ...]

Rules, applied in order:
1. Registry link. Each Open Payments or Part D record joins at most one registry (NPPES) record: the adjudicator's
   choice where it decided (including "none"), otherwise Splink's top candidate if unique and >= the record threshold.
2. Cross link. An Open Payments and a Part D record are joined directly only if each is the other's best cross-source
   candidate with p >= the pair threshold, and the join doesn't contradict rule 1: they may not point at two different
   registry records (two NPIs are two people), and a record the adjudicator explicitly placed with no registry record
   isn't pulled under one.
3. Entities. Every registry record is an entity; linked records join it; a cross-linked pair without a registry
   record forms its own entity; anything left alone is a singleton. Because cross links are mutual-best, nothing chains,
   and no entity can hold two registry records.

Outputs in data/<split>/<state>/graph/ (git-ignored): graph_links.parquet, graph_entities.parquet, meta.json.
"""

import argparse
import hashlib
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from adjudicate import select

ROOT = Path(__file__).resolve().parent.parent
SOURCE_OF = """CASE WHEN {c} LIKE 'np\\_%' ESCAPE '\\' THEN 'nppes' WHEN {c} LIKE 'op\\_%' ESCAPE '\\' THEN 'open_payments'
                    ELSE 'part_d' END"""


def load_decisions(con, run_dirs: list[Path], sample: int = 0) -> dict:
    """dec(record_id, status, registry, confidence, flagged, run_id) from adjudication runs. Later runs win on overlap.
    Returns what the runs were, so the graph records its provenance; mixing configurations is refused."""
    con.execute("""CREATE OR REPLACE TABLE dec (record_id VARCHAR, status VARCHAR, registry VARCHAR,
                                                confidence DOUBLE, flagged BOOLEAN, run_id VARCHAR)""")
    configs, runs = set(), []
    for d in run_dirs:
        summary = json.loads((d / "summary.json").read_text())
        # Runs from before summaries recorded effort used their profile's setting, which hasn't changed since.
        effort = summary.get("effort") or tomllib.loads((ROOT / "config" / "llm.toml").read_text())[summary["profile"]].get("effort")
        configs.add((summary["llm"]["provider"], summary["llm"]["model"], summary.get("prompt_sha"), effort,
                     summary.get("show_scores"), summary.get("order", "shuffled"), summary.get("repair", True)))
        rows = []
        for line in (d / f"decisions_{sample}.jsonl").read_text().splitlines():
            x = json.loads(line)
            if x["status"] == "not_run":
                continue
            rows.append((x["record_id"], x["status"], x.get("registry"), x.get("confidence"),
                         bool(x.get("first_contradictions")), summary["run_id"]))
        con.executemany("INSERT INTO dec VALUES (?, ?, ?, ?, ?, ?)", rows)
        runs.append({"run_id": summary["run_id"], "subset": summary["subset"], "records": len(rows)})
    if len(configs) > 1:
        raise SystemExit(f"decision runs use different configurations; rebuild from one: {sorted(map(str, configs))}")
    con.execute("CREATE OR REPLACE TABLE dec AS SELECT * FROM dec QUALIFY row_number() OVER (PARTITION BY record_id ORDER BY run_id DESC) = 1")
    config = dict(zip(["provider", "model", "prompt_sha", "effort", "show_scores", "order", "repair"], next(iter(configs)))) if configs else None
    return {"runs": runs, "config": config, "sample": sample}


def build(con, stripped: Path, scores: str, record_t: float, pair_t: float) -> dict:
    """Create g_links and g_entities. Expects table `dec` (possibly empty) and the Splink table named by `scores`."""
    select.load_sources(con, stripped)
    select.registry_candidates(con, scores)
    con.execute("""
        CREATE OR REPLACE TABLE gr_records AS
        SELECT record_id, 'nppes' AS source FROM src_nppes
        UNION ALL SELECT record_id, 'open_payments' FROM src_open_payments
        UNION ALL SELECT record_id, 'part_d' FROM src_part_d
    """)
    # Rule 1: one registry link (or none) per non-registry record.
    con.execute(f"""
        CREATE OR REPLACE TABLE reg AS
        WITH top AS (
            SELECT record_id, max(p) AS p_top,
                   CASE WHEN count(*) FILTER (WHERE rank_p = 1) = 1 THEN any_value(registry) FILTER (WHERE rank_p = 1) END AS top_registry
            FROM (SELECT *, rank() OVER (PARTITION BY record_id ORDER BY p DESC) AS rank_p FROM cand_registry)
            GROUP BY record_id
        )
        SELECT r.record_id, r.source,
               CASE WHEN d.status = 'decided' THEN d.registry
                    WHEN t.p_top >= {record_t} THEN t.top_registry END AS registry,
               CASE WHEN d.status = 'decided' AND d.registry IS NOT NULL THEN 'llm'
                    WHEN d.status = 'decided' THEN 'llm_none'
                    WHEN d.status IS NOT NULL AND t.p_top >= {record_t} AND t.top_registry IS NOT NULL THEN 'splink_after_llm_fallback'
                    WHEN t.p_top >= {record_t} AND t.top_registry IS NOT NULL THEN 'splink'
                    ELSE 'splink_none' END AS decided_by,
               d.confidence, coalesce(d.flagged, false) AS flagged, d.run_id
        FROM gr_records r
        LEFT JOIN top t USING (record_id)
        LEFT JOIN dec d USING (record_id)
        WHERE r.source <> 'nppes'
    """)
    # Rule 2: mutual-best Open Payments <-> Part D pairs above the pair threshold, minus contradictions.
    con.execute(f"""
        CREATE OR REPLACE TABLE cross_pairs AS
        WITH pairs AS (
            SELECT CASE WHEN a LIKE 'op\\_%' ESCAPE '\\' THEN a ELSE b END AS op,
                   CASE WHEN a LIKE 'op\\_%' ESCAPE '\\' THEN b ELSE a END AS pd, p
            FROM {scores}
            WHERE NOT (a LIKE 'np\\_%' ESCAPE '\\' OR b LIKE 'np\\_%' ESCAPE '\\')
              AND (a LIKE 'op\\_%' ESCAPE '\\') <> (b LIKE 'op\\_%' ESCAPE '\\')
        ),
        ranked AS (
            SELECT *, row_number() OVER (PARTITION BY op ORDER BY p DESC, pd) AS r_op,
                      row_number() OVER (PARTITION BY pd ORDER BY p DESC, op) AS r_pd
            FROM pairs
        )
        SELECT r.op, r.pd, r.p,
               CASE WHEN ro.registry IS NOT NULL AND rp.registry IS NOT NULL AND ro.registry <> rp.registry
                        THEN 'rejected: different registry records'
                    WHEN (ro.registry IS NULL AND ro.decided_by = 'llm_none' AND rp.registry IS NOT NULL)
                      OR (rp.registry IS NULL AND rp.decided_by = 'llm_none' AND ro.registry IS NOT NULL)
                        THEN 'rejected: adjudicator placed one with no registry record'
                    ELSE 'accepted' END AS outcome
        FROM ranked r
        JOIN reg ro ON ro.record_id = r.op
        JOIN reg rp ON rp.record_id = r.pd
        WHERE r.r_op = 1 AND r.r_pd = 1 AND r.p >= {pair_t}
    """)
    # Provenance: every registry link and every mutual-best cross pair, accepted or not.
    con.execute("""
        CREATE OR REPLACE TABLE g_links AS
        SELECT g.record_id AS a, g.registry AS b, 'registry' AS kind, c.p AS p_splink, g.decided_by,
               g.confidence, g.flagged, true AS accepted, 'accepted' AS outcome, g.run_id
        FROM reg g LEFT JOIN cand_registry c ON c.record_id = g.record_id AND c.registry = g.registry
        WHERE g.registry IS NOT NULL
        UNION ALL
        SELECT op, pd, 'cross_source', p, 'splink_mutual_best', NULL, false, outcome = 'accepted', outcome, NULL
        FROM cross_pairs
    """)
    # Rule 3: entities. Root = registry record if any; else the cross-linked partner's registry; else the pair's min id.
    con.execute("""
        CREATE OR REPLACE TABLE g_root AS
        WITH acc AS (SELECT op, pd FROM cross_pairs WHERE outcome = 'accepted'),
        partner AS (SELECT op AS record_id, pd AS other FROM acc UNION ALL SELECT pd, op FROM acc)
        SELECT r.record_id, r.source,
               CASE WHEN r.source = 'nppes' THEN r.record_id
                    WHEN g.registry IS NOT NULL THEN g.registry
                    WHEN go.registry IS NOT NULL THEN go.registry
                    WHEN p.other IS NOT NULL THEN least(r.record_id, p.other)
                    ELSE r.record_id END AS root,
               CASE WHEN r.source = 'nppes' THEN 'registry'
                    WHEN g.registry IS NOT NULL THEN 'registry_link'
                    WHEN go.registry IS NOT NULL THEN 'via_cross_link'
                    WHEN p.other IS NOT NULL THEN 'cross_link_only'
                    ELSE 'singleton' END AS joined_by
        FROM gr_records r
        LEFT JOIN reg g USING (record_id)
        LEFT JOIN partner p USING (record_id)
        LEFT JOIN reg go ON go.record_id = p.other
    """)
    con.execute("""
        CREATE OR REPLACE TABLE g_entities AS
        SELECT 'E' || left(md5(root), 12) AS entity_id, record_id, source, joined_by,
               count(*) OVER (PARTITION BY root) AS entity_size,
               count(*) FILTER (WHERE source = 'nppes') OVER (PARTITION BY root) AS registry_records
        FROM g_root
    """)
    bad = con.execute("SELECT count(DISTINCT entity_id) FROM g_entities WHERE registry_records > 1").fetchone()[0]
    if bad:
        raise RuntimeError(f"{bad} entities hold more than one registry record; the build rules are broken")
    stats = dict(zip(["records", "entities", "multi_source_entities", "largest_entity"], con.execute("""
        SELECT count(*), count(DISTINCT entity_id), count(DISTINCT entity_id) FILTER (WHERE entity_size > 1), max(entity_size)
        FROM g_entities""").fetchone()))
    stats["joined_by"] = dict(con.execute("SELECT joined_by, count(*) FROM g_entities GROUP BY 1 ORDER BY 1").fetchall())
    stats["registry_links_by"] = dict(con.execute("SELECT decided_by, count(*) FROM g_links WHERE kind = 'registry' GROUP BY 1 ORDER BY 1").fetchall())
    stats["cross_pairs"] = dict(con.execute("SELECT outcome, count(*) FROM cross_pairs GROUP BY 1 ORDER BY 1").fetchall())
    return stats


def write(con, out: Path, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for table, name in (("g_links", "graph_links"), ("g_entities", "graph_entities")):
        con.execute(f"COPY {table} TO '{out / (name + '.parquet')}' (FORMAT parquet)")
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def default_decision_runs(split: str, state: str) -> list[Path]:
    """The latest eval and tune runs of the configuration the graph uses by default (gpt-oss-120b on Groq)."""
    runs = ROOT / "runs"
    picks = [sorted(runs.glob(f"*_adjudicate_{split}_{state}_eval_groq_host_eval_v2"))[-1:],
             sorted(runs.glob(f"*_adjudicate_{split}_{state}_tune_abl_ref"))[-1:]]
    return [p for group in picks for p in group]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--state", default="TX")
    parser.add_argument("--baseline", help="Phase 3 run dir under runs/ (default: latest *_baseline)")
    parser.add_argument("--decisions", nargs="*", help="adjudication run dirs (default: latest gpt-oss/Groq eval + tune)")
    parser.add_argument("--sample", type=int, default=0, help="which repeat's decisions to use")
    parser.add_argument("--no-llm", action="store_true", help="Splink alone")
    parser.add_argument("--out", help="output dir (default: data/<split>/<state>/graph, or graph_splink with --no-llm)")
    args = parser.parse_args()

    baseline = ROOT / "runs" / args.baseline if args.baseline else sorted((ROOT / "runs").glob(f"*_matching_{args.split}_{args.state}_baseline"))[-1]
    thresholds = json.loads((baseline / "summary.json").read_text())["results"]["splink"][0]
    record_t, pair_t = thresholds["record_threshold"], thresholds["pair_threshold"]
    data = ROOT / "data" / args.split / args.state
    out = Path(args.out) if args.out else data / ("graph_splink" if args.no_llm else "graph")

    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"CREATE TABLE splink AS SELECT * FROM read_parquet('{baseline / 'splink_0.parquet'}')")
    runs = [] if args.no_llm else [Path(d) if Path(d).is_absolute() else ROOT / d for d in (args.decisions or default_decision_runs(args.split, args.state))]
    if not args.no_llm and not runs:
        raise SystemExit("no adjudication runs found; pass --decisions or --no-llm")
    provenance = load_decisions(con, runs, args.sample)
    stats = build(con, data / "stripped", "splink", record_t, pair_t)
    meta = {"built": datetime.now(timezone.utc).isoformat(timespec="seconds"), "split": args.split, "state": args.state,
            "baseline": baseline.name, "thresholds": {"record": record_t, "pair": pair_t}, "llm": provenance if runs else None,
            "stats": stats, "fingerprint": hashlib.sha256(json.dumps(stats, sort_keys=True, default=str).encode()).hexdigest()[:12]}
    write(con, out, meta)
    print(json.dumps({"out": str(out), **stats}, indent=2, default=str))


if __name__ == "__main__":
    main()
