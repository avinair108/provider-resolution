"""Score a candidate table <name>(a, b, p) against hidden-NPI gold. The only matching module that reads gold.

Two views, both computed from 1,000 probability buckets so any threshold is free:
- Pairs: precision and recall of cross-source links, per source pair.
- Records: for each Open Payments or Part D record, the registry record it would be resolved to (the top candidate,
  if unique and above the threshold) or none. This is the decision the agent relies on, and the one slices describe.
Unlabeled records (no NPI) are left out of both: their truth is unknown, not "no match".
"""

from pathlib import Path

BUCKETS = 1000


def load_gold(con, gold: Path) -> None:
    con.execute(f"CREATE OR REPLACE TABLE g_membership AS SELECT * FROM read_parquet('{gold / 'membership.parquet'}')")
    con.execute(f"CREATE OR REPLACE TABLE g_focus AS SELECT * FROM read_parquet('{gold / 'focus_entities.parquet'}')")
    con.execute(f"CREATE OR REPLACE TABLE g_unlabeled AS SELECT * FROM read_parquet('{gold / 'unlabeled.parquet'}')")
    con.execute(f"CREATE OR REPLACE TABLE g_slices AS SELECT * FROM read_parquet('{gold / 'slices.parquet'}')")
    con.execute(f"""
        CREATE OR REPLACE TABLE g_truth AS
        SELECT least(left_record, right_record) AS a, greatest(left_record, right_record) AS b,
               least(left_source, right_source) || '~' || greatest(left_source, right_source) AS source_pair,
               entity_id IN (SELECT entity_id FROM g_focus) AS focus
        FROM read_parquet('{gold / 'links.parquet'}') WHERE left_source <> right_source
    """)
    # Each non-registry record's true registry record, if its provider has one (at most one, checked at build time).
    con.execute("""
        CREATE OR REPLACE TABLE g_records AS
        SELECT m.record_id, m.source, n.record_id AS true_registry,
               m.entity_id IN (SELECT entity_id FROM g_focus) AS focus
        FROM g_membership m
        LEFT JOIN (SELECT entity_id, record_id FROM g_membership WHERE source = 'nppes') n USING (entity_id)
        WHERE m.source <> 'nppes'
    """)


def _source_of(column: str) -> str:
    return (f"CASE WHEN {column} LIKE 'np\\_%' ESCAPE '\\' THEN 'nppes' "
            f"WHEN {column} LIKE 'op\\_%' ESCAPE '\\' THEN 'open_payments' ELSE 'part_d' END")


def _bucket(p: str) -> str:
    return f"least(CAST(floor({p} * {BUCKETS}) AS INTEGER), {BUCKETS - 1})"


def pair_curve(con, name: str) -> dict:
    """Per scope ('all', 'focus') and source pair: counts of true and false candidate pairs per bucket, plus totals."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE scored AS
        SELECT c.a, c.b, c.p, {_source_of('c.a')} || '~' || {_source_of('c.b')} AS source_pair,
               t.a IS NOT NULL AS hit,
               coalesce(t.focus, ma.entity_id IN (SELECT entity_id FROM g_focus) OR mb.entity_id IN (SELECT entity_id FROM g_focus)) AS focus
        FROM {name} c
        LEFT JOIN g_truth t USING (a, b)
        JOIN g_membership ma ON ma.record_id = c.a
        JOIN g_membership mb ON mb.record_id = c.b
    """)  # the inner joins to membership drop pairs touching unlabeled records
    rows = con.execute(f"""
        SELECT source_pair, focus, {_bucket('p')} AS bucket, count(*) FILTER (hit), count(*) FILTER (NOT hit)
        FROM scored GROUP BY ALL
    """).fetchall()
    totals = con.execute("SELECT source_pair, focus, count(*) FROM g_truth GROUP BY ALL").fetchall()
    found = con.execute("SELECT source_pair, focus, count(*) FROM scored WHERE hit GROUP BY ALL").fetchall()
    return {"rows": rows, "totals": totals, "found": found}


def pair_metrics(curve: dict, threshold: float, scope: str = "all", source_pair: str | None = None) -> dict:
    """Precision and recall of pairs with p >= threshold. scope 'focus' counts pairs touching a cardiologist."""
    keep = lambda sp, f: (source_pair is None or sp == source_pair) and (scope == "all" or f)
    cut = int(threshold * BUCKETS) if threshold < 1 else BUCKETS - 1
    tp = sum(h for sp, f, b, h, _ in curve["rows"] if keep(sp, f) and b >= cut)
    fp = sum(m for sp, f, b, _, m in curve["rows"] if keep(sp, f) and b >= cut)
    total = sum(n for sp, f, n in curve["totals"] if keep(sp, f))
    blocked = sum(n for sp, f, n in curve["found"] if keep(sp, f))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / total if total else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    return {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp,
            "true_pairs": total, "blocking_recall": blocked / total if total else None}


def record_resolution(con, name: str) -> None:
    """Top registry candidate per non-registry record, with ties marked: a tie can't be resolved and abstains."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE record_top AS
        WITH registry_pairs AS (
            SELECT CASE WHEN a LIKE 'np\\_%' ESCAPE '\\' THEN b ELSE a END AS record_id,
                   CASE WHEN a LIKE 'np\\_%' ESCAPE '\\' THEN a ELSE b END AS registry, p
            FROM {name}
            WHERE (a LIKE 'np\\_%' ESCAPE '\\') <> (b LIKE 'np\\_%' ESCAPE '\\')
        ),
        ranked AS (
            SELECT *, max(p) OVER (PARTITION BY record_id) AS p_max,
                   count(*) FILTER (WHERE p >= 1e-12) OVER (PARTITION BY record_id, p) AS n_at_p
            FROM registry_pairs
        )
        SELECT r.record_id, r.source, r.focus, r.true_registry,
               max(k.p) AS p_top,
               CASE WHEN count(k.registry) FILTER (WHERE k.p = k.p_max) = 1
                    THEN any_value(k.registry) FILTER (WHERE k.p = k.p_max) END AS top_registry,
               bool_or(k.registry = r.true_registry) AS true_in_candidates
        FROM g_records r LEFT JOIN ranked k USING (record_id)
        GROUP BY ALL
    """)


def record_curve(con, slices: bool = True) -> list[tuple]:
    """Rows: (group, scope_focus, bucket_of_p_top, outcome, count). Groups are each source and each slice."""
    outcome = """CASE
        WHEN top_registry IS NULL THEN CASE WHEN true_registry IS NULL THEN 'none_correct' ELSE 'tie_or_no_candidate' END
        WHEN top_registry = true_registry THEN 'correct'
        WHEN true_registry IS NULL THEN 'linked_without_counterpart'
        ELSE 'wrong' END"""
    base = f"SELECT record_id, source, focus, true_registry IS NOT NULL AS has_counterpart, coalesce({_bucket('p_top')}, -1) AS bucket, {outcome} AS outcome FROM record_top"
    by_source = con.execute(f"SELECT source, focus, has_counterpart, bucket, outcome, count(*) FROM ({base}) GROUP BY ALL").fetchall()
    if not slices:
        return by_source
    by_slice = con.execute(f"""
        SELECT 'slice:' || s.slice, x.focus, x.has_counterpart, x.bucket, x.outcome, count(*)
        FROM ({base}) x JOIN g_slices s USING (record_id) GROUP BY ALL
    """).fetchall()
    return by_source + by_slice


def record_metrics(curve: list[tuple], threshold: float, group: str | None = None, scope: str = "all") -> dict:
    """At a threshold, a record resolves to its top candidate if unique and p_top >= threshold; otherwise none.

    accuracy: right registry record, or rightly none. precision: share of resolved records resolved correctly.
    recall: share of records with a counterpart that resolved to it."""
    cut = int(threshold * BUCKETS) if threshold < 1 else BUCKETS - 1
    counts = {"correct": 0, "wrong": 0, "linked_without_counterpart": 0, "missed": 0, "none_correct": 0}
    for g, focus, has_cp, bucket, outcome, n in curve:
        if group is None and g.startswith("slice:"):
            continue
        if group is not None and g != group:
            continue
        if scope == "focus" and not focus:
            continue
        above = bucket >= cut
        if outcome in ("correct", "wrong", "linked_without_counterpart") and above:
            counts[outcome] += n
        elif has_cp:
            counts["missed"] += n
        else:
            counts["none_correct"] += n
    n = sum(counts.values())
    resolved = counts["correct"] + counts["wrong"] + counts["linked_without_counterpart"]
    with_cp = counts["correct"] + counts["wrong"] + counts["missed"]
    return {
        "threshold": threshold, "n": n, **counts,
        "accuracy": (counts["correct"] + counts["none_correct"]) / n if n else None,
        "precision": counts["correct"] / resolved if resolved else None,
        "recall": counts["correct"] / with_cp if with_cp else None,
    }


def top_candidate_calibration(con, n_bins: int = 10) -> dict:
    """Is a top candidate at p = 0.8 right about 80% of the time? Over records whose top candidate is unique."""
    rows = con.execute("""
        SELECT p_top, top_registry = coalesce(true_registry, '') FROM record_top WHERE top_registry IS NOT NULL
    """).fetchall()
    from evals.scorers.linkage import expected_calibration_error

    return expected_calibration_error([min(max(float(p), 0.0), 1.0) for p, _ in rows], [bool(ok) for _, ok in rows], n_bins)


def cluster_metrics(con, name: str, threshold: float) -> dict:
    """Accept every pair at or above the threshold, close transitively, and score the clusters against gold."""
    from evals.scorers.linkage import clusters_from_links, score_clusters

    entity_of = dict(con.execute("SELECT record_id, entity_id FROM g_membership").fetchall())
    links = con.execute(f"SELECT a, b FROM {name} WHERE p >= ?", [threshold]).fetchall()
    return score_clusters(clusters_from_links(entity_of, links), entity_of)


def best_threshold(metric, grid=None) -> float:
    grid = grid or [round(0.05 * i, 2) for i in range(1, 20)] + [0.97, 0.99, 0.995, 0.999]
    return max(grid, key=lambda t: (metric(t) or 0.0, t))
