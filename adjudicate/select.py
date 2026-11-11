"""Choose the records the model sees and their candidates, from Splink scores and the sources' own fields only.

A record is in the band when its best registry candidate scores in [low, high) and cardiology shows up in its own
specialty field or in a candidate's registry taxonomy. Gold says who is a cardiologist; the agent can't, so the band
is chosen the way it would be in production.
"""

import hashlib
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_CANDIDATES = 6
MIN_CANDIDATE_P = 0.001


def _sql_list(values: list[str]) -> str:
    return "[" + ", ".join("'" + v.replace("'", "''") + "'" for v in values) + "]"


def load_sources(con, stripped: Path, scope: dict | None = None) -> None:
    """Raw stripped tables as src_nppes, src_open_payments, src_part_d, plus src_cardiology (record ids)."""
    scope = scope or tomllib.loads((ROOT / "config" / "scope.toml").read_text())
    con.execute(f"CREATE OR REPLACE TABLE src_nppes AS SELECT * FROM read_parquet('{stripped / 'nppes.parquet'}')")
    con.execute(f"CREATE OR REPLACE TABLE src_open_payments AS SELECT * FROM read_parquet('{stripped / 'open_payments_recipients.parquet'}')")
    con.execute(f"CREATE OR REPLACE TABLE src_part_d AS SELECT * FROM read_parquet('{stripped / 'part_d_prescribers.parquet'}')")
    con.execute(f"""
        CREATE OR REPLACE TABLE src_cardiology AS
        SELECT record_id FROM src_nppes WHERE list_has_any(taxonomy_codes, {_sql_list(scope['nppes']['taxonomy_codes'])})
        UNION ALL
        SELECT record_id FROM src_open_payments
        WHERE list_has_any(list_transform(specialties, lambda x: trim(string_split(x, '|')[-1])),
                           {_sql_list(scope['open_payments']['specialties'])})
        UNION ALL
        SELECT record_id FROM src_part_d WHERE prescriber_type IN (SELECT unnest({_sql_list(scope['part_d']['prescriber_types'])}))
    """)


def registry_candidates(con, scores: str) -> None:
    """cand_registry(record_id, registry, p, rank): each non-registry record's registry candidates, best first."""
    con.execute(f"""
        CREATE OR REPLACE TABLE cand_registry AS
        WITH pairs AS (
            SELECT CASE WHEN a LIKE 'np\\_%' ESCAPE '\\' THEN b ELSE a END AS record_id,
                   CASE WHEN a LIKE 'np\\_%' ESCAPE '\\' THEN a ELSE b END AS registry, p
            FROM {scores}
            WHERE (a LIKE 'np\\_%' ESCAPE '\\') <> (b LIKE 'np\\_%' ESCAPE '\\')
        )
        SELECT *, row_number() OVER (PARTITION BY record_id ORDER BY p DESC, registry) AS rank FROM pairs
    """)


def split_of(record_id: str, salt: str = "adjudicate-tune") -> str:
    """A fixed fifth of the band ('tune') is for prompt iteration; the rest ('eval') is what gets reported."""
    return "tune" if int(hashlib.md5(f"{salt}:{record_id}".encode()).hexdigest(), 16) % 5 == 0 else "eval"


def band(con, low: float, high: float, split: str = "all", limit: int | None = None) -> list[dict]:
    """Band records with their candidates (ids and Splink p), in a stable order."""
    rows = con.execute(f"""
        WITH top AS (SELECT record_id, max(p) AS p_top FROM cand_registry GROUP BY 1),
        cardio AS (
            SELECT DISTINCT c.record_id FROM cand_registry c
            WHERE c.p >= {MIN_CANDIDATE_P} AND (c.record_id IN (SELECT record_id FROM src_cardiology)
                                                OR c.registry IN (SELECT record_id FROM src_cardiology))
        )
        SELECT t.record_id, t.p_top,
               list(struct_pack(registry := c.registry, p := c.p) ORDER BY c.rank) AS candidates
        FROM top t JOIN cardio USING (record_id)
        JOIN cand_registry c ON c.record_id = t.record_id AND c.rank <= {MAX_CANDIDATES} AND c.p >= {MIN_CANDIDATE_P}
        WHERE t.p_top >= ? AND t.p_top < ?
        GROUP BY ALL ORDER BY t.record_id
    """, [low, high]).fetchall()
    items = [{"record_id": r, "p_top": p, "candidates": [dict(c) for c in cands]} for r, p, cands in rows]
    if split != "all":
        items = [i for i in items if split_of(i["record_id"]) == split]
    return items[:limit] if limit else items
