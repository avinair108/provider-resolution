"""The agent's tools: read-only lookups over the identity graph. They never read gold.

Each tool takes plain arguments and returns a small JSON-able result, trimmed so a step's prompt stays short.
Entities are the graph's (data/<split>/<state>/graph), so a linkage mistake there reaches the agent's answer.
"""

import tomllib
from pathlib import Path

from graph import views

ROOT = Path(__file__).resolve().parent.parent
CARDIOLOGY_CODES = tomllib.loads((ROOT / "config" / "scope.toml").read_text())["nppes"]["taxonomy_codes"]
MAX_RECORDS = 25

SPEC = {
    "find_entities": {"args": {"last_name": "str", "first_name": "str, optional", "city": "str, optional"},
                      "does": "Providers with any record matching the name (and city, if given), across all sources. Exact, case-insensitive."},
    "entity_profile": {"args": {"entity_id": "str"},
                       "does": "One provider: every record in each source (name, city, ZIP, specialty), payment and Part D totals, "
                               "top payers and top drugs."},
    "payer_total": {"args": {"entity_id": "str", "payer": "str"},
                    "does": "Open Payments general payments one payer made to one provider (payer name as written, case-insensitive)."},
    "top_prescribers": {"args": {"drug": "str", "limit": "int, optional (default 5)"},
                        "does": "Providers with the most Medicare Part D claims for a drug (generic or brand name, case-insensitive)."},
    "registry_providers": {"args": {"zip": "str (5 digits)", "cardiology_only": "bool, optional (default true)"},
                           "does": "Providers whose NPPES registry practice address is in a ZIP, with their payment totals."},
}


def _round(row: dict) -> dict:
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}


class Tools:
    def __init__(self, con, graph: Path, stripped: Path):
        self.con = con
        views.register(con, graph, stripped)
        codes = "[" + ", ".join(f"'{c}'" for c in CARDIOLOGY_CODES) + "]"
        con.execute(f"""
            CREATE OR REPLACE VIEW registry AS
            SELECT e.entity_id, n.record_id, left(n.practice_zip, 5) AS zip, list_has_any(n.taxonomy_codes, {codes}) AS cardiology
            FROM read_parquet('{stripped / 'nppes.parquet'}') n JOIN entities e USING (record_id)
        """)

    def call(self, name: str, args: dict):
        if name not in SPEC:
            raise ValueError(f"unknown tool {name!r}; tools are {sorted(SPEC)}")
        allowed = set(SPEC[name]["args"])
        extra = set(args) - allowed
        if extra:
            raise ValueError(f"{name} takes {sorted(allowed)}, not {sorted(extra)}")
        return getattr(self, name)(**args)

    def find_entities(self, last_name: str, first_name: str | None = None, city: str | None = None) -> list[dict]:
        return [_round(r) for r in views.find_entities(self.con, last_name, first_name, city, limit=10)]

    def entity_profile(self, entity_id: str) -> dict | None:
        p = views.entity_profile(self.con, entity_id)
        if p is None:
            return None
        records = [{k: r[k] for k in ("record_id", "source", "first_name", "middle", "last_name", "credential", "city", "zip", "specialty")}
                   for r in p["records"][:MAX_RECORDS]]
        out = {k: p[k] for k in ("entity_id", "last_name", "first_name", "city", "sources", "n_records", "in_registry",
                                 "paid_usd", "n_payments", "part_d_claims", "part_d_cost")}
        return _round({**out, "records": records, "records_shown": len(records),
                       "top_payers": [_round(x) for x in p["top_payers"]], "top_drugs": [_round(x) for x in p["top_drugs"]]})

    def payer_total(self, entity_id: str, payer: str) -> dict:
        usd, n = self.con.execute("""
            SELECT coalesce(sum(amount_usd), 0), count(*) FROM payments
            WHERE upper(payer) = upper(?) AND record_id IN (SELECT record_id FROM entities WHERE entity_id = ?)
        """, [payer, entity_id]).fetchone()
        return {"entity_id": entity_id, "payer": payer, "usd": round(usd, 2), "n_payments": n}

    def top_prescribers(self, drug: str, limit: int = 5) -> list[dict]:
        cur = self.con.execute(f"""
            SELECT e.entity_id, sum(d.total_claims) AS claims
            FROM drugs d JOIN entities e USING (record_id)
            WHERE upper(d.generic_name) = upper(?) OR upper(d.brand_name) = upper(?)
            GROUP BY 1 ORDER BY claims DESC, entity_id LIMIT {max(1, min(int(limit), 20))}
        """, [drug, drug])
        top = cur.fetchall()
        summaries = {r["entity_id"]: r for r in self._summaries([e for e, _ in top])}
        return [_round({**{k: summaries[e][k] for k in ("entity_id", "last_name", "first_name", "city")}, "claims_for_drug": c})
                for e, c in top]

    def registry_providers(self, zip: str, cardiology_only: bool = True) -> list[dict]:
        ids = [e for (e,) in self.con.execute(
            "SELECT DISTINCT entity_id FROM registry WHERE zip = ? AND (cardiology OR NOT ?) ORDER BY 1", [str(zip)[:5], cardiology_only]).fetchall()]
        return [_round({k: r[k] for k in ("entity_id", "last_name", "first_name", "city", "paid_usd", "n_payments")})
                for r in self._summaries(ids)][:50]

    def _summaries(self, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        cur = self.con.execute("SELECT * FROM entity_summary WHERE entity_id IN (SELECT unnest(?::VARCHAR[])) ORDER BY entity_id", [ids])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
