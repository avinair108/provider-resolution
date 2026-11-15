"""DuckDB views over the identity graph, and the two lookups the question agent will call.

    con = duckdb.connect(); views.register(con, graph_dir, stripped_dir)
    views.find_entities(con, "SMITH", first_name="JOHN", city="AUSTIN")
    views.entity_profile(con, "E1a2b3c4d5e6f")

Everything is local: the views read the git-ignored Parquet files and return Python dicts. Nothing here is served.
"""

from pathlib import Path


def register(con, graph: Path, stripped: Path) -> None:
    con.execute(f"CREATE OR REPLACE VIEW entities AS SELECT * FROM read_parquet('{graph / 'graph_entities.parquet'}')")
    con.execute(f"CREATE OR REPLACE VIEW links AS SELECT * FROM read_parquet('{graph / 'graph_links.parquet'}')")
    con.execute(f"CREATE OR REPLACE VIEW payments AS SELECT * FROM read_parquet('{stripped / 'open_payments_payments.parquet'}')")
    con.execute(f"CREATE OR REPLACE VIEW drugs AS SELECT * FROM read_parquet('{stripped / 'part_d_drugs.parquet'}')")
    # Totals come from the prescriber-level file: CMS drops drug rows under 11 claims, so drug rows undercount (~14%).
    con.execute(f"CREATE OR REPLACE VIEW prescribers AS SELECT * FROM read_parquet('{stripped / 'part_d_prescribers.parquet'}')")
    con.execute(f"""
        CREATE OR REPLACE VIEW members AS
        SELECT e.entity_id, e.record_id, e.source, e.joined_by, r.last_name, r.first_name, r.middle, r.credential,
               r.address, r.city, r.zip, r.specialty
        FROM entities e JOIN (
            SELECT record_id, last_name, first_name, middle_name AS middle, credential, practice_address_1 AS address,
                   practice_city AS city, left(practice_zip, 5) AS zip, array_to_string(taxonomy_codes, ', ') AS specialty
            FROM read_parquet('{stripped / 'nppes.parquet'}')
            UNION ALL
            SELECT record_id, last_name, first_name, middle_name, NULL, address_1, city, left(zip, 5),
                   array_to_string(list_transform(specialties, lambda x: trim(string_split(x, '|')[-1])), '; ')
            FROM read_parquet('{stripped / 'open_payments_recipients.parquet'}')
            UNION ALL
            SELECT record_id, last_name, first_name, middle_initial, credential, address_1, city, left(zip, 5), prescriber_type
            FROM read_parquet('{stripped / 'part_d_prescribers.parquet'}')
        ) r USING (record_id)
    """)
    con.execute("""
        CREATE OR REPLACE VIEW entity_summary AS
        WITH pay AS (SELECT e.entity_id, sum(p.amount_usd) AS paid_usd, count(*) AS n_payments
                     FROM payments p JOIN entities e USING (record_id) GROUP BY 1),
        rx AS (SELECT e.entity_id, sum(p.total_claims) AS claims, sum(p.total_drug_cost) AS drug_cost
               FROM prescribers p JOIN entities e USING (record_id) GROUP BY 1)
        SELECT m.entity_id,
               coalesce(any_value(m.last_name) FILTER (WHERE m.source = 'nppes'), any_value(m.last_name)) AS last_name,
               coalesce(any_value(m.first_name) FILTER (WHERE m.source = 'nppes'), any_value(m.first_name)) AS first_name,
               coalesce(any_value(m.city) FILTER (WHERE m.source = 'nppes'), any_value(m.city)) AS city,
               list(DISTINCT m.source ORDER BY m.source) AS sources, count(*) AS n_records,
               bool_or(m.source = 'nppes') AS in_registry,
               coalesce(any_value(pay.paid_usd), 0) AS paid_usd, coalesce(any_value(pay.n_payments), 0) AS n_payments,
               coalesce(any_value(rx.claims), 0) AS part_d_claims, coalesce(any_value(rx.drug_cost), 0) AS part_d_cost
        FROM members m LEFT JOIN pay USING (entity_id) LEFT JOIN rx USING (entity_id)
        GROUP BY m.entity_id
    """)


def _dicts(cur) -> list[dict]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def find_entities(con, last_name: str, first_name: str | None = None, city: str | None = None, limit: int = 20) -> list[dict]:
    """Entities with a member record matching the name (and city), most-connected first.

    Surnames compare on letters only (GARCIA-LOPEZ = Garcia Lopez); first names on their first word, so "Leslie"
    finds a record stored as "LESLIE AMMABEL". Cities compare on letters only. Case never matters."""
    letters = "regexp_replace(upper({}), '[^A-Z]', '', 'g')"
    first_word = "regexp_replace(split_part(upper(trim({})), ' ', 1), '[^A-Z]', '', 'g')"
    where, params = [f"{letters.format('m.last_name')} = {letters.format('?')}"], [last_name]
    if first_name:
        where.append(f"{first_word.format('m.first_name')} = {first_word.format('?')}")
        params.append(first_name)
    if city:
        where.append(f"{letters.format('m.city')} = {letters.format('?')}")
        params.append(city)
    return _dicts(con.execute(f"""
        SELECT s.* FROM entity_summary s
        WHERE s.entity_id IN (SELECT entity_id FROM members m WHERE {' AND '.join(where)})
        ORDER BY s.n_records DESC, s.entity_id LIMIT {int(limit)}
    """, params))


def entity_profile(con, entity_id: str, top: int = 5) -> dict | None:
    """Everything about one entity: its records, how they were joined, and payments and prescribing across sources."""
    summary = _dicts(con.execute("SELECT * FROM entity_summary WHERE entity_id = ?", [entity_id]))
    if not summary:
        return None
    return {
        **summary[0],
        "records": _dicts(con.execute("SELECT * EXCLUDE (entity_id) FROM members WHERE entity_id = ? ORDER BY source, record_id", [entity_id])),
        "links": _dicts(con.execute("""
            SELECT l.* FROM links l WHERE l.a IN (SELECT record_id FROM entities WHERE entity_id = ?) ORDER BY kind, a
        """, [entity_id])),
        "top_payers": _dicts(con.execute(f"""
            SELECT payer, sum(amount_usd) AS usd, count(*) AS n FROM payments
            WHERE record_id IN (SELECT record_id FROM entities WHERE entity_id = ?) GROUP BY 1 ORDER BY usd DESC LIMIT {int(top)}
        """, [entity_id])),
        "top_drugs": _dicts(con.execute(f"""
            SELECT coalesce(generic_name, brand_name) AS drug, sum(total_claims) AS claims, sum(total_drug_cost) AS cost FROM drugs
            WHERE record_id IN (SELECT record_id FROM entities WHERE entity_id = ?) GROUP BY 1 ORDER BY claims DESC LIMIT {int(top)}
        """, [entity_id])),
    }
