"""Standardize the stripped tables into one comparable shape per source.

Reads data/<split>/<state>/stripped only. Nothing in the matching package may read gold: the evaluation does that.
"""

from pathlib import Path

import duckdb

from pipeline.perturb import NICKNAMES

SOURCES = ("nppes", "open_payments", "part_d")


def _letters(expression: str) -> str:
    return f"nullif(regexp_replace(upper(coalesce({expression}, '')), '[^A-Z]', '', 'g'), '')"


def _first_token(expression: str) -> str:
    """First word of a name, so "GARCIA-LOPEZ" and "GARCIA" share a key."""
    return f"nullif(regexp_replace(upper(regexp_extract(coalesce({expression}, ''), '^[^ -]+')), '[^A-Z]', '', 'g'), '')"


def _street_number(expression: str) -> str:
    return f"nullif(regexp_extract(coalesce({expression}, ''), '^\\s*([0-9]+)', 1), '')"


def _street_name(expression: str) -> str:
    """First word after the house number, upper-cased: enough to tell two buildings on one ZIP apart."""
    return f"nullif(regexp_replace(upper(regexp_extract(coalesce({expression}, ''), '^\\s*[0-9]+\\s+(?:[NSEW]\\.?\\s+)?([A-Za-z0-9]+)', 1)), '[^A-Z0-9]', '', 'g'), '')"


def _nickname_table(con) -> None:
    rows = [(nick, formal) for formal, nicks in NICKNAMES.items() for nick in nicks]
    con.execute("CREATE OR REPLACE TABLE nickname_map (nick VARCHAR, formal VARCHAR)")
    con.executemany("INSERT INTO nickname_map VALUES (?, ?)", rows)
    # A nickname with several formal names (PAT) maps to one of them; good enough for a comparison feature.
    con.execute("CREATE OR REPLACE TABLE nickname_map AS SELECT nick, min(formal) AS formal FROM nickname_map GROUP BY 1")


def load(con: duckdb.DuckDBPyConnection, stripped: Path) -> dict[str, int]:
    """Create one table per source, named after it, with identical columns. Returns row counts."""
    _nickname_table(con)
    tables = {
        "nppes": f"""SELECT record_id, last_name, first_name, middle_name, other_last_name,
                            practice_address_1 AS address_1, practice_city AS city, practice_zip AS zip,
                            mailing_address_1 AS alt_address_1, mailing_zip AS alt_zip
                     FROM read_parquet('{stripped / 'nppes.parquet'}')""",
        "open_payments": f"""SELECT record_id, last_name, first_name, middle_name, NULL AS other_last_name,
                                    address_1, city, zip, NULL AS alt_address_1, NULL AS alt_zip
                             FROM read_parquet('{stripped / 'open_payments_recipients.parquet'}')""",
        "part_d": f"""SELECT record_id, last_name, first_name, middle_initial AS middle_name, NULL AS other_last_name,
                             address_1, city, zip, NULL AS alt_address_1, NULL AS alt_zip
                      FROM read_parquet('{stripped / 'part_d_prescribers.parquet'}')""",
    }
    counts = {}
    for source, query in tables.items():
        con.execute(f"""
            CREATE OR REPLACE TABLE {source} AS
            WITH s AS ({query})
            SELECT
                record_id AS unique_id,
                {_letters('last_name')} AS last_name,
                {_first_token('last_name')} AS last_token,
                coalesce({_letters('other_last_name')}, {_letters('last_name')}) AS last_name_alt,
                {_letters('first_name')} AS first_name,
                coalesce(n.formal, {_letters('first_name')}) AS first_name_formal,
                left({_letters('first_name')}, 1) AS first_initial,
                left({_letters('middle_name')}, 1) AS middle_initial,
                {_street_number('address_1')} AS street_number,
                {_street_name('address_1')} AS street_name,
                {_letters('city')} AS city,
                left(zip, 5) AS zip,
                {_street_number('alt_address_1')} AS alt_street_number,
                left(alt_zip, 5) AS alt_zip
            FROM s LEFT JOIN nickname_map n ON n.nick = {_letters('first_name')}
        """)
        counts[source] = con.execute(f"SELECT count(*) FROM {source}").fetchone()[0]
    return counts
