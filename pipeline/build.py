"""Build the scoped, ID-stripped tables and the hidden-NPI gold.

  python -m pipeline.build

data/<split>/<state>/stripped/  what the system may query: no identifiers, shuffled opaque record IDs
data/<split>/<state>/gold/      truth, read only by evals: record -> entity, slices, perturbation tags
reports/phase1_data_<split>_<state>.md  aggregates only, safe to commit
"""

import csv
import json
import random
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path

import duckdb

from pipeline import leak, perturb

ROOT = Path(__file__).resolve().parent.parent

N_TAXONOMY_SLOTS = 15
N_SPECIALTY_SLOTS = 6
SHARED_ADDRESS_MIN_FOCUS_PROVIDERS = 20
HANDCHECK_NON_MATCHES = 10
# Open Payments also lists teaching hospitals, with blank names and no NPI. This pool is of people.
OP_PERSON_TYPES = ["Covered Recipient Physician", "Covered Recipient Non-Physician Practitioner"]


class IntegrityError(RuntimeError):
    pass


@dataclass
class Paths:
    nppes: Path
    op_general: Path
    op_profiles: Path
    partd_provider: Path
    partd_drug: Path
    stripped: Path
    gold: Path
    report: Path | None = None


def state_dir(split: str, state: str) -> Path:
    return ROOT / "data" / split / state


def default_paths(split: str, state: str) -> Paths:
    raw = ROOT / "data" / "raw"

    def one(pattern: str) -> Path:
        found = [p for p in raw.glob(pattern) if not p.name.endswith((".done", ".zip"))]
        found = [p for p in found if "fileheader" not in p.name.lower()]
        if len(found) != 1:
            raise FileNotFoundError(f"Expected one file for {pattern}, found {[p.name for p in found]}")
        return found[0]

    return Paths(
        nppes=one("nppes/npidata_pfile*.csv"),
        op_general=one("open_payments_general/*.csv"),
        op_profiles=one("open_payments_profiles/*.csv"),
        partd_provider=one("partd_provider/*.csv"),
        partd_drug=one("partd_drug/*.csv"),
        stripped=state_dir(split, state) / "stripped",
        gold=state_dir(split, state) / "gold",
        report=ROOT / "reports" / f"phase1_data_{split}_{state}.md",
    )


def _csv(path: Path) -> str:
    return f"read_csv('{path}', header=true, all_varchar=true, delim=',', quote='\"', escape='\"', max_line_size=20000000, parallel=false)"


def _sql_list(values: list[str]) -> str:
    return "[" + ", ".join("'" + v.replace("'", "''") + "'" for v in values) + "]"


def _blank(column: str) -> str:
    return f"nullif(trim(\"{column}\"), '')"


def _text(column: str) -> str:
    """A free-text cell without long digit runs. Real data has NPIs and phone numbers typed into addresses, middle names
    and credentials, which no column allowlist can catch; none of these fields legitimately holds 10 digits in a row."""
    return f"nullif(trim(regexp_replace(coalesce(\"{column}\", ''), '[0-9]{{10,}}', '', 'g')), '')"


def _norm(expression: str) -> str:
    """Upper-case letters and digits only, for comparing names and addresses in gold labels."""
    return f"regexp_replace(upper(coalesce({expression}, '')), '[^A-Z0-9]', '', 'g')"


def _address_key(address: str, zip_code: str) -> str:
    """House number plus ZIP: tolerant of suite and abbreviation differences, which is what the slices need."""
    return f"(regexp_extract(coalesce({address}, ''), '^[0-9]+') || '|' || left(coalesce({zip_code}, ''), 5))"


def _padded(prefix_sql: str, number_sql: str) -> str:
    """Zero-padded ID. Not lpad: it silently truncates longer values, which made different providers share an ID."""
    return f"{prefix_sql} || printf('%07d', {number_sql})"


def _shuffled_ids(prefix: str, key_sql: str, seed: int) -> str:
    """Opaque ID from a salted hash order, salted per source so order says nothing across sources."""
    return _padded(f"'{prefix}_'", f"row_number() OVER (ORDER BY md5('{seed}:{prefix}:' || {key_sql}))")


def load_sources(con, paths: Paths, scope: dict) -> dict:
    state = scope["state"]
    taxonomy_slots = ", ".join(_blank(f"Healthcare Provider Taxonomy Code_{i}") for i in range(1, N_TAXONOMY_SLOTS + 1))
    primary = " ".join(
        f"""WHEN "Healthcare Provider Primary Taxonomy Switch_{i}" = 'Y' THEN {_blank(f'Healthcare Provider Taxonomy Code_{i}')}"""
        for i in range(1, N_TAXONOMY_SLOTS + 1)
    )
    con.execute(f"""
        CREATE TABLE k_nppes AS SELECT
            "NPI" AS npi,
            {_text('Provider Last Name (Legal Name)')} AS last_name,
            {_text('Provider First Name')} AS first_name,
            {_text('Provider Middle Name')} AS middle_name,
            {_text('Provider Name Suffix Text')} AS suffix,
            {_text('Provider Credential Text')} AS credential,
            {_text('Provider Other Last Name')} AS other_last_name,
            {_text('Provider Other First Name')} AS other_first_name,
            {_text('Provider Other Middle Name')} AS other_middle_name,
            {_text('Provider Other Credential Text')} AS other_credential,
            {_text('Provider First Line Business Practice Location Address')} AS practice_address_1,
            {_text('Provider Second Line Business Practice Location Address')} AS practice_address_2,
            {_text('Provider Business Practice Location Address City Name')} AS practice_city,
            {_text('Provider Business Practice Location Address State Name')} AS practice_state,
            left({_blank('Provider Business Practice Location Address Postal Code')}, 5) AS practice_zip,
            {_text('Provider First Line Business Mailing Address')} AS mailing_address_1,
            {_text('Provider Second Line Business Mailing Address')} AS mailing_address_2,
            {_text('Provider Business Mailing Address City Name')} AS mailing_city,
            {_text('Provider Business Mailing Address State Name')} AS mailing_state,
            left({_blank('Provider Business Mailing Address Postal Code')}, 5) AS mailing_zip,
            list_filter([{taxonomy_slots}], lambda x: x IS NOT NULL) AS taxonomy_codes,
            CASE {primary} END AS primary_taxonomy,
            {_text('Last Update Date')} AS last_update,
            list_has_any(list_filter([{taxonomy_slots}], lambda x: x IS NOT NULL), {_sql_list(scope['nppes']['taxonomy_codes'])}) AS is_focus,
            regexp_matches(concat_ws(' ', "Provider Last Name (Legal Name)", "Provider First Name", "Provider Middle Name",
                "Provider Name Suffix Text", "Provider Credential Text", "Provider Other Last Name", "Provider Other First Name",
                "Provider Other Middle Name", "Provider Other Credential Text",
                "Provider First Line Business Practice Location Address", "Provider Second Line Business Practice Location Address",
                "Provider First Line Business Mailing Address", "Provider Second Line Business Mailing Address"),
                '[0-9]{{10,}}') AS text_scrubbed
        FROM {_csv(paths.nppes)}
        WHERE "Entity Type Code" = '1'
          AND "Provider Business Practice Location Address State Name" = '{state}'
    """)

    specialty_slots = ", ".join(_blank(f"Covered_Recipient_Specialty_{i}") for i in range(1, N_SPECIALTY_SLOTS + 1))
    terminal = _sql_list(scope["open_payments"]["specialties"])
    products = ", ".join(_text(f"Name_of_Drug_or_Biological_or_Device_or_Medical_Supply_{i}") for i in range(1, 6))
    # An NPI missing on the payment row can be recovered from the profile ID. Identifiers stay in k_* tables only.
    con.execute(f"""
        CREATE TABLE k_profiles AS
        SELECT "Covered_Recipient_Profile_ID" AS profile_id, any_value({_blank('Covered_Recipient_NPI')}) AS npi
        FROM {_csv(paths.op_profiles)} GROUP BY 1
    """)
    identity = "r.last_name, r.first_name, r.middle_name, r.suffix, r.address_1, r.address_2, r.city, r.state, r.zip, r.specialties"
    con.execute(f"""
        CREATE TABLE k_op_rows AS
        WITH scoped AS (
            SELECT *, list_filter([{specialty_slots}], lambda x: x IS NOT NULL) AS specialties
            FROM {_csv(paths.op_general)}
            WHERE "Recipient_State" = '{state}' AND "Covered_Recipient_Type" IN ({_sql_list(OP_PERSON_TYPES)[1:-1]})
        ),
        r AS (
            SELECT
                {_blank('Covered_Recipient_NPI')} AS row_npi,
                {_blank('Covered_Recipient_Profile_ID')} AS profile_id,
                {_text('Covered_Recipient_Last_Name')} AS last_name,
                {_text('Covered_Recipient_First_Name')} AS first_name,
                {_text('Covered_Recipient_Middle_Name')} AS middle_name,
                {_text('Covered_Recipient_Name_Suffix')} AS suffix,
                {_text('Recipient_Primary_Business_Street_Address_Line1')} AS address_1,
                {_text('Recipient_Primary_Business_Street_Address_Line2')} AS address_2,
                {_text('Recipient_City')} AS city,
                {_blank('Recipient_State')} AS state,
                left({_blank('Recipient_Zip_Code')}, 5) AS zip,
                specialties,
                list_has_any(list_transform(specialties, lambda x: trim(string_split(x, '|')[-1])), {terminal}) AS is_focus,
                {_text('Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name')} AS payer,
                TRY_CAST("Total_Amount_of_Payment_USDollars" AS DOUBLE) AS amount_usd,
                {_blank('Date_of_Payment')} AS payment_date,
                TRY_CAST("Number_of_Payments_Included_in_Total_Amount" AS INTEGER) AS n_payments_in_total,
                {_text('Form_of_Payment_or_Transfer_of_Value')} AS payment_form,
                {_text('Nature_of_Payment_or_Transfer_of_Value')} AS payment_nature,
                list_filter([{products}], lambda x: x IS NOT NULL) AS products,
                "Record_ID" AS source_payment_id,
                regexp_matches(coalesce("Recipient_Primary_Business_Street_Address_Line1", '') || ' ' ||
                               coalesce("Recipient_Primary_Business_Street_Address_Line2", ''), '[0-9]{{10,}}') AS address_scrubbed
            FROM scoped
        )
        SELECT r.*, coalesce(r.row_npi, p.npi) AS npi,
               md5(CAST(row(coalesce(r.row_npi, p.npi), r.profile_id, {identity}) AS VARCHAR)) AS mention_key
        FROM r LEFT JOIN k_profiles p USING (profile_id)
    """)
    con.execute("""
        CREATE TABLE k_op_mentions AS
        SELECT mention_key, npi, profile_id, last_name, first_name, middle_name, suffix,
               address_1, address_2, city, state, zip, specialties, is_focus, count(*) AS n_payments
        FROM k_op_rows
        GROUP BY ALL
    """)

    types = _sql_list(scope["part_d"]["prescriber_types"])
    con.execute(f"""
        CREATE TABLE k_partd AS SELECT
            "Prscrbr_NPI" AS npi,
            {_text('Prscrbr_Last_Org_Name')} AS last_name,
            {_text('Prscrbr_First_Name')} AS first_name,
            {_text('Prscrbr_MI')} AS middle_initial,
            {_text('Prscrbr_Crdntls')} AS credential,
            {_text('Prscrbr_St1')} AS address_1,
            {_text('Prscrbr_St2')} AS address_2,
            {_text('Prscrbr_City')} AS city,
            {_blank('Prscrbr_State_Abrvtn')} AS state,
            {_blank('Prscrbr_Zip5')} AS zip,
            {_text('Prscrbr_Type')} AS prescriber_type,
            TRY_CAST("Tot_Clms" AS DOUBLE) AS total_claims,
            TRY_CAST("Tot_Drug_Cst" AS DOUBLE) AS total_drug_cost,
            TRY_CAST("Tot_Day_Suply" AS DOUBLE) AS total_day_supply,
            TRY_CAST("Tot_Benes" AS DOUBLE) AS total_beneficiaries,
            "Prscrbr_Type" IN (SELECT unnest({types})) AS is_focus
        FROM {_csv(paths.partd_provider)}
        WHERE "Prscrbr_Ent_Cd" = 'I' AND "Prscrbr_State_Abrvtn" = '{state}'
    """)
    con.execute(f"""
        CREATE TABLE k_partd_drug AS SELECT
            "Prscrbr_NPI" AS npi,
            {_text('Brnd_Name')} AS brand_name,
            {_text('Gnrc_Name')} AS generic_name,
            TRY_CAST("Tot_Clms" AS DOUBLE) AS total_claims,
            TRY_CAST("Tot_30day_Fills" AS DOUBLE) AS total_30day_fills,
            TRY_CAST("Tot_Day_Suply" AS DOUBLE) AS total_day_supply,
            TRY_CAST("Tot_Drug_Cst" AS DOUBLE) AS total_drug_cost,
            TRY_CAST("Tot_Benes" AS DOUBLE) AS total_beneficiaries
        FROM {_csv(paths.partd_drug)}
        WHERE "Prscrbr_NPI" IN (SELECT npi FROM k_partd)
    """)
    return {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            for t in ("k_nppes", "k_op_rows", "k_op_mentions", "k_partd", "k_partd_drug")}


def assign_ids(con, seed: int) -> None:
    con.execute(f"ALTER TABLE k_nppes ADD COLUMN record_id VARCHAR")
    con.execute(f"CREATE TABLE ids_nppes AS SELECT npi, {_shuffled_ids('np', 'npi', seed)} AS record_id FROM k_nppes")
    con.execute("UPDATE k_nppes k SET record_id = i.record_id FROM ids_nppes i WHERE k.npi = i.npi")

    con.execute(f"""
        CREATE TABLE k_op AS
        SELECT *, {_shuffled_ids('op', 'mention_key', seed)} AS record_id FROM k_op_mentions
    """)
    con.execute(f"ALTER TABLE k_partd ADD COLUMN record_id VARCHAR")
    con.execute(f"CREATE TABLE ids_partd AS SELECT npi, {_shuffled_ids('pd', 'npi', seed)} AS record_id FROM k_partd")
    con.execute("UPDATE k_partd k SET record_id = i.record_id FROM ids_partd i WHERE k.npi = i.npi")


def write_stripped(con, out: Path, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)

    def save(name: str, query: str) -> None:
        con.execute(f"COPY ({query}) TO '{out / name}.parquet' (FORMAT parquet)")

    save("nppes", """SELECT record_id, last_name, first_name, middle_name, suffix, credential, other_last_name,
            other_first_name, other_middle_name, other_credential, practice_address_1, practice_address_2, practice_city,
            practice_state, practice_zip, mailing_address_1, mailing_address_2, mailing_city, mailing_state, mailing_zip,
            taxonomy_codes, primary_taxonomy, last_update FROM k_nppes ORDER BY record_id""")
    save("open_payments_recipients", """SELECT record_id, last_name, first_name, middle_name, suffix, address_1, address_2,
            city, state, zip, specialties, n_payments FROM k_op ORDER BY record_id""")
    payment_id = _shuffled_ids("pay", "r.source_payment_id", seed)
    save("open_payments_payments", f"""
        SELECT {payment_id} AS payment_id, k.record_id, r.payer, r.amount_usd, r.payment_date, r.n_payments_in_total,
               r.payment_form, r.payment_nature, r.products
        FROM k_op_rows r JOIN k_op k USING (mention_key)
        ORDER BY payment_id""")
    save("part_d_prescribers", """SELECT record_id, last_name, first_name, middle_initial, credential, address_1, address_2,
            city, state, zip, prescriber_type, total_claims, total_drug_cost, total_day_supply, total_beneficiaries
            FROM k_partd ORDER BY record_id""")
    save("part_d_drugs", """SELECT p.record_id, d.brand_name, d.generic_name, d.total_claims, d.total_30day_fills,
            d.total_day_supply, d.total_drug_cost, d.total_beneficiaries
            FROM k_partd_drug d JOIN k_partd p USING (npi)
            ORDER BY md5(p.record_id || coalesce(d.brand_name, '') || coalesce(d.generic_name, ''))""")
    con.execute("""
        CREATE TABLE identity_records AS
        SELECT record_id, 'nppes' AS source, last_name, first_name, middle_name, suffix,
               practice_address_1 AS address_1, practice_address_2 AS address_2, practice_city AS city,
               practice_state AS state, practice_zip AS zip FROM k_nppes
        UNION ALL
        SELECT record_id, 'open_payments', last_name, first_name, middle_name, suffix, address_1, address_2, city, state, zip FROM k_op
        UNION ALL
        SELECT record_id, 'part_d', last_name, first_name, middle_initial, NULL, address_1, address_2, city, state, zip FROM k_partd
    """)
    save("identity_records", "SELECT * FROM identity_records ORDER BY record_id")


def write_gold(con, out: Path, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        CREATE TABLE k_all AS
        SELECT record_id, 'nppes' AS source, npi FROM k_nppes
        UNION ALL SELECT record_id, 'open_payments', npi FROM k_op
        UNION ALL SELECT record_id, 'part_d', npi FROM k_partd
    """)
    con.execute(f"""
        CREATE TABLE entities AS
        SELECT npi, {_padded("'E'", f"row_number() OVER (ORDER BY md5('{seed}:entity:' || npi))")} AS entity_id
        FROM (SELECT DISTINCT npi FROM k_all WHERE npi IS NOT NULL)
    """)
    con.execute("""
        CREATE TABLE membership AS
        SELECT a.record_id, a.source, e.entity_id FROM k_all a JOIN entities e USING (npi)
    """)
    # Questions target this group; everyone else in the state is in the pool as a realistic decoy. A provider counts if
    # any source's own fields say so, since payers often list a cardiologist under a broader specialty.
    con.execute("""
        CREATE TABLE focus AS
        SELECT DISTINCT e.entity_id FROM entities e JOIN (
            SELECT npi FROM k_nppes WHERE is_focus
            UNION SELECT npi FROM k_op WHERE is_focus
            UNION SELECT npi FROM k_partd WHERE is_focus
        ) f USING (npi)
    """)
    con.execute("COPY (SELECT * FROM focus ORDER BY entity_id) TO '%s' (FORMAT parquet)" % (out / "focus_entities.parquet"))
    con.execute("COPY (SELECT * FROM membership ORDER BY record_id) TO '%s' (FORMAT parquet)" % (out / "membership.parquet"))
    con.execute("COPY (SELECT record_id, source FROM k_all WHERE npi IS NULL ORDER BY record_id) TO '%s' (FORMAT parquet)"
                % (out / "unlabeled.parquet"))
    con.execute("COPY (SELECT entity_id, npi FROM entities ORDER BY entity_id) TO '%s' (FORMAT parquet)" % (out / "npi_map.parquet"))
    con.execute(f"""
        COPY (
            SELECT DISTINCT a.record_id AS left_record, a.source AS left_source, b.record_id AS right_record,
                   b.source AS right_source, a.entity_id
            FROM membership a JOIN membership b ON a.entity_id = b.entity_id AND a.record_id < b.record_id
            ORDER BY 1, 3
        ) TO '{out / "links.parquet"}' (FORMAT parquet)
    """)


def check_integrity(con) -> None:
    """Facts the gold depends on, checked on real output. Raises rather than build on a broken join or colliding IDs."""
    checks = {
        "an NPI maps to one entity and an entity to one NPI":
            "SELECT count(*) - count(DISTINCT npi) + count(*) - count(DISTINCT entity_id) FROM entities",
        "record IDs are unique": "SELECT count(*) - count(DISTINCT record_id) FROM k_all",
        "every labeled record has an entity":
            "SELECT count(*) FROM k_all a LEFT JOIN entities e USING (npi) WHERE a.npi IS NOT NULL AND e.entity_id IS NULL",
        "a provider has at most one registry record":
            "SELECT count(*) FROM (SELECT entity_id FROM membership WHERE source = 'nppes' GROUP BY 1 HAVING count(*) > 1)",
    }
    failed = [name for name, query in checks.items() if con.execute(query).fetchone()[0] != 0]
    if failed:
        raise IntegrityError("Gold integrity check failed: " + "; ".join(failed))


def write_slices(con, out: Path) -> None:
    """Hard cases labeled from the real data, for focus providers. Collisions are counted against everyone in the state.
    shared_group_address means many cardiologists at one address, so address plus specialty can't identify someone.
    Counting all providers instead flagged about 70% of records, because hospital campuses hold hundreds.
    A record can sit in several slices."""
    name_key = f"{_norm('last_name')} || '|' || {_norm('first_name')}"
    initial_key = f"{_norm('last_name')} || '|' || left({_norm('first_name')}, 1)"
    city = _norm("practice_city")
    practice_key = _address_key("practice_address_1", "practice_zip")
    mailing_key = _address_key("mailing_address_1", "mailing_zip")
    con.execute(f"""
        CREATE TABLE slices AS
        SELECT DISTINCT s.record_id, s.source, s.slice FROM (
        WITH np AS (SELECT n.*, m.entity_id FROM k_nppes n JOIN membership m USING (record_id)),
        namesake_city AS (SELECT entity_id FROM np QUALIFY count(*) OVER (PARTITION BY {name_key}, {city}) > 1),
        namesake_state AS (SELECT entity_id FROM np QUALIFY count(*) OVER (PARTITION BY {name_key}) > 1),
        initial_city AS (SELECT entity_id FROM np QUALIFY count(*) OVER (PARTITION BY {initial_key}, {city}) > 1),
        shared AS (SELECT {practice_key} AS key FROM np
                   WHERE entity_id IN (SELECT entity_id FROM focus) AND regexp_matches(coalesce(practice_address_1, ''), '^[0-9]')
                   GROUP BY 1 HAVING count(*) >= {SHARED_ADDRESS_MIN_FOCUS_PROVIDERS}),
        same_surname_address AS (SELECT entity_id FROM np QUALIFY count(*) OVER (PARTITION BY {practice_key}, {_norm('last_name')}) > 1),
        others AS (
            SELECT o.record_id, o.source, o.last_name, o.first_name, o.address_1, o.zip, m.entity_id
            FROM (SELECT record_id, 'open_payments' AS source, last_name, first_name, address_1, zip FROM k_op
                  UNION ALL SELECT record_id, 'part_d', last_name, first_name, address_1, zip FROM k_partd) o
            JOIN membership m USING (record_id)
        ),
        joined AS (SELECT o.*, n.last_name AS np_last, n.first_name AS np_first, n.other_last_name AS np_other_last,
                          {practice_key.replace('practice_address_1', 'n.practice_address_1').replace('practice_zip', 'n.practice_zip')} AS np_practice,
                          {mailing_key.replace('mailing_address_1', 'n.mailing_address_1').replace('mailing_zip', 'n.mailing_zip')} AS np_mailing,
                          n.practice_zip AS np_practice_zip, n.mailing_zip AS np_mailing_zip
                   FROM others o JOIN np n USING (entity_id))
        SELECT m.record_id, m.source, 'namesake_same_city' AS slice FROM membership m WHERE entity_id IN (SELECT entity_id FROM namesake_city)
        UNION ALL SELECT m.record_id, m.source, 'namesake_same_state' FROM membership m WHERE entity_id IN (SELECT entity_id FROM namesake_state)
        UNION ALL SELECT m.record_id, m.source, 'initial_collision_same_city' FROM membership m WHERE entity_id IN (SELECT entity_id FROM initial_city)
        UNION ALL SELECT m.record_id, m.source, 'shared_group_address' FROM membership m JOIN np USING (entity_id)
                  WHERE {practice_key.replace('practice_address_1', 'np.practice_address_1').replace('practice_zip', 'np.practice_zip')} IN (SELECT key FROM shared)
        UNION ALL SELECT record_id, source, 'address_differs_zip' FROM joined
                  WHERE zip IS NOT NULL AND zip IS DISTINCT FROM np_practice_zip AND zip IS DISTINCT FROM np_mailing_zip
        UNION ALL SELECT record_id, source, 'address_differs_street' FROM joined
                  WHERE {_address_key('address_1', 'zip')} NOT IN (np_practice, np_mailing)
        UNION ALL SELECT record_id, source, 'last_name_matches_other_name' FROM joined
                  WHERE {_norm('np_other_last')} <> '' AND {_norm('last_name')} = {_norm('np_other_last')}
                    AND {_norm('last_name')} <> {_norm('np_last')}
        UNION ALL SELECT record_id, source, 'last_name_differs_unexplained' FROM joined
                  WHERE {_norm('last_name')} <> {_norm('np_last')} AND {_norm('last_name')} <> {_norm('np_other_last')}
        UNION ALL SELECT record_id, source, 'first_name_differs' FROM joined WHERE {_norm('first_name')} <> {_norm('np_first')}
        UNION ALL SELECT m.record_id, m.source, 'no_registry_counterpart' FROM membership m
                  WHERE m.source <> 'nppes' AND m.entity_id NOT IN (SELECT entity_id FROM np)
        UNION ALL SELECT m.record_id, m.source, 'absent_from_part_d' FROM membership m
                  WHERE m.source = 'nppes' AND m.entity_id NOT IN (SELECT entity_id FROM membership WHERE source = 'part_d')
        UNION ALL SELECT m.record_id, m.source, 'absent_from_open_payments' FROM membership m
                  WHERE m.source = 'nppes' AND m.entity_id NOT IN (SELECT entity_id FROM membership WHERE source = 'open_payments')
        UNION ALL SELECT m.record_id, m.source, 'same_surname_same_address' FROM membership m
                  WHERE entity_id IN (SELECT entity_id FROM same_surname_address)
        ) s JOIN membership f USING (record_id) WHERE f.entity_id IN (SELECT entity_id FROM focus)
    """)
    con.execute("COPY (SELECT * FROM slices ORDER BY slice, record_id) TO '%s' (FORMAT parquet)" % (out / "slices.parquet"))


def write_perturbations(con, stripped: Path, gold: Path, scope: dict) -> dict:
    """Perturb focus providers' Open Payments and Part D records that have a registry match, so gold stays answerable."""
    seed, per_type = scope["seed"], scope["perturbations"]["per_type"]
    columns = ["record_id", "source", "last_name", "first_name", "middle_name", "suffix", "address_1", "address_2", "city", "state", "zip"]
    fetch = lambda query: [dict(zip(columns, row)) for row in con.execute(query).fetchall()]
    records = fetch("""SELECT i.* FROM identity_records i JOIN membership m USING (record_id)
                       WHERE i.source <> 'nppes' AND m.entity_id IN (SELECT entity_id FROM focus)
                         AND m.entity_id IN (SELECT entity_id FROM membership WHERE source = 'nppes')
                       ORDER BY i.record_id""")
    donors = fetch("""SELECT i.* FROM identity_records i JOIN membership m USING (record_id)
                      WHERE i.source = 'nppes' AND m.entity_id IN (SELECT entity_id FROM focus) ORDER BY i.record_id""")
    generated = perturb.generate(records, donors, per_type, seed)
    # IDs are assigned after a shuffle, so ID order doesn't reveal the perturbation type.
    random.Random(f"{seed}:px").shuffle(generated)
    rows, mapping = [], []
    for i, item in enumerate(generated, start=1):
        record_id = f"px_{i:06d}"
        rows.append(tuple(record_id if c == "record_id" else item["record"].get(c) for c in columns))
        mapping.append((record_id, item["type"], item["origin"]))
    con.execute(f"CREATE TABLE perturbed ({', '.join(c + ' VARCHAR' for c in columns)})")
    con.execute("CREATE TABLE perturbed_map (record_id VARCHAR, perturbation_type VARCHAR, origin_record VARCHAR)")
    if rows:
        con.executemany(f"INSERT INTO perturbed VALUES ({', '.join('?' * len(columns))})", rows)
        con.executemany("INSERT INTO perturbed_map VALUES (?, ?, ?)", mapping)
    con.execute(f"COPY (SELECT * FROM perturbed ORDER BY record_id) TO '{stripped / 'perturbed_records.parquet'}' (FORMAT parquet)")
    con.execute(f"""
        COPY (SELECT p.record_id, p.perturbation_type, p.origin_record, m.entity_id
              FROM perturbed_map p JOIN membership m ON p.origin_record = m.record_id ORDER BY p.record_id)
        TO '{gold / 'perturbed_map.parquet'}' (FORMAT parquet)
    """)
    counts = con.execute("SELECT perturbation_type, count(*) FROM perturbed_map GROUP BY 1 ORDER BY 1").fetchall()
    return dict(counts)


def write_handcheck(con, out: Path, seed: int, n: int = 50) -> int:
    """A blind sample for a person to judge: true links across the hard slices, mixed with same-name non-matches so
    that "same" can't be the right answer every time. Answers sit in a separate key. Named data: stays under data/."""
    matches = con.execute(f"""
        WITH candidates AS (
            SELECT a.record_id AS left_record, b.record_id AS right_record, a.source || '~' || b.source AS link_type,
                   coalesce(s.slice, 'easy') AS slice
            FROM (SELECT * FROM membership WHERE entity_id IN (SELECT entity_id FROM focus)) a
            JOIN membership b ON a.entity_id = b.entity_id AND a.source < b.source
            LEFT JOIN slices s ON s.record_id = b.record_id AND s.slice IN
                ('address_differs_zip', 'last_name_matches_other_name', 'last_name_differs_unexplained', 'first_name_differs', 'namesake_same_city', 'same_surname_same_address', 'shared_group_address')
        ),
        ranked AS (
            SELECT *, row_number() OVER (PARTITION BY link_type, slice ORDER BY md5('{seed}:check:' || left_record || right_record)) AS rn
            FROM candidates
        )
        SELECT left_record, right_record, link_type, slice, 'same' FROM ranked
        ORDER BY rn, link_type, slice LIMIT {n - HANDCHECK_NON_MATCHES}
    """).fetchall()
    last, first = _norm("i.last_name"), _norm("i.first_name")
    non_matches = con.execute(f"""
        WITH named AS (
            SELECT i.record_id, i.source, m.entity_id, {last} AS l, {first} AS f
            FROM identity_records i JOIN membership m USING (record_id)
            WHERE {last} <> '' AND {first} <> ''
        )
        SELECT a.record_id, b.record_id, a.source || '~' || b.source, 'same_name_non_match', 'different'
        FROM named a JOIN named b ON a.l = b.l AND a.f = b.f AND a.source < b.source AND a.entity_id <> b.entity_id
        WHERE a.entity_id IN (SELECT entity_id FROM focus) OR b.entity_id IN (SELECT entity_id FROM focus)
        ORDER BY md5('{seed}:check:' || a.record_id || b.record_id) LIMIT {HANDCHECK_NON_MATCHES}
    """).fetchall()
    pairs = matches + non_matches
    random.Random(f"{seed}:handcheck").shuffle(pairs)
    columns = "source, last_name, first_name, middle_name, address_1, address_2, city, zip"
    identity = {row[0]: row[1:] for row in con.execute(f"""
        SELECT record_id, {columns} FROM identity_records
        WHERE record_id IN (SELECT unnest(?))""", [[r for pair in pairs for r in pair[:2]]]).fetchall()}
    fields = [c.strip() for c in columns.split(",")]
    header = ["pair", *[f"a_{f}" for f in fields], *[f"b_{f}" for f in fields], "verdict", "notes"]
    with (out / "handcheck_sample.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, pair in enumerate(pairs, start=1):
            writer.writerow([i, *identity[pair[0]], *identity[pair[1]], "", ""])
    with (out / "handcheck_key.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair", "gold", "link_type", "slice", "a_record", "b_record"])
        for i, pair in enumerate(pairs, start=1):
            writer.writerow([i, pair[4], pair[2], pair[3], pair[0], pair[1]])
    return len(pairs)


def build_stats(con, loaded: dict) -> dict:
    one = lambda q: con.execute(q).fetchone()[0]
    stats = {
        "loaded": loaded,
        "payments_written": one("SELECT count(*) FROM k_op_rows"),
        "op_rows_address_scrubbed": one("SELECT count(*) FROM k_op_rows WHERE address_scrubbed"),
        "nppes_rows_text_scrubbed": one("SELECT count(*) FROM k_nppes WHERE text_scrubbed"),
        "entities": one("SELECT count(*) FROM entities"),
        "focus_entities": one("SELECT count(*) FROM focus"),
        "op_unlabeled_rows": one("SELECT count(*) FROM k_op_rows WHERE npi IS NULL"),
        "op_unlabeled_dollar_share": round(one("SELECT coalesce(sum(amount_usd) FILTER (npi IS NULL), 0) / sum(amount_usd) FROM k_op_rows"), 5),
        "focus_entities_by_source_combo": dict(con.execute("""
            SELECT combo, count(*) FROM (SELECT entity_id, string_agg(DISTINCT source, '+' ORDER BY source) AS combo
                                         FROM membership WHERE entity_id IN (SELECT entity_id FROM focus) GROUP BY 1)
            GROUP BY 1 ORDER BY 2 DESC""").fetchall()),
        "records_by_source": dict(con.execute("SELECT source, count(*) FROM k_all GROUP BY 1 ORDER BY 1").fetchall()),
        "unlabeled_by_source": dict(con.execute("SELECT source, count(*) FROM k_all WHERE npi IS NULL GROUP BY 1 ORDER BY 1").fetchall()),
        "op_rows_npi_from_profile": one("SELECT count(*) FROM k_op_rows WHERE row_npi IS NULL AND npi IS NOT NULL"),
        "entities_by_source_combo": dict(con.execute("""
            SELECT combo, count(*) FROM (SELECT entity_id, string_agg(DISTINCT source, '+' ORDER BY source) AS combo
                                         FROM membership GROUP BY 1) GROUP BY 1 ORDER BY 2 DESC""").fetchall()),
        "op_mentions_per_entity": dict(zip(("mean", "max"), con.execute("""
            SELECT round(avg(c), 2), max(c) FROM (SELECT count(*) AS c FROM membership WHERE source = 'open_payments' GROUP BY entity_id)""").fetchone())),
        "slices": dict(con.execute("SELECT slice, count(DISTINCT record_id) FROM slices GROUP BY 1 ORDER BY 1").fetchall()),
    }
    return stats


def write_report(path: Path, stats: dict, scope: dict) -> None:
    lines = [
        "# Phase 1 data build", "",
        f"Scope: all individual providers in {scope['state']}, program year {scope['program_year']}, with "
        f"{stats['focus_entities']:,} cardiology providers as the focus group. Aggregates only; no provider-level data.", "",
        "## Records", "", "| Source | Records | Without known NPI |", "|---|---|---|",
    ]
    for source, count in stats["records_by_source"].items():
        lines.append(f"| {source} | {count:,} | {stats['unlabeled_by_source'].get(source, 0):,} |")
    lines += ["", f"Open Payments rows in scope: {stats['payments_written']:,}. "
              f"Rows with no NPI: {stats['op_unlabeled_rows']:,} ({stats['op_unlabeled_dollar_share']:.2%} of dollars). "
              f"NPIs recovered from the profile file: {stats['op_rows_npi_from_profile']:,} rows. "
              f"Long digit runs removed: {stats['op_rows_address_scrubbed']:,} Open Payments rows (address), "
              f"{stats['nppes_rows_text_scrubbed']:,} NPPES records (any text field). "
              f"Recipient records per provider: mean {stats['op_mentions_per_entity']['mean']}, "
              f"max {stats['op_mentions_per_entity']['max']}.", "",
              "## Providers by which sources contain them", "", "| Sources | All providers | Focus providers |", "|---|---|---|"]
    lines += [f"| {combo} | {count:,} | {stats['focus_entities_by_source_combo'].get(combo, 0):,} |"
              for combo, count in stats["entities_by_source_combo"].items()]
    lines += ["", "## Hard-case slices (focus providers' records)", "", "| Slice | Records |", "|---|---|"]
    lines += [f"| {name} | {count:,} |" for name, count in stats["slices"].items()]
    lines += ["", "## Perturbations", "", "| Type | Records |", "|---|---|"]
    lines += [f"| {name} | {count:,} |" for name, count in stats["perturbations"].items()]
    lines += ["", f"Leak check: passed over {stats['leak']['tables']} tables. Cross-source rank correlation of record IDs: "
              f"{stats['leak']['order_rho']}.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def build(paths: Paths, scope: dict) -> dict:
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    seed = scope["seed"]
    loaded = load_sources(con, paths, scope)
    assign_ids(con, seed)
    write_stripped(con, paths.stripped, seed)
    write_gold(con, paths.gold, seed)
    check_integrity(con)
    write_slices(con, paths.gold)
    perturbation_counts = write_perturbations(con, paths.stripped, paths.gold, scope)
    handcheck = write_handcheck(con, paths.gold, seed)

    npis = {row[0] for row in con.execute("SELECT npi FROM entities").fetchall()}
    try:
        tables = leak.check_stripped(paths.stripped, npis)
    except leak.LeakError:
        shutil.rmtree(paths.stripped)  # tables that failed the check must not stay around to be queried
        raise
    order_pairs = con.execute("""
        SELECT try_cast(split_part(a.record_id, '_', 2) AS DOUBLE), try_cast(split_part(b.record_id, '_', 2) AS DOUBLE)
        FROM membership a JOIN membership b USING (entity_id) WHERE a.source = 'nppes' AND b.source = 'part_d'
    """).fetchall()
    rho = leak.check_order_leak(order_pairs)

    stats = build_stats(con, loaded)
    stats.update(perturbations=perturbation_counts, handcheck_rows=handcheck,
                 leak={"tables": len(tables), "order_rho": None if rho is None else round(rho, 4)})
    (paths.gold / "build_stats.json").write_text(json.dumps(stats, indent=2))
    if paths.report:
        write_report(paths.report, stats, scope)
    return stats


def state_scope(scope: dict, state: str) -> dict:
    """One state's build settings. The seed is per state, so IDs don't line up between dev and held-out."""
    return {**scope, "state": state, "seed": f"{scope['seed']}:{state}"}


def main() -> None:
    scope = tomllib.loads((ROOT / "config" / "scope.toml").read_text())
    # Built separately into separate directories: tuning code pointed at data/dev can't read held-out records.
    # A provider's TX and CA records are not linked across the two builds.
    for split, states in scope["splits"].items():
        for state in states:
            stats = build(default_paths(split, state), state_scope(scope, state))
            print(split, state, json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
