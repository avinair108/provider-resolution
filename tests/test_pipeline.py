"""The build, end to end, on small synthetic CSVs with the real column names. No real provider data."""

import csv
import dataclasses
import random

import duckdb
import pytest

from pipeline import build as build_module
from pipeline import clean, handcheck, leak, perturb
from pipeline.build import Paths, build

# Synthetic NPIs: valid check digits, artificial 1999999 prefix, not taken from any real or documented example.
NPI_A, NPI_B, NPI_C, NPI_D = "1999999901", "1999999919", "1999999927", "1999999935"

SCOPE = {
    "program_year": 2024, "state": "TX", "seed": 7,
    "nppes": {"taxonomy_codes": ["207RC0000X"]},
    "open_payments": {"specialties": ["Cardiovascular Disease"]},
    "part_d": {"prescriber_types": ["Cardiology"]},
    "perturbations": {"per_type": 2},
}


def write_csv(path, rows):
    columns = list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def nppes_row(npi, last, first, state="TX", taxonomy="207RC0000X", entity="1", street="100 MAIN ST", zip_code="752011234",
              other_last=""):
    row = {
        "NPI": npi, "Entity Type Code": entity, "Employer Identification Number (EIN)": "",
        "Provider Last Name (Legal Name)": last, "Provider First Name": first, "Provider Middle Name": "Q",
        "Provider Name Suffix Text": "", "Provider Credential Text": "M.D.", "Provider Other Last Name": other_last,
        "Provider Other First Name": "", "Provider Other Middle Name": "", "Provider Other Credential Text": "",
        "Provider First Line Business Practice Location Address": street,
        "Provider Second Line Business Practice Location Address": "",
        "Provider Business Practice Location Address City Name": "DALLAS",
        "Provider Business Practice Location Address State Name": state,
        "Provider Business Practice Location Address Postal Code": zip_code,
        "Provider First Line Business Mailing Address": street, "Provider Second Line Business Mailing Address": "",
        "Provider Business Mailing Address City Name": "DALLAS", "Provider Business Mailing Address State Name": state,
        "Provider Business Mailing Address Postal Code": zip_code, "Last Update Date": "01/01/2025",
        "Provider License Number_1": "SECRETLICENSE",
    }
    for i in range(1, 16):
        row[f"Healthcare Provider Taxonomy Code_{i}"] = taxonomy if i == 1 else ""
        row[f"Healthcare Provider Primary Taxonomy Switch_{i}"] = "Y" if i == 1 else ""
    return row


def op_row(record_id, npi, profile, last, first, amount, street="100 MAIN STREET", zip_code="75201-1234", state="TX",
           specialty="Allopathic & Osteopathic Physicians|Internal Medicine|Cardiovascular Disease"):
    row = {
        "Covered_Recipient_Type": "Covered Recipient Physician", "Covered_Recipient_Profile_ID": profile,
        "Covered_Recipient_NPI": npi, "Covered_Recipient_First_Name": first, "Covered_Recipient_Middle_Name": "",
        "Covered_Recipient_Last_Name": last, "Covered_Recipient_Name_Suffix": "",
        "Recipient_Primary_Business_Street_Address_Line1": street, "Recipient_Primary_Business_Street_Address_Line2": "",
        "Recipient_City": "DALLAS", "Recipient_State": state, "Recipient_Zip_Code": zip_code,
        "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name": "ACME PHARMA",
        "Total_Amount_of_Payment_USDollars": amount, "Date_of_Payment": "03/01/2024",
        "Number_of_Payments_Included_in_Total_Amount": "1", "Form_of_Payment_or_Transfer_of_Value": "Cash",
        "Nature_of_Payment_or_Transfer_of_Value": "Food and Beverage", "Record_ID": record_id,
        "Contextual_Information": f"Contact NPI {NPI_A}",
    }
    for i in range(1, 7):
        row[f"Covered_Recipient_Specialty_{i}"] = specialty if i == 1 else ""
    for i in range(1, 6):
        row[f"Name_of_Drug_or_Biological_or_Device_or_Medical_Supply_{i}"] = "ZYNTRA" if i == 1 else ""
    return row


def partd_row(npi, last, first, state="TX", prescriber_type="Cardiology"):
    return {
        "Prscrbr_NPI": npi, "Prscrbr_Last_Org_Name": last, "Prscrbr_First_Name": first, "Prscrbr_MI": "Q",
        "Prscrbr_Crdntls": "MD", "Prscrbr_Ent_Cd": "I", "Prscrbr_St1": "100 Main St", "Prscrbr_St2": "",
        "Prscrbr_City": "Dallas", "Prscrbr_State_Abrvtn": state, "Prscrbr_Zip5": "75201", "Prscrbr_Type": prescriber_type,
        "Tot_Clms": "100", "Tot_Drug_Cst": "5000.5", "Tot_Day_Suply": "3000", "Tot_Benes": "40",
    }


def drug_row(npi, brand):
    return {
        "Prscrbr_NPI": npi, "Brnd_Name": brand, "Gnrc_Name": brand.lower(), "Tot_Clms": "20", "Tot_30day_Fills": "22",
        "Tot_Day_Suply": "600", "Tot_Drug_Cst": "900", "Tot_Benes": "15",
    }


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("build")
    paths = Paths(
        nppes=write_csv(d / "nppes.csv", [
            nppes_row(NPI_A, "SMITH", "JOHN"),
            nppes_row(NPI_B, "NGUYEN", "ANNA", other_last="TRAN"),
            nppes_row(NPI_C, "OUTSIDE", "STATE", state="OK"),                  # wrong state
            nppes_row(NPI_D, "GENERAL", "PRACTICE", taxonomy="207Q00000X"),    # wrong specialty
        ]),
        op_general=write_csv(d / "op.csv", [
            op_row("r1", NPI_A, "111", "Smith", "John", "100.50"),
            op_row("r2", NPI_A, "111", "Smith", "John", "50.00"),               # same mention, second payment
            op_row("r3", NPI_A, "111", "Smith", "J", "25.00"),                  # second mention of the same provider
            op_row("r4", "", "222", "Tran", "Anna", "75.00", zip_code="78701"),  # NPI only via profile; other name; moved
            op_row("r5", "", "333", "Mystery", "Person", "10.00"),              # no NPI anywhere: unlabeled
            op_row("r6", NPI_C, "444", "Other", "State", "10.00", state="OK"),  # out of scope
        ]),
        op_profiles=write_csv(d / "profiles.csv", [
            {"Covered_Recipient_Profile_ID": "222", "Covered_Recipient_NPI": NPI_B},
            {"Covered_Recipient_Profile_ID": "333", "Covered_Recipient_NPI": ""},
        ]),
        partd_provider=write_csv(d / "partd.csv", [
            partd_row(NPI_A, "SMITH", "JOHN"),
            partd_row(NPI_D, "GENERAL", "PRACTICE", prescriber_type="Family Practice"),  # out of scope
        ]),
        partd_drug=write_csv(d / "drug.csv", [drug_row(NPI_A, "ZYNTRA"), drug_row(NPI_D, "OTHER")]),
        stripped=d / "stripped", gold=d / "gold", report=d / "report.md",
    )
    stats = build(paths, SCOPE)
    return paths, stats, duckdb.connect()


def query(con, path, sql="SELECT * FROM t"):
    return con.execute(sql.replace(" t", f" read_parquet('{path}')", 1)).fetchall()


def test_pool_is_whole_state_and_focus_is_gold_only(built):
    paths, stats, con = built
    # Non-cardiologists stay in the pool as decoys; other states don't.
    assert stats["records_by_source"] == {"nppes": 3, "open_payments": 4, "part_d": 2}
    assert stats["entities"] == 3 and stats["focus_entities"] == 2
    for path in paths.stripped.glob("*.parquet"):
        columns = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()}
        assert "is_focus" not in columns


def test_payments_under_a_broader_specialty_are_kept_and_mark_focus(built, tmp_path):
    paths, _, con = built
    row = op_row("r10", NPI_D, "555", "General", "Practice", "9.00",
                 specialty="Allopathic & Osteopathic Physicians|Internal Medicine")
    again, stats = _rebuild_with_op_rows(paths, tmp_path, [row])
    total = con.execute(f"SELECT sum(amount_usd) FROM read_parquet('{again.stripped / 'open_payments_payments.parquet'}')").fetchone()[0]
    assert total == pytest.approx(269.5) and stats["focus_entities"] == 2  # kept, but not cardiology by any source


def test_any_sources_fields_can_mark_focus(built, tmp_path):
    paths, _, _ = built
    row = op_row("r11", NPI_D, "555", "General", "Practice", "9.00")  # cardiology only according to Open Payments
    _, stats = _rebuild_with_op_rows(paths, tmp_path, [row])
    assert stats["focus_entities"] == 3


def test_payment_rows_collapse_into_mentions(built):
    paths, _, con = built
    payments = query(con, paths.stripped / "open_payments_payments.parquet")
    recipients = query(con, paths.stripped / "open_payments_recipients.parquet")
    assert len(payments) == 5 and len(recipients) == 4  # r1+r2 collapse; r3, r4 and r5 are each their own
    amounts = con.execute(f"SELECT sum(amount_usd) FROM read_parquet('{paths.stripped / 'open_payments_payments.parquet'}')").fetchone()[0]
    assert amounts == pytest.approx(260.5)


def test_npi_recovered_from_profile(built):
    paths, stats, con = built
    assert stats["op_rows_npi_from_profile"] == 1
    assert stats["unlabeled_by_source"] == {"open_payments": 1}


def test_gold_links_same_provider_across_sources(built):
    paths, _, con = built
    rows = con.execute(f"""SELECT entity_id, count(*), count(DISTINCT source) FROM read_parquet('{paths.gold / 'membership.parquet'}')
                           GROUP BY 1 ORDER BY 2 DESC""").fetchall()
    assert [(r[1], r[2]) for r in rows] == [(4, 3), (2, 2), (2, 2)]


def test_stripped_output_contains_no_identifiers(built):
    paths, _, con = built
    for path in paths.stripped.glob("*.parquet"):
        blob = " ".join(str(row) for row in query(con, path))
        for secret in (NPI_A, NPI_B, NPI_C, NPI_D, "SECRETLICENSE", "Contact NPI"):
            assert secret not in blob, f"{secret} leaked into {path.name}"


def test_slices_label_hard_cases(built):
    paths, _, con = built
    rows = query(con, paths.gold / "slices.parquet", "SELECT slice, count(*) FROM t GROUP BY 1")
    found = dict(rows)
    assert found["last_name_matches_other_name"] == 1
    assert found["address_differs_zip"] == 1
    assert found["first_name_differs"] == 1
    assert found["absent_from_part_d"] == 1  # the non-cardiologist's records are never labeled


def test_perturbed_records_never_show_their_type(built):
    paths, _, con = built
    columns = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{paths.stripped / 'perturbed_records.parquet'}')").fetchall()}
    assert not columns & {"perturbation_type", "origin_record", "type", "origin"}
    types = dict(query(con, paths.gold / "perturbed_map.parquet", "SELECT perturbation_type, count(*) FROM t GROUP BY 1"))
    assert types and all(count <= 2 for count in types.values())


def test_handcheck_written_with_blank_verdicts_and_no_answers(built):
    paths, stats, _ = built
    with (paths.gold / "handcheck_sample.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == stats["handcheck_rows"] > 0 and all(r["verdict"] == "" for r in rows)
    assert not set(rows[0]) & {"gold", "slice", "link_type", "a_record", "b_record"}  # nothing that gives it away


def test_handcheck_mixes_in_same_name_non_matches(built, tmp_path):
    paths, _, _ = built
    # Same name as an existing provider, but a different NPI: must appear as a non-match in the key.
    row = nppes_row(NPI_D, "SMITH", "JOHN", taxonomy="207Q00000X", street="9 ELM RD")
    with paths.nppes.open() as f:
        rows = [r for r in csv.DictReader(f) if r["NPI"] != NPI_D]
    again = dataclasses.replace(paths, nppes=write_csv(tmp_path / "nppes.csv", rows + [row]),
                                stripped=tmp_path / "stripped", gold=tmp_path / "gold", report=None)
    build(again, SCOPE)
    with (again.gold / "handcheck_key.csv").open() as f:
        key = list(csv.DictReader(f))
    assert {k["gold"] for k in key} == {"same", "different"}
    assert any(k["slice"] == "same_name_non_match" for k in key)


def test_teaching_hospitals_are_not_in_the_pool(built, tmp_path):
    paths, stats, con = built
    row = op_row("r12", "", "666", "", "", "1000000.00")
    row["Covered_Recipient_Type"] = "Covered Recipient Teaching Hospital"
    again, new = _rebuild_with_op_rows(paths, tmp_path, [row])
    assert new["records_by_source"] == stats["records_by_source"]
    total = con.execute(f"SELECT sum(amount_usd) FROM read_parquet('{again.stripped / 'open_payments_payments.parquet'}')").fetchone()[0]
    assert total == pytest.approx(260.5)


def test_shared_address_slice_counts_focus_peers_only(built, tmp_path):
    paths, _, con = built
    with paths.nppes.open() as f:
        rows = list(csv.DictReader(f))
    # 19 other providers at the cardiologist's address are not enough unless they are cardiologists too.
    others = [nppes_row(f"20000{i:05d}", "DOE", f"N{i}", taxonomy="207Q00000X") for i in range(40)]
    peers = [nppes_row(f"30000{i:05d}", "ROE", f"N{i}", street="55 HEART WAY", zip_code="750011234") for i in range(20)]
    again = dataclasses.replace(paths, nppes=write_csv(tmp_path / "nppes.csv", rows + others + peers),
                                stripped=tmp_path / "stripped", gold=tmp_path / "gold", report=None)
    build(again, {**SCOPE, "perturbations": {"per_type": 0}})
    labeled = con.execute(f"""SELECT count(*) FROM read_parquet('{again.gold / 'slices.parquet'}') s
                              JOIN read_parquet('{again.stripped / 'nppes.parquet'}') n USING (record_id)
                              WHERE s.slice = 'shared_group_address'""").fetchone()[0]
    assert labeled == 20


def test_report_has_no_provider_names(built):
    paths, _, _ = built
    text = paths.report.read_text()
    assert "Leak check: passed" in text
    for name in ("SMITH", "Smith", "NGUYEN", "Tran", "Mystery"):
        assert name not in text


def test_build_is_deterministic(built, tmp_path):
    paths, _, con = built
    again = Paths(paths.nppes, paths.op_general, paths.op_profiles, paths.partd_provider, paths.partd_drug,
                  tmp_path / "stripped", tmp_path / "gold")
    build(again, SCOPE)
    for name in ("identity_records", "perturbed_records"):
        assert query(con, paths.stripped / f"{name}.parquet", "SELECT * FROM t ORDER BY record_id") == \
               query(con, again.stripped / f"{name}.parquet", "SELECT * FROM t ORDER BY record_id")


def _rebuild_with_op_rows(paths, tmp_path, extra_rows):
    with paths.op_general.open() as f:
        rows = list(csv.DictReader(f))
    op_general = write_csv(tmp_path / "op.csv", rows + extra_rows)
    again = Paths(paths.nppes, op_general, paths.op_profiles, paths.partd_provider, paths.partd_drug,
                  tmp_path / "stripped", tmp_path / "gold")
    return again, build(again, SCOPE)


def test_npi_typed_into_an_address_is_removed(built, tmp_path):
    paths, _, con = built
    rows = [op_row("r7", NPI_B, "222", "Tran", "Anna", "5.00", street=NPI_B),
            op_row("r8", NPI_B, "222", "Tran", "Anna", "6.00", street=f"12 Oak Rd {NPI_C}")]
    again, stats = _rebuild_with_op_rows(paths, tmp_path, rows)
    assert stats["op_rows_address_scrubbed"] == 2
    addresses = [r[0] for r in con.execute(
        f"SELECT address_1 FROM read_parquet('{again.stripped / 'open_payments_recipients.parquet'}')").fetchall()]
    assert None in addresses and "12 Oak Rd" in addresses


def test_npi_typed_into_registry_text_fields_is_removed(built, tmp_path):
    paths, _, con = built
    row = nppes_row("1999999943", "NEW", "PERSON", other_last=f"OLD {NPI_C}")
    row["Provider Credential Text"] = f"MD {NPI_C}"
    row["Provider Middle Name"] = NPI_D
    with paths.nppes.open() as f:
        rows = list(csv.DictReader(f))
    again = dataclasses.replace(paths, nppes=write_csv(tmp_path / "nppes.csv", rows + [row]),
                                stripped=tmp_path / "stripped", gold=tmp_path / "gold", report=None)
    stats = build(again, SCOPE)
    assert stats["nppes_rows_text_scrubbed"] == 1
    found = con.execute(f"""SELECT credential, middle_name, other_last_name FROM read_parquet('{again.stripped / 'nppes.parquet'}')
                            WHERE last_name = 'NEW'""").fetchone()
    assert found == ("MD", None, "OLD")


def test_shorter_digit_runs_in_addresses_are_kept(built, tmp_path):
    paths, _, con = built
    again, stats = _rebuild_with_op_rows(paths, tmp_path, [op_row("r9", NPI_B, "222", "Tran", "Anna", "5.00", street="123456789 Oak Rd")])
    assert stats["op_rows_address_scrubbed"] == 0
    addresses = [r[0] for r in con.execute(
        f"SELECT address_1 FROM read_parquet('{again.stripped / 'open_payments_recipients.parquet'}')").fetchall()]
    assert "123456789 Oak Rd" in addresses


def test_failed_leak_check_deletes_stripped_tables(built, tmp_path, monkeypatch):
    paths, _, _ = built

    def fail(*args):
        raise leak.LeakError("planted")

    monkeypatch.setattr(leak, "check_stripped", fail)
    stripped = tmp_path / "stripped"
    with pytest.raises(leak.LeakError):
        _rebuild_with_op_rows(paths, tmp_path, [])
    assert not stripped.exists()


def test_ids_never_truncate_past_their_padding():
    con = duckdb.connect()
    expression = build_module._padded("'E'", "n")
    rows = con.execute(f"SELECT {expression} FROM (VALUES (1), (1234567), (12345678), (12345679)) t(n)").fetchall()
    assert [r[0] for r in rows] == ["E0000001", "E1234567", "E12345678", "E12345679"]


def _integrity_tables(entities, records, membership):
    con = duckdb.connect()
    con.execute("CREATE TABLE entities (npi VARCHAR, entity_id VARCHAR)")
    con.execute("CREATE TABLE k_all (record_id VARCHAR, source VARCHAR, npi VARCHAR)")
    con.execute("CREATE TABLE membership (record_id VARCHAR, source VARCHAR, entity_id VARCHAR)")
    for table, rows in (("entities", entities), ("k_all", records), ("membership", membership)):
        if rows:
            con.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' * len(rows[0]))})", rows)
    return con


GOOD = dict(entities=[("1", "E1"), ("2", "E2")],
            records=[("np_1", "nppes", "1"), ("op_1", "open_payments", "1"), ("op_2", "open_payments", None)],
            membership=[("np_1", "nppes", "E1"), ("op_1", "open_payments", "E1")])


def test_integrity_passes_good_gold():
    build_module.check_integrity(_integrity_tables(**GOOD))


@pytest.mark.parametrize("change, message", [
    ({"entities": [("1", "E1"), ("2", "E1")]}, "one entity"),                    # the truncation bug: two NPIs, one ID
    ({"records": [*GOOD["records"], ("np_1", "part_d", "1")]}, "unique"),
    ({"records": [*GOOD["records"], ("pd_1", "part_d", "9")]}, "has an entity"),
    ({"membership": [*GOOD["membership"], ("np_2", "nppes", "E1")]}, "one registry record"),
])
def test_integrity_catches_broken_gold(change, message):
    with pytest.raises(build_module.IntegrityError, match=message):
        build_module.check_integrity(_integrity_tables(**{**GOOD, **change}))


def _handcheck_files(tmp_path, verdicts, gold):
    write_csv(tmp_path / "handcheck_sample.csv", [{"pair": i, "verdict": v, "notes": ""} for i, v in enumerate(verdicts, 1)])
    write_csv(tmp_path / "handcheck_key.csv", [{"pair": i, "gold": g, "slice": "x"} for i, g in enumerate(gold, 1)])
    return tmp_path


def test_handcheck_score_counts_agreement_and_reports_pairs_only(tmp_path):
    result = handcheck.score(_handcheck_files(tmp_path, ["same", "Different ", "same", "unsure", ""],
                                              ["same", "different", "different", "same", "same"]))
    assert result == {"total": 5, "done": 4, "blank": 1, "unsure": 1, "agree": 2, "disagree": [3], "invalid": []}


def test_handcheck_flags_invalid_verdicts(tmp_path):
    assert handcheck.score(_handcheck_files(tmp_path, ["yes"], ["same"]))["invalid"] == [1]


def test_state_scope_gives_each_state_its_own_seed():
    tx = build_module.state_scope({"seed": 1}, "TX")
    ca = build_module.state_scope({"seed": 1}, "CA")
    assert tx["state"] == "TX" and ca["state"] == "CA" and tx["seed"] != ca["seed"]


def test_clean_keeps_only_the_manifest(tmp_path):
    remove = ["raw/nppes/snapshot.zip", "raw/nppes/npidata_pfile.csv", "raw/partd_drug/drug.csv", "raw/partd_drug/drug.csv.done"]
    for name in ["raw/manifest.json", *remove]:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).touch()
    assert {p.relative_to(tmp_path).as_posix() for p in clean.removable(tmp_path / "raw")} == set(remove)


def test_clean_refuses_until_every_build_exists(tmp_path):
    splits = {"dev": ["TX"], "heldout": ["CA"]}
    (tmp_path / "dev" / "TX" / "gold").mkdir(parents=True)
    (tmp_path / "dev" / "TX" / "gold" / "build_stats.json").touch()
    assert clean.missing_builds(tmp_path, splits) == ["heldout/CA"]


# --- leak check -------------------------------------------------------------------------------------------------

def _table(con, path, sql):
    con.execute(f"COPY ({sql}) TO '{path}' (FORMAT parquet)")


@pytest.mark.parametrize("value, valid", [(NPI_A, True), (NPI_B, True), ("1234567890", False), ("123456789", False), ("abcdefghij", False)])
def test_npi_check_digit(value, valid):
    assert leak.is_valid_npi(value) is valid


def test_assert_clean_catches_a_buried_npi():
    with pytest.raises(leak.LeakError) as error:
        leak.assert_clean(f"Found provider, ref {NPI_A}, in Dallas")
    assert NPI_A not in str(error.value)  # the error itself must not repeat the identifier


def test_assert_clean_allows_ordinary_numbers():
    leak.assert_clean("Paid $1,234.56 on 2024-03-01, ZIP 75201, 752011234, id np_000123, cost 1234567890.12")


def test_check_catches_planted_known_npi(tmp_path):
    con = duckdb.connect()
    _table(con, tmp_path / "nppes.parquet", f"SELECT 'np_1' AS record_id, 'see {NPI_A}' AS last_name")
    with pytest.raises(leak.LeakError, match="known NPI"):
        leak.check_stripped(tmp_path, {NPI_A})


def test_check_catches_unknown_value_passing_checksum(tmp_path):
    con = duckdb.connect()
    _table(con, tmp_path / "nppes.parquet", f"SELECT 'np_1' AS record_id, 'x{NPI_B}x' AS last_name")
    with pytest.raises(leak.LeakError, match="check digit"):
        leak.check_stripped(tmp_path, set())


def test_check_catches_forbidden_and_unlisted_columns(tmp_path):
    con = duckdb.connect()
    _table(con, tmp_path / "nppes.parquet", "SELECT 'np_1' AS record_id, 'x' AS provider_npi, 'y' AS surprise")
    with pytest.raises(leak.LeakError) as error:
        leak.check_stripped(tmp_path, set())
    assert "provider_npi: forbidden" in str(error.value) and "surprise: column is not in the allowlist" in str(error.value)


def test_check_catches_unknown_table_and_empty_dir(tmp_path):
    with pytest.raises(leak.LeakError, match="No stripped tables"):
        leak.check_stripped(tmp_path, set())
    con = duckdb.connect()
    _table(con, tmp_path / "mystery.parquet", "SELECT 1 AS record_id")
    with pytest.raises(leak.LeakError, match="not in the allowlist"):
        leak.check_stripped(tmp_path, set())


def test_order_leak_detects_sorted_ids():
    sorted_pairs = [(float(i), float(i)) for i in range(200)]
    with pytest.raises(leak.LeakError):
        leak.check_order_leak(sorted_pairs)


def test_order_leak_passes_shuffled_and_skips_small():
    rng = random.Random(1)
    ys = list(range(1000))
    rng.shuffle(ys)
    assert abs(leak.check_order_leak([(float(i), float(y)) for i, y in enumerate(ys)])) < 0.1
    assert leak.check_order_leak([(1.0, 1.0)]) is None


# --- perturbations ----------------------------------------------------------------------------------------------

RECORD = {"record_id": "op_1", "first_name": "ROBERT", "last_name": "JOHNSON", "middle_name": "ALAN", "suffix": None,
          "address_1": "100 NORTH MAIN STREET", "address_2": "SUITE 200", "city": "DALLAS", "state": "TX", "zip": "75201"}
DONORS = [{**RECORD, "record_id": "np_9", "address_1": "9 ELM RD", "address_2": None, "city": "AUSTIN", "zip": "78701"}]


@pytest.mark.parametrize("kind", perturb.PERTURBATIONS)
def test_each_perturbation_changes_the_record_and_keeps_identity_id(kind):
    changed = perturb.PERTURBATIONS[kind](RECORD, random.Random(3))
    assert changed is not None and changed != RECORD and changed["record_id"] == "op_1"


def test_typo_stays_one_edit_and_keeps_first_letter():
    for seed in range(200):
        changed = perturb.typo(RECORD, random.Random(seed))
        field = "last_name" if changed["last_name"] != RECORD["last_name"] else "first_name"
        assert changed[field][0] == RECORD[field][0]
        assert abs(len(changed[field]) - len(RECORD[field])) <= 1


def test_nickname_both_directions_and_inapplicable():
    assert perturb.nickname(RECORD, random.Random(0))["first_name"] in {"BOB", "ROB", "BOBBY"}
    assert perturb.nickname({**RECORD, "first_name": "BOB"}, random.Random(0))["first_name"] == "ROBERT"
    assert perturb.nickname({**RECORD, "first_name": "ZEPHYRINE"}, random.Random(0)) is None


def test_initial_only():
    changed = perturb.initial_only(RECORD, random.Random(0))
    assert changed["first_name"] == "R" and changed["middle_name"] == "A"
    assert perturb.initial_only({**RECORD, "first_name": "R"}, random.Random(0)) is None


def test_swapped_name_inapplicable_when_same():
    assert perturb.swapped_name({**RECORD, "first_name": "LEE", "last_name": "LEE"}, random.Random(0)) is None


def test_address_abbreviation():
    changed = perturb.address_abbreviation(RECORD, random.Random(0))
    assert changed["address_1"] == "100 N MAIN ST" and changed["address_2"] == "STE 200"
    assert perturb.address_abbreviation({**RECORD, "address_1": "100 ELM", "address_2": None}, random.Random(0)) is None


def test_old_address_comes_from_a_different_zip():
    changed = perturb.old_address(RECORD, random.Random(0), DONORS)
    assert changed["zip"] == "78701" and changed["city"] == "AUSTIN" and changed["last_name"] == RECORD["last_name"]
    assert perturb.old_address(RECORD, random.Random(0), [RECORD]) is None


def test_generate_is_deterministic_and_tagged():
    records = [{**RECORD, "record_id": f"op_{i}"} for i in range(5)]
    first = perturb.generate(records, DONORS, per_type=2, seed=1)
    assert first == perturb.generate(records, DONORS, per_type=2, seed=1)
    assert {item["type"] for item in first} == set(perturb.ALL_TYPES)
    assert all(sum(item["type"] == t for item in first) == 2 for t in perturb.ALL_TYPES)
