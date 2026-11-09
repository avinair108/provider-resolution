"""Leak check: nothing the system can query may carry an identifier that reveals the gold.

Runs at build time over data/stripped (failing the build) and on every tool result at run time.
"""

import re
from pathlib import Path

import duckdb

FORBIDDEN_COLUMN = re.compile(r"npi|profile_?id|ein\b|tin\b|license|ccn|teaching_hospital_id|entity_id", re.IGNORECASE)
_TEN_DIGITS = re.compile(r"(?<![\d.])\d{10}(?![\d])")

# The only columns each stripped table may have. An allowlist, so a new upstream ID column can't slip through.
ALLOWED_COLUMNS = {
    "nppes": {
        "record_id", "last_name", "first_name", "middle_name", "suffix", "credential",
        "other_last_name", "other_first_name", "other_middle_name", "other_credential",
        "practice_address_1", "practice_address_2", "practice_city", "practice_state", "practice_zip",
        "mailing_address_1", "mailing_address_2", "mailing_city", "mailing_state", "mailing_zip",
        "taxonomy_codes", "primary_taxonomy", "last_update",
    },
    "open_payments_recipients": {
        "record_id", "last_name", "first_name", "middle_name", "suffix",
        "address_1", "address_2", "city", "state", "zip", "specialties", "n_payments",
    },
    "open_payments_payments": {
        "payment_id", "record_id", "payer", "amount_usd", "payment_date", "n_payments_in_total",
        "payment_form", "payment_nature", "products",
    },
    "part_d_prescribers": {
        "record_id", "last_name", "first_name", "middle_initial", "credential",
        "address_1", "address_2", "city", "state", "zip", "prescriber_type",
        "total_claims", "total_drug_cost", "total_day_supply", "total_beneficiaries",
    },
    "part_d_drugs": {
        "record_id", "brand_name", "generic_name", "total_claims", "total_30day_fills",
        "total_day_supply", "total_drug_cost", "total_beneficiaries",
    },
    "identity_records": {
        "record_id", "source", "last_name", "first_name", "middle_name", "suffix",
        "address_1", "address_2", "city", "state", "zip",
    },
    "perturbed_records": {
        "record_id", "source", "last_name", "first_name", "middle_name", "suffix",
        "address_1", "address_2", "city", "state", "zip",
    },
}


class LeakError(RuntimeError):
    pass


def is_valid_npi(value: str) -> bool:
    """NPI check digit: Luhn over the number with the 80840 issuer prefix."""
    if not re.fullmatch(r"\d{10}", value):
        return False
    total = 0
    for i, ch in enumerate(reversed("80840" + value)):
        digit = int(ch)
        if i % 2 == 1:
            digit = digit * 2 - 9 if digit > 4 else digit * 2
        total += digit
    return total % 10 == 0


def npi_like_values(text: str) -> list[str]:
    return [m for m in _TEN_DIGITS.findall(text) if is_valid_npi(m)]


def assert_clean(text: str) -> None:
    """Run-time gate for tool results."""
    found = npi_like_values(text)
    if found:
        raise LeakError(f"{len(found)} NPI-like value(s) in a tool result")


def check_stripped(stripped_dir: Path, npis: set[str]) -> dict:
    """Column allowlist, then a value scan of every cell. Raises LeakError listing every problem. Error messages
    carry table and column only, so a failed build never prints the identifier it found."""
    problems, scanned = [], {}
    con = duckdb.connect()
    files = sorted(Path(stripped_dir).glob("*.parquet"))
    if not files:
        raise LeakError(f"No stripped tables in {stripped_dir}")
    for path in files:
        table = path.stem
        if table not in ALLOWED_COLUMNS:
            problems.append(f"{table}: table is not in the allowlist")
            continue
        schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
        columns = {row[0]: row[1] for row in schema}
        for column in columns:
            if column not in ALLOWED_COLUMNS[table]:
                problems.append(f"{table}.{column}: column is not in the allowlist")
            if FORBIDDEN_COLUMN.search(column):
                problems.append(f"{table}.{column}: forbidden column name")
        for column, column_type in columns.items():
            text = f'CAST("{column}" AS VARCHAR)'
            cells = con.execute(
                f"""SELECT DISTINCT unnest(regexp_extract_all({text}, '[0-9]{{10}}')) FROM read_parquet('{path}')
                    WHERE regexp_matches({text}, '[0-9]{{10}}')"""
            ).fetchall()
            digits = {c[0] for c in cells}
            known = digits & npis
            if known:
                problems.append(f"{table}.{column}: {len(known)} known NPI value(s)")
            if column_type == "VARCHAR" or column_type.endswith("[]"):
                checksum = [d for d in digits - known if npi_like_values(d)]
                if checksum:
                    problems.append(f"{table}.{column}: {len(checksum)} value(s) passing the NPI check digit")
        scanned[table] = len(columns)
    if problems:
        raise LeakError("Leak check failed:\n  " + "\n  ".join(problems))
    return scanned


def check_order_leak(pairs: list[tuple[float, float]], threshold: float = 0.1) -> float | None:
    """Across sources, record order for the same provider must be unrelated, or IDs alone would reveal links.
    Returns the rank correlation, or None with too few pairs to say."""
    n = len(pairs)
    if n < 100:
        return None

    def ranks(values: list[float]) -> list[int]:
        order = sorted(range(n), key=values.__getitem__)
        result = [0] * n
        for rank, index in enumerate(order):
            result[index] = rank
        return result

    x, y = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    mean = (n - 1) / 2
    numerator = sum((a - mean) * (b - mean) for a, b in zip(x, y))
    denominator = sum((a - mean) ** 2 for a in x)
    rho = numerator / denominator
    if abs(rho) > threshold:
        raise LeakError(f"Record IDs correlate across sources (rho={rho:.3f})")
    return rho
