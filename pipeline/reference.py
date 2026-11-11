"""Small public reference tables the pipeline and agent read, kept in data/reference/ (git-ignored, ~0.5 MB).

  python -m pipeline.reference
"""

import csv
import hashlib
import json
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "data" / "reference"
NUCC = REFERENCE / "nucc_taxonomy.csv"


def fetch_nucc() -> Path:
    spec = tomllib.loads((ROOT / "config" / "sources.toml").read_text())["nucc_taxonomy"]
    REFERENCE.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(spec["url"], headers={"User-Agent": "Mozilla/5.0 (provider-resolution)"})
    with urllib.request.urlopen(request, timeout=60) as r:
        body = r.read()
    partial = NUCC.with_suffix(".partial")
    partial.write_bytes(body)
    partial.replace(NUCC)
    (REFERENCE / "manifest.json").write_text(json.dumps(
        {"nucc_taxonomy": {"url": spec["url"], "version": spec["version"], "bytes": len(body),
                           "sha256": hashlib.sha256(body).hexdigest()}}, indent=2))
    return NUCC


def nucc_names(path: Path = NUCC) -> dict[str, str]:
    """Taxonomy code -> display name, e.g. 390200000X -> 'Student in an Organized Health Care Education/Training
    Program'. Empty if the file hasn't been fetched."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", errors="replace") as f:
        return {row["Code"].strip(): (row.get("Display Name") or row.get("Classification") or "").strip()
                for row in csv.DictReader(f) if row.get("Code")}


if __name__ == "__main__":
    path = fetch_nucc()
    print(f"{path}: {len(nucc_names(path)):,} codes")
