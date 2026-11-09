"""Score the hand check of gold links against its hidden key.

  python -m pipeline.handcheck

Prints pair numbers only, never names, so the output is safe to share.
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VALID = {"same", "different", "unsure"}


def score(gold_dir: Path) -> dict:
    with (gold_dir / "handcheck_sample.csv").open(newline="", encoding="utf-8-sig") as f:  # Excel adds a BOM
        rows = list(csv.DictReader(f))
    if rows and "pair" not in rows[0]:
        raise SystemExit("handcheck_sample.csv has an older layout (no 'pair' column). It may have been saved over the "
                         "current file by a spreadsheet that still had it open.")
    verdicts = {row["pair"]: (row["verdict"] or "").strip().lower() for row in rows}
    with (gold_dir / "handcheck_key.csv").open(newline="") as f:
        key = {row["pair"]: row["gold"] for row in csv.DictReader(f)}
    filled = {pair: v for pair, v in verdicts.items() if v}
    decided = {pair: v for pair, v in filled.items() if v in ("same", "different")}
    return {
        "total": len(key),
        "done": len(filled),
        "blank": len(verdicts) - len(filled),
        "unsure": sum(v == "unsure" for v in filled.values()),
        "agree": sum(key[pair] == v for pair, v in decided.items()),
        "disagree": sorted(int(pair) for pair, v in decided.items() if key[pair] != v),
        "invalid": sorted(int(pair) for pair, v in filled.items() if v not in VALID),
    }


def main() -> None:
    result = score(ROOT / "data" / "dev" / "TX" / "gold")
    print(f"Filled in {result['done']} of {result['total']} ({result['blank']} blank, {result['unsure']} unsure)")
    print(f"Agree with gold: {result['agree']}")
    if result["invalid"]:
        print(f"Not same / different / unsure, please fix: pairs {result['invalid']}")
    if result["disagree"]:
        print(f"Disagree with gold, look at these with Claude: pairs {result['disagree']}")


if __name__ == "__main__":
    main()
