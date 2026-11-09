"""Free the raw downloads once the builds exist. Shows what it would do unless given --yes.

  python -m pipeline.clean        # dry run
  python -m pipeline.clean --yes  # delete

Everything comes back with `python -m pipeline.download`, for as long as the pinned files stay up. NPPES comes from
NBER's archive of monthly files; the manifest keeps each file's checksum, so a later download can be compared.
"""

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def removable(raw: Path) -> list[Path]:
    return sorted(p for p in raw.rglob("*") if p.is_file() and p.name != "manifest.json")


def missing_builds(data: Path, splits: dict[str, list[str]]) -> list[str]:
    return [f"{split}/{state}" for split, states in splits.items() for state in states
            if not (data / split / state / "gold" / "build_stats.json").exists()]


def main() -> None:
    data = ROOT / "data"
    splits = tomllib.loads((ROOT / "config" / "scope.toml").read_text())["splits"]
    missing = missing_builds(data, splits)
    if missing:
        sys.exit(f"Refusing to clean: no completed build for {', '.join(missing)}")
    files = removable(data / "raw")
    total = sum(p.stat().st_size for p in files)
    for path in files:
        print(f"{path.stat().st_size / 1e9:7.2f} GB  {path.relative_to(data)}")
    if "--yes" in sys.argv[1:]:
        for path in files:
            path.unlink()
        print(f"Deleted {total / 1e9:.2f} GB")
    else:
        print(f"Would delete {total / 1e9:.2f} GB. Re-run with --yes to delete.")


if __name__ == "__main__":
    main()
