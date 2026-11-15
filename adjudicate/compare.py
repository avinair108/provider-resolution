"""Line up Phase 4 runs (ablations, models, hosts) in one table. Reads only the runs' summary.json files.

  python -m adjudicate.compare runs/*_abl_*            # any run directories
  python -m adjudicate.compare --tags abl_ref abl_scores --out reports/phase4_ablations_dev_TX_tune.md

Comparable across bands: errors among all cardiologists' records in the state (the band only decides which records
the model sees). Ranges are min-max over repeats.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _errors(result: dict) -> int:
    r = result["records"]["focus"]["overall"]
    return r["n"] - r["correct"] - r["none_correct"]


def _rng(values, fmt="{:.1%}"):
    values = [v for v in values if v is not None]
    if not values:
        return "–"
    lo, hi = min(values), max(values)
    return fmt.format(lo) if fmt.format(lo) == fmt.format(hi) else f"{fmt.format(lo)}–{fmt.format(hi)}"


def describe(s: dict) -> str:
    parts = [s["llm"]["model"].split("/")[-1], s["llm"]["provider"].replace("openrouter:", "")]
    if s.get("effort"):
        parts.append(f"effort {s['effort']}")
    parts.append(f"band [{s['band'][0]}, {s['band'][1]})")
    if s.get("order", "shuffled") != "shuffled":
        parts.append(f"order: {s['order']}")
    if s.get("show_scores"):
        parts.append("Splink scores shown")
    if s.get("repair") is False:
        parts.append("no repair")
    return ", ".join(parts)


def row(s: dict) -> dict:
    combined = s["results"]["splink_llm"]
    return {
        "variant": describe(s), "run": s["run_id"], "subset": s["subset"], "n_band": s["n_band"],
        "splink_errors": _errors(s["results"]["splink"]),
        "errors": [_errors(r) for r in combined],
        "accuracy": [r["records"]["focus"]["overall"]["accuracy"] for r in combined],
        "fixed": [f["focus"]["fixed"] for f in s["flips"]], "broken": [f["focus"]["broken"] for f in s["flips"]],
        "fallback": [p["fallback"] for p in s["process"]], "ece": [p["ece"] for p in s["process"]],
        "cost": s["llm"]["cost_usd"], "live": s["llm"]["live_calls"], "repeats": s["repeats"],
    }


def to_markdown(rows: list[dict], title: str) -> str:
    base = rows[0]["splink_errors"]
    lines = [f"# {title}", "",
             f"Subset `{rows[0]['subset']}`. Splink alone makes {base:,} errors on cardiologists' records statewide; "
             "every variant below changes only the band records it sends to the model. Ranges over repeats. Cost is "
             "live spend for that run (cached answers are free). Aggregates only.", "",
             "| Variant | Band records | Errors (cardiologists, statewide) | Accuracy | Fixed | Broken | Fallbacks | ECE | Cost |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['variant']} | {r['n_band']:,} | {_rng(r['errors'], '{:,}')} | {_rng(r['accuracy'], '{:.2%}')} | "
                     f"{_rng(r['fixed'], '{:,}')} | {_rng(r['broken'], '{:,}')} | {_rng(r['fallback'], '{:,}')} | "
                     f"{_rng(r['ece'], '{:.3f}')} | ${r['cost']:.2f} |")
    lines += ["", "Runs: " + ", ".join(f"`{r['run']}`" for r in rows)]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="run directories")
    parser.add_argument("--tags", nargs="*", default=[], help="latest run whose id ends with each tag")
    parser.add_argument("--title", default="Phase 4 ablations")
    parser.add_argument("--out")
    args = parser.parse_args()
    dirs = [Path(r) for r in args.runs]
    for tag in args.tags:
        found = sorted((ROOT / "runs").glob(f"*_adjudicate_*_{tag}"))
        if not found:
            raise SystemExit(f"no run tagged {tag}")
        dirs.append(found[-1])
    rows = [row(json.loads((d / "summary.json").read_text())) for d in dirs]
    text = to_markdown(rows, args.title)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
