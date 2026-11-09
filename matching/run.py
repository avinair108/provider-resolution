"""Phase 3: the Splink baseline and the exact-match baselines, scored the same way.

  python -m matching.run                       # Texas dev, 3 Splink repeats
  python -m matching.run --repeats 1 --tag try

Writes runs/<run_id>/ (summary.json, report.md, model_<seed>.json, splink_<seed>.parquet). Thresholds are chosen on
this split; the held-out state is scored only with thresholds fixed here.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import duckdb

from matching import candidates, evaluate, inputs

ROOT = Path(__file__).resolve().parent.parent
BASELINES = {"exact_name": ["last_name", "first_name"], "exact_name_zip": ["last_name", "first_name", "zip"]}


def score(con, name: str, pair_threshold: float | None = None, record_threshold: float | None = None) -> dict:
    curve = evaluate.pair_curve(con, name)
    evaluate.record_resolution(con, name)
    rcurve = evaluate.record_curve(con)
    if pair_threshold is None:
        pair_threshold = evaluate.best_threshold(lambda t: evaluate.pair_metrics(curve, t)["f1"])
    if record_threshold is None:
        record_threshold = evaluate.best_threshold(lambda t: evaluate.record_metrics(rcurve, t)["accuracy"])
    source_pairs = sorted({sp for sp, *_ in curve["totals"]})
    groups = sorted({g for g, *_ in rcurve})
    return {
        "pair_threshold": pair_threshold,
        "record_threshold": record_threshold,
        "pairs": {scope: {"all_sources": evaluate.pair_metrics(curve, pair_threshold, scope),
                          **{sp: evaluate.pair_metrics(curve, pair_threshold, scope, sp) for sp in source_pairs}}
                  for scope in ("all", "focus")},
        "records": {scope: {(g or "overall"): evaluate.record_metrics(rcurve, record_threshold, g, scope) for g in [None, *groups]}
                    for scope in ("all", "focus")},
        "pr_curve": [evaluate.pair_metrics(curve, t, "focus") for t in (0.5, 0.8, 0.9, 0.95, 0.99)],
        "calibration": evaluate.top_candidate_calibration(con),
        "clusters": evaluate.cluster_metrics(con, name, pair_threshold),
    }


def _fmt(x, pct=True):
    if x is None:
        return "–"
    return f"{x:.1%}" if pct else f"{x:,}"


def _range(values):
    values = [v for v in values if v is not None]
    if not values:
        return "–"
    lo, hi = min(values), max(values)
    return f"{lo:.1%}" if abs(hi - lo) < 0.0005 else f"{lo:.1%}–{hi:.1%}"


def to_markdown(summary: dict) -> str:
    results = summary["results"]
    names = list(results)
    first = lambda n: results[n][0]
    lines = [f"# Phase 3: matching baselines ({summary['split']} {summary['state']})", "",
             f"Run `{summary['run_id']}`. Splink repeats: {summary['repeats']}. Ranges are min–max over repeats. "
             "Aggregates only.", "",
             "## Resolving each Open Payments and Part D record to its registry record", "",
             "Cardiologists' records. A record resolves to its top candidate if unique and above the threshold.", "",
             "| Method | Threshold | Accuracy | Precision | Recall | Wrong links | Missed |", "|---|---|---|---|---|---|---|"]
    for n in names:
        rs = [r["records"]["focus"]["overall"] for r in results[n]]
        lines.append(f"| {n} | {first(n)['record_threshold']} | {_range([r['accuracy'] for r in rs])} | "
                     f"{_range([r['precision'] for r in rs])} | {_range([r['recall'] for r in rs])} | "
                     f"{rs[0]['wrong'] + rs[0]['linked_without_counterpart']:,} | {rs[0]['missed']:,} |")
    lines += ["", "Same, all providers in the pool:", "", "| Method | Accuracy | Precision | Recall |", "|---|---|---|---|"]
    for n in names:
        rs = [r["records"]["all"]["overall"] for r in results[n]]
        lines.append(f"| {n} | {_range([r['accuracy'] for r in rs])} | {_range([r['precision'] for r in rs])} | {_range([r['recall'] for r in rs])} |")
    lines += ["", "## Pairs across sources (cardiologists)", "",
              "| Method | Threshold | Precision | Recall | F1 | Blocking recall | B-cubed F1 (all) | Over-merged clusters |",
              "|---|---|---|---|---|---|---|---|"]
    for n in names:
        ps = [r["pairs"]["focus"]["all_sources"] for r in results[n]]
        cs = [r["clusters"] for r in results[n]]
        lines.append(f"| {n} | {first(n)['pair_threshold']} | {_range([p['precision'] for p in ps])} | {_range([p['recall'] for p in ps])} | "
                     f"{_range([p['f1'] for p in ps])} | {_range([p['blocking_recall'] for p in ps])} | "
                     f"{_range([c['bcubed_f1'] for c in cs])} | {cs[0]['overmerged_clusters']:,} |")
    lines += ["", "## By slice (cardiologists' records, record resolution, first repeat)", "",
              "| Slice | Records | " + " | ".join(f"{n} accuracy" for n in names) + " |", "|---|---|" + "---|" * len(names)]
    slice_groups = sorted(g for g in first(names[0])["records"]["focus"] if g.startswith("slice:"))
    for g in slice_groups:
        cells = [first(n)["records"]["focus"].get(g, {}) for n in names]
        lines.append(f"| {g[6:]} | {cells[0].get('n', 0):,} | " + " | ".join(_fmt(c.get("accuracy")) for c in cells) + " |")
    for n in names:
        cal = first(n)["calibration"]
        if cal["ece"] is not None and n.startswith("splink"):
            lines += ["", f"## Calibration of the top candidate ({n})", "",
                      f"Expected calibration error {cal['ece']:.3f} over {cal['n']:,} records.", "",
                      "| p | Records | Mean p | Correct |", "|---|---|---|---|"]
            lines += [f"| {b['low']:.1f}–{b['high']:.1f} | {b['n']:,} | {b['mean_confidence']:.3f} | {b['accuracy']:.3f} |" for b in cal["bins"]]
            break
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--state", default="TX")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    base = ROOT / "data" / args.split / args.state
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_matching_{args.split}_{args.state}" + (f"_{args.tag}" if args.tag else "")
    out = ROOT / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    counts = inputs.load(con, base / "stripped")
    evaluate.load_gold(con, base / "gold")
    results, timings = {}, {}
    for name, columns in BASELINES.items():
        t = time.time()
        candidates.exact_baseline(con, name, columns)
        results[name] = [score(con, name)]
        timings[name] = round(time.time() - t, 1)
        print(f"{name}: {results[name][0]['records']['focus']['overall']}", flush=True)
    results["splink"] = []
    for seed in range(args.repeats):
        t = time.time()
        info = candidates.splink_model(con, "splink", seed=seed)
        (out / f"model_{seed}.json").write_text(json.dumps(info["model"], indent=2))
        if seed == 0:
            con.execute(f"COPY splink TO '{out / 'splink_0.parquet'}' (FORMAT parquet)")
        results["splink"].append(score(con, "splink"))
        timings[f"splink_{seed}"] = round(time.time() - t, 1)
        print(f"splink seed {seed} ({timings[f'splink_{seed}']}s): {results['splink'][-1]['records']['focus']['overall']}", flush=True)
    summary = {"run_id": run_id, "split": args.split, "state": args.state, "repeats": args.repeats,
               "input_records": counts, "timings_s": timings, "results": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "report.md").write_text(to_markdown(json.loads(json.dumps(summary, default=str))))
    print(f"Report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
