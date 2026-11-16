"""Supervised baseline: a gradient-boosted classifier over record-registry candidate pairs, trained on NPI labels.

  python -m matching.supervised train                                   # fit on Texas, report on the held-back band
  python -m matching.supervised apply --split heldout --state CA --final --model runs/<train run>

Once labels exist, this is the matcher an LLM adjudicator has to beat: Splink, as used here, never sees labels.
Training uses Texas records outside the Phase 4 evaluation band; a fifth of them (by hash) is held back to pick the
decision threshold. Reported numbers come only from records the model never trained on: Texas's evaluation band, and
the held-out state, scored once with the Texas threshold.

Features per (record, registry candidate): Splink's score, rank and gaps; name agreements (exact, other name, first
word, Jaro-Winkler, nickname, initial only, middle initial); location (building, ZIP, street, city, ZIP3); how common
the names are; cardiology on either side; and how many candidates share the name. No ids, so nothing to memorise.
"""

import argparse
import hashlib
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from adjudicate import select
from adjudicate.run import band_outcomes
from matching import evaluate, inputs
from matching.run import score

ROOT = Path(__file__).resolve().parent.parent
MAX_RANK = 10
FEATURES = ["logit", "rank", "n_candidates", "gap_top", "margin", "last_exact", "last_alt", "last_token", "last_jw",
            "first_exact", "first_formal", "first_jw", "first_initial_only", "middle", "zip_same", "building_same",
            "street_same", "city_same", "zip3_same", "last_freq", "first_freq", "src_cardiology", "reg_cardiology",
            "is_part_d", "same_name_candidates"]


def build_features(con, stripped: Path, scores: str) -> None:
    """feat(record_id, registry, <FEATURES>) for each record's top MAX_RANK registry candidates. Reads no gold."""
    inputs.load(con, stripped)
    select.load_sources(con, stripped)
    select.registry_candidates(con, scores)
    con.execute(f"""
        CREATE OR REPLACE TABLE feat AS
        WITH src AS (SELECT * FROM open_payments UNION ALL SELECT * FROM part_d),
        c AS (SELECT * FROM cand_registry WHERE rank <= {MAX_RANK}),
        s AS (SELECT record_id, max(p) AS p_top, count(*) AS n_c, max(p) FILTER (WHERE rank = 2) AS p2 FROM c GROUP BY 1),
        lf AS (SELECT last_name, ln(count(*)) AS f FROM nppes GROUP BY 1),
        ff AS (SELECT first_name, ln(count(*)) AS f FROM nppes GROUP BY 1)
        SELECT c.record_id, c.registry,
               ln(greatest(c.p, 1e-9)) - ln(greatest(1 - c.p, 1e-9)) AS logit,
               c.rank, s.n_c AS n_candidates, s.p_top - c.p AS gap_top,
               CASE WHEN c.rank = 1 THEN c.p - coalesce(s.p2, 0) ELSE 0 END AS margin,
               (x.last_name = r.last_name)::INT AS last_exact,
               (x.last_name = r.last_name_alt AND r.last_name_alt <> r.last_name)::INT AS last_alt,
               (x.last_token = r.last_token)::INT AS last_token,
               jaro_winkler_similarity(coalesce(x.last_name, ''), coalesce(r.last_name, '')) AS last_jw,
               (x.first_name = r.first_name)::INT AS first_exact,
               (x.first_name_formal = r.first_name_formal)::INT AS first_formal,
               jaro_winkler_similarity(coalesce(x.first_name, ''), coalesce(r.first_name, '')) AS first_jw,
               (x.first_initial = r.first_initial AND (length(x.first_name) = 1 OR length(r.first_name) = 1))::INT AS first_initial_only,
               CASE WHEN x.middle_initial IS NULL OR r.middle_initial IS NULL THEN 0
                    WHEN x.middle_initial = r.middle_initial THEN 1 ELSE -1 END AS middle,
               (x.zip = r.zip OR x.zip = r.alt_zip)::INT AS zip_same,
               ((x.zip = r.zip AND x.street_number = r.street_number) OR (x.zip = r.alt_zip AND x.street_number = r.alt_street_number))::INT AS building_same,
               (x.street_name = r.street_name)::INT AS street_same,
               (x.city = r.city)::INT AS city_same,
               (left(x.zip, 3) = left(r.zip, 3))::INT AS zip3_same,
               lf.f AS last_freq, ff.f AS first_freq,
               (c.record_id IN (SELECT record_id FROM src_cardiology))::INT AS src_cardiology,
               (c.registry IN (SELECT record_id FROM src_cardiology))::INT AS reg_cardiology,
               (c.record_id LIKE 'pd\\_%' ESCAPE '\\')::INT AS is_part_d,
               count(*) OVER (PARTITION BY c.record_id, r.last_name, r.first_name) AS same_name_candidates
        FROM c JOIN s USING (record_id)
        JOIN src x ON x.unique_id = c.record_id
        JOIN nppes r ON r.unique_id = c.registry
        LEFT JOIN lf ON lf.last_name = r.last_name LEFT JOIN ff ON ff.first_name = r.first_name
    """)


def matrix(con, where: str = "true") -> tuple[np.ndarray, list[str], list[str]]:
    cols = ", ".join(f"CAST({f} AS DOUBLE) AS {f}" for f in FEATURES)
    arrays = con.execute(f"SELECT record_id, registry, {cols} FROM feat WHERE {where} ORDER BY record_id, registry").fetchnumpy()
    X = np.column_stack([np.asarray(arrays[f], dtype=float) for f in FEATURES])
    return X, list(arrays["record_id"]), list(arrays["registry"])


def scored_table(con, name: str, scores: str, records: list[str], registries: list[str], q: np.ndarray) -> None:
    """Splink's table with each record-registry pair re-scored by the classifier (0 outside the top MAX_RANK);
    Open Payments-Part D pairs keep Splink's score."""
    con.execute("CREATE OR REPLACE TEMP TABLE q (record_id VARCHAR, registry VARCHAR, q DOUBLE)")
    con.executemany("INSERT INTO q VALUES (?, ?, ?)", list(zip(records, registries, q.tolist())))
    con.execute(f"""
        CREATE OR REPLACE TABLE {name} AS
        SELECT s.a, s.b,
               CASE WHEN (s.a LIKE 'np\\_%' ESCAPE '\\') <> (s.b LIKE 'np\\_%' ESCAPE '\\') THEN coalesce(q.q, 0.0) ELSE s.p END AS p
        FROM {scores} s
        LEFT JOIN q ON q.registry = CASE WHEN s.a LIKE 'np\\_%' ESCAPE '\\' THEN s.a ELSE s.b END
                   AND q.record_id = CASE WHEN s.a LIKE 'np\\_%' ESCAPE '\\' THEN s.b ELSE s.a END
    """)


def record_accuracy(records: list[str], registries: list[str], q: np.ndarray, truth: dict, threshold: float) -> float:
    """Top candidate if unique and >= threshold, else none; right registry record or rightly none."""
    best: dict[str, tuple[float, str | None, int]] = {}
    for r, g, p in zip(records, registries, q):
        b = best.get(r)
        if b is None or p > b[0]:
            best[r] = (p, g, 1)
        elif p == b[0]:
            best[r] = (p, b[1], b[2] + 1)
    hits = [((g if (p >= threshold and n == 1) else None) == truth[r]) for r, (p, g, n) in best.items() if r in truth]
    return sum(hits) / len(hits) if hits else 0.0


def _held(record_id: str) -> bool:
    return int(hashlib.md5(f"supervised-threshold:{record_id}".encode()).hexdigest(), 16) % 5 == 0


def _baseline(split: str, state: str) -> Path:
    return sorted((ROOT / "runs").glob(f"*_matching_{split}_{state}_baseline"))[-1]


def train(args) -> None:
    t0 = time.time()
    base = _baseline("dev", args.state)
    thresholds = json.loads((base / "summary.json").read_text())["results"]["splink"][0]
    data = ROOT / "data" / "dev" / args.state
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"CREATE TABLE splink AS SELECT * FROM read_parquet('{base / 'splink_0.parquet'}')")
    build_features(con, data / "stripped", "splink")
    eval_band = [i["record_id"] for i in select.band(con, 0.01, 0.99, split="eval")]
    con.execute("CREATE OR REPLACE TABLE eval_band AS SELECT unnest(?::VARCHAR[]) AS record_id", [eval_band])
    evaluate.load_gold(con, data / "gold")  # labels: read after features, used only to fit and to score
    truth = dict(con.execute("SELECT record_id, true_registry FROM g_records").fetchall())

    X, recs, regs = matrix(con, "record_id IN (SELECT record_id FROM g_records) AND record_id NOT IN (SELECT record_id FROM eval_band)")
    y = np.array([truth[r] == g for r, g in zip(recs, regs)], dtype=int)
    held = np.array([_held(r) for r in recs])
    model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=63, l2_regularization=1.0,
                                           early_stopping=True, validation_fraction=0.1, random_state=0)
    model.fit(X[~held], y[~held])
    q_held = model.predict_proba(X[held])[:, 1]
    held_recs, held_regs = [r for r, h in zip(recs, held) if h], [g for g, h in zip(regs, held) if h]
    grid = [round(0.05 * i, 2) for i in range(1, 20)]
    record_t = max(grid, key=lambda t: (record_accuracy(held_recs, held_regs, q_held, truth, t), t))
    print(f"trained on {int((~held).sum()):,} pairs ({int(y[~held].sum()):,} matches) in {time.time() - t0:.0f}s; "
          f"threshold {record_t} picked on {int(held.sum()):,} held-back pairs", flush=True)

    # Score: every pair re-scored, but only the evaluation band (never trained on) is reported for Texas.
    Xa, ra, ga = matrix(con)
    scored_table(con, "supervised", "splink", ra, ga, model.predict_proba(Xa)[:, 1])
    result = score(con, "supervised", thresholds["pair_threshold"], record_t)
    band = band_outcomes(con, eval_band, record_t)
    evaluate.record_resolution(con, "splink")
    splink_band = band_outcomes(con, eval_band, thresholds["record_threshold"])
    out = ROOT / "runs" / (datetime.now().strftime("%Y%m%d-%H%M%S") + f"_supervised_dev_{args.state}" + (f"_{args.tag}" if args.tag else ""))
    out.mkdir(parents=True)
    (out / "model.pkl").write_bytes(pickle.dumps({"model": model, "features": FEATURES, "record_threshold": record_t}))
    importances = dict(zip(FEATURES, getattr(model, "feature_importances_", [None] * len(FEATURES))))
    summary = {"run_id": out.name, "split": "dev", "state": args.state, "baseline": base.name, "record_threshold": record_t,
               "train_pairs": int((~held).sum()), "train_matches": int(y[~held].sum()), "eval_band_records": len(eval_band),
               "band_accuracy": {"splink": _acc(splink_band), "supervised": _acc(band)},
               "results": {"supervised": [result]}, "wall_s": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "report.md").write_text(to_markdown(summary))
    print((out / "report.md").read_text())


def _acc(outcomes: dict) -> dict:
    out = {}
    for scope in ("all", "focus"):
        vals = [ok for focus, ok in outcomes.values() if focus or scope == "all"]
        out[scope] = sum(vals) / len(vals) if vals else None
    return out


def apply(args) -> None:
    if args.split != "dev" and not args.final:
        raise SystemExit("score the held-out state once, with the model frozen: pass --final")
    t0 = time.time()
    trained = json.loads((ROOT / args.model / "summary.json").read_text())
    bundle = pickle.loads((ROOT / args.model / "model.pkl").read_bytes())
    base = _baseline(args.split, args.state)
    dev_thresholds = json.loads((ROOT / "runs" / trained["baseline"] / "summary.json").read_text())["results"]["splink"][0]
    data = ROOT / "data" / args.split / args.state
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"CREATE TABLE splink AS SELECT * FROM read_parquet('{base / 'splink_0.parquet'}')")
    build_features(con, data / "stripped", "splink")
    band_ids = [i["record_id"] for i in select.band(con, 0.01, 0.99, split="all")]
    Xa, ra, ga = matrix(con)
    scored_table(con, "supervised", "splink", ra, ga, bundle["model"].predict_proba(Xa)[:, 1])
    evaluate.load_gold(con, data / "gold")
    record_t = bundle["record_threshold"]
    result = score(con, "supervised", dev_thresholds["pair_threshold"], record_t)
    band = band_outcomes(con, band_ids, record_t)
    evaluate.record_resolution(con, "splink")
    splink_band = band_outcomes(con, band_ids, dev_thresholds["record_threshold"])
    out = ROOT / "runs" / (datetime.now().strftime("%Y%m%d-%H%M%S") + f"_supervised_{args.split}_{args.state}" + (f"_{args.tag}" if args.tag else ""))
    out.mkdir(parents=True)
    summary = {"run_id": out.name, "split": args.split, "state": args.state, "baseline": base.name, "model": args.model,
               "record_threshold": record_t, "band_records": len(band_ids),
               "band_accuracy": {"splink": _acc(splink_band), "supervised": _acc(band)},
               "results": {"supervised": [result]}, "wall_s": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "report.md").write_text(to_markdown(summary))
    print((out / "report.md").read_text())


def to_markdown(s: dict) -> str:
    r = s["results"]["supervised"][0]["records"]
    b = s["band_accuracy"]
    pct = lambda x: "–" if x is None else f"{x:.1%}"
    errors = lambda m: m["n"] - m["correct"] - m["none_correct"]
    lines = [f"# Supervised baseline ({s['split']} {s['state']})", "",
             f"Run `{s['run_id']}`. Gradient-boosted classifier over record-registry candidates, trained on Texas NPI "
             f"labels outside the Phase 4 evaluation band; record threshold {s['record_threshold']} picked on held-back "
             "Texas records. Aggregates only.", ""]
    if s["split"] == "dev":
        lines += [f"Texas numbers come only from the {s['eval_band_records']:,} evaluation-band records, which training "
                  "never saw; statewide figures would be in-sample.", ""]
    lines += ["## Band records", "", "| Method | All | Cardiologists |", "|---|---|---|",
              f"| Splink | {pct(b['splink']['all'])} | {pct(b['splink']['focus'])} |",
              f"| Supervised | {pct(b['supervised']['all'])} | {pct(b['supervised']['focus'])} |"]
    if s["split"] != "dev":
        f = r["focus"]["overall"]
        lines += ["", "## Statewide, cardiologists' records", "",
                  f"Accuracy {pct(f['accuracy'])}, precision {pct(f['precision'])}, recall {pct(f['recall'])}; "
                  f"{errors(f):,} errors of {f['n']:,}."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["train", "apply"])
    parser.add_argument("--split", default="dev")
    parser.add_argument("--state", default="TX")
    parser.add_argument("--model", help="apply: the train run dir under runs/ (e.g. runs/<id>)")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    train(args) if args.command == "train" else apply(args)


if __name__ == "__main__":
    main()
