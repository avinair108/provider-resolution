"""Phase 4 run: send the uncertain band to the model, verify, then score Splink alone vs Splink + LLM.

  python -m adjudicate.run --subset tune --limit 50 --profile small     # iterate on the prompt
  python -m adjudicate.run --subset eval                                 # the reported number
  python -m adjudicate.run --subset eval --repeats 3                     # ranges over shuffled letter orders

Every response is cached, so a run stopped by the free tier's daily cap resumes where it left off when rerun.
Decisions are made before gold is loaded; thresholds come from the Phase 3 run, never re-tuned here.
"""

import argparse
import hashlib
import json
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import datetime
from pathlib import Path

import duckdb

from adjudicate import prompt, select, verify
from llm import LLM, BudgetExceeded, ProviderError, QuotaExhausted
from matching import evaluate
from matching.run import score

ROOT = Path(__file__).resolve().parent.parent


def _rows(con, table: str, ids: list[str]) -> dict[str, dict]:
    cur = con.execute(f"SELECT * FROM {table} WHERE record_id IN (SELECT unnest(?::VARCHAR[]))", [ids])
    cols = [d[0] for d in cur.description]
    return {r[0]: dict(zip(cols, r)) for r in cur.fetchall()}


def adjudicate(llm: LLM, item: dict, source_row: dict, candidate_rows: dict, sample: int, show_scores: bool,
               order: str = "shuffled", repair: bool = True, system: str = prompt.SYSTEM) -> dict:
    """One record. repair=False is the no-verifier ablation: the first reply stands unless it's unusable (bad JSON
    or a choice that isn't on the list); contradictions with the records are recorded but not acted on."""
    user, mapping = prompt.render(item, source_row, candidate_rows, sample=sample, show_scores=show_scores, order=order)
    out = {"record_id": item["record_id"], "sample": sample, "n_candidates": len(mapping), "attempts": 0,
           "cached": True, "input_tokens": 0, "output_tokens": 0}
    message, verdict = user, None
    for attempt in range(2 if repair else 1):  # the first answer, then at most one repair
        response = llm.complete(llm.request(user=message, system=system, json=True, sample=sample))
        out["attempts"] += 1
        out["cached"] &= response.cached
        out["input_tokens"] += response.input_tokens
        out["output_tokens"] += response.output_tokens
        out["model"] = response.model
        try:
            reply, error = response.parsed(), None
        except ValueError as e:  # json.JSONDecodeError is a ValueError
            reply, error = None, ("cut off at max_tokens" if response.finish_reason == "length" else str(e)[:120])
        verdict = verify.check(reply, mapping, source_row, candidate_rows, error)
        if attempt == 0:
            out["first_try_ok"] = verdict.ok
            out["first_errors"] = verdict.errors
            out["first_contradictions"] = verdict.contradictions
        if verdict.ok or not repair:
            break
        message = verify.repair_message(user, response.text, verdict)
    usable = verdict.ok if repair else not verdict.errors
    out.update(status="decided" if usable else "fallback", choice=verdict.choice, registry=verdict.registry,
               confidence=verdict.confidence, reason=verdict.reason, evidence=verdict.evidence,
               problems=verdict.problems(), letter_rank={k: [c["registry"] for c in item["candidates"]].index(v)
                                                         for k, v in mapping.items()})
    return out


def apply_decisions(con, base: str, name: str, decisions: list[dict]) -> None:
    """Copy the Splink table; for each decided record, its chosen registry pair gets p=1 and its other registry
    pairs p=0 ("none" sets them all to 0). Fallbacks and records never sent keep Splink's scores."""
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM {base}")
    con.execute("CREATE OR REPLACE TEMP TABLE dec (record_id VARCHAR, registry VARCHAR)")
    rows = [(d["record_id"], d["registry"]) for d in decisions if d["status"] == "decided"]
    if rows:
        con.executemany("INSERT INTO dec VALUES (?, ?)", rows)
    con.execute(f"""
        UPDATE {name} SET p = CASE WHEN d.registry IS NOT NULL AND (d.registry = {name}.a OR d.registry = {name}.b) THEN 1.0 ELSE 0.0 END
        FROM dec d
        WHERE (d.record_id = {name}.a OR d.record_id = {name}.b)
          AND (({name}.a LIKE 'np\\_%' ESCAPE '\\') <> ({name}.b LIKE 'np\\_%' ESCAPE '\\'))
    """)


def band_outcomes(con, band_ids: list[str], threshold: float) -> dict:
    """After record_resolution: accuracy on the band records themselves (labeled ones), all and cardiologists."""
    con.execute("CREATE OR REPLACE TEMP TABLE band_ids AS SELECT unnest(?::VARCHAR[]) AS record_id", [band_ids])
    rows = con.execute(f"""
        SELECT r.record_id, r.focus,
               (CASE WHEN r.p_top >= {threshold} THEN r.top_registry END) IS NOT DISTINCT FROM r.true_registry AS ok
        FROM record_top r JOIN band_ids USING (record_id)
    """).fetchall()
    return {rid: (focus, ok) for rid, focus, ok in rows}


def _acc(outcomes: dict, focus_only: bool) -> float | None:
    vals = [ok for focus, ok in outcomes.values() if focus or not focus_only]
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--state", default="TX")
    parser.add_argument("--baseline", help="Phase 3 run dir under runs/ (default: latest *_baseline)")
    parser.add_argument("--profile", default="default", help="profile in config/llm.toml")
    parser.add_argument("--band", nargs=2, type=float, default=[0.01, 0.99], metavar=("LOW", "HIGH"))
    parser.add_argument("--subset", default="eval", choices=["tune", "eval", "all"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--show-scores", action="store_true", help="ablation: show Splink scores to the model")
    parser.add_argument("--order", default="shuffled", choices=prompt.ORDERS, help="ablation: candidate order")
    parser.add_argument("--no-repair", action="store_true", help="ablation: no verifier-driven repair")
    parser.add_argument("--effort", help="override the profile's reasoning effort (low | medium | high)")
    parser.add_argument("--max-cost", type=float, help="dollar cap for this run's live calls (default: the profile's)")
    parser.add_argument("--workers", type=int, default=1, help="parallel requests; keep 1 on rate-limited free tiers")
    parser.add_argument("--final", action="store_true", help="the one run on the held-out split, with the frozen method")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    if args.split != "dev" and not args.final:
        raise SystemExit("Phase 4 tunes on dev only; score the held-out state once, with the method frozen, using --final.")
    if args.final and args.subset != "all":
        raise SystemExit("--final scores the whole held-out band: use --subset all")

    baseline = ROOT / "runs" / args.baseline if args.baseline else sorted((ROOT / "runs").glob(f"*_matching_{args.split}_{args.state}_baseline"))[-1]
    base_summary = json.loads((baseline / "summary.json").read_text())
    thresholds = base_summary["results"]["splink"][0]
    record_t, pair_t = thresholds["record_threshold"], thresholds["pair_threshold"]
    data = ROOT / "data" / args.split / args.state
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_adjudicate_{args.split}_{args.state}_{args.subset}" + (f"_{args.tag}" if args.tag else "")
    out = ROOT / "runs" / run_id
    out.mkdir(parents=True)

    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"CREATE TABLE splink AS SELECT * FROM read_parquet('{baseline / 'splink_0.parquet'}')")
    select.load_sources(con, data / "stripped")
    select.registry_candidates(con, "splink")
    items = select.band(con, *args.band, split=args.subset, limit=args.limit)
    ids = [i["record_id"] for i in items]
    sources = {**_rows(con, "src_open_payments", [i for i in ids if i.startswith("op_")]),
               **_rows(con, "src_part_d", [i for i in ids if not i.startswith("op_")])}
    registry = _rows(con, "src_nppes", sorted({c["registry"] for i in items for c in i["candidates"]}))
    print(f"Band [{args.band[0]}, {args.band[1]}) subset {args.subset}: {len(items):,} records. Baseline {baseline.name}.", flush=True)

    overrides = {**({"max_cost_usd": args.max_cost} if args.max_cost is not None else {}),
                 **({"effort": args.effort} if args.effort else {})}
    llm = LLM.from_config(args.profile, **overrides)
    prompt_name = tomllib.loads((ROOT / "config" / "llm.toml").read_text())[args.profile].get("prompt", "full")
    system = prompt.SYSTEMS[prompt_name]  # "compact" only for a model fine-tuned on it
    all_decisions, stop = {}, threading.Event()
    stopped = None
    t0 = time.time()
    done = [0]
    lock = threading.Lock()

    def run_one(item: dict, sample: int) -> dict:
        nonlocal stopped
        if stop.is_set():
            return {"record_id": item["record_id"], "sample": sample, "status": "not_run"}
        cands = {c["registry"]: registry[c["registry"]] for c in item["candidates"]}
        try:
            decision = adjudicate(llm, item, sources[item["record_id"]], cands, sample, args.show_scores,
                                  order=args.order, repair=not args.no_repair, system=system)
        except (QuotaExhausted, BudgetExceeded) as e:
            with lock:
                if not stop.is_set():
                    stopped = f"{type(e).__name__}: {e}"
                    print(f"Stopped in sample {sample}: {stopped}. Rerun the same command to resume from cache.", flush=True)
                stop.set()
            return {"record_id": item["record_id"], "sample": sample, "status": "not_run"}
        except ProviderError as e:  # one record's request failed for good: record it, keep going
            print(f"  {item['record_id'][:3]} record failed: {str(e)[:160]}", flush=True)
            decision = {"record_id": item["record_id"], "sample": sample, "status": "error", "problems": str(e)[:300]}
        with lock:
            done[0] += 1
            if done[0] % 100 == 0:
                s = llm.stats
                print(f"  {done[0]}/{len(items) * args.repeats}  live {s['live_calls']}  cached {s['cache_hits']}  "
                      f"${s['cost_usd']:.3f}  {time.time() - t0:.0f}s", flush=True)
        return decision

    for sample in range(args.repeats):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            decisions = list(pool.map(lambda item: run_one(item, sample), items))  # map keeps input order
        all_decisions[sample] = decisions
        with (out / f"decisions_{sample}.jsonl").open("w") as f:
            for d in decisions:
                f.write(json.dumps(d, default=str) + "\n")

    # Scoring: the first point at which gold is read.
    evaluate.load_gold(con, data / "gold")
    results = {"splink": score(con, "splink", pair_t, record_t)}
    splink_band = band_outcomes(con, ids, record_t)
    combined, band_results = [], []
    for sample, decisions in all_decisions.items():
        apply_decisions(con, "splink", "splink_llm", decisions)
        combined.append(score(con, "splink_llm", pair_t, record_t))
        band_results.append(band_outcomes(con, ids, record_t))
    results["splink_llm"] = combined
    gold_registry = dict(con.execute("SELECT record_id, true_registry FROM g_records").fetchall())
    focus_of = dict(con.execute("SELECT record_id, focus FROM g_records").fetchall())

    summary = {"run_id": run_id, "baseline": baseline.name, "split": args.split, "state": args.state, "subset": args.subset,
               "band": args.band, "profile": args.profile,
               "prompt": prompt_name, "prompt_sha": hashlib.sha256(system.encode()).hexdigest()[:12], "show_scores": args.show_scores, "order": args.order,
               "repair": not args.no_repair, "effort": llm.defaults.get("effort"), "repeats": args.repeats,
               "n_band": len(items), "stopped": stopped, "thresholds": {"record": record_t, "pair": pair_t},
               "llm": llm.summary(), "workers": args.workers, "wall_s": round(time.time() - t0, 1),
               "process": [process_metrics(d, gold_registry, focus_of) for d in all_decisions.values()],
               "band_accuracy": {"splink": {"all": _acc(splink_band, False), "focus": _acc(splink_band, True)},
                                 "splink_llm": [{"all": _acc(b, False), "focus": _acc(b, True)} for b in band_results]},
               "flips": [flips(splink_band, b) for b in band_results],
               "results": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "report.md").write_text(to_markdown(json.loads(json.dumps(summary, default=str))))
    print(f"Report: {out / 'report.md'}")


def process_metrics(decisions: list[dict], gold_registry: dict, focus_of: dict) -> dict:
    """How the agent behaved: validity, repairs, contradictions, letter bias, and calibration of its confidence."""
    from evals.scorers.linkage import expected_calibration_error

    run = [d for d in decisions if d["status"] != "not_run"]
    decided = [d for d in run if d["status"] == "decided"]
    labeled = [d for d in decided if d["record_id"] in gold_registry]
    correct = [d["registry"] == gold_registry[d["record_id"]] for d in labeled]  # None == None: rightly "none"
    cal = expected_calibration_error([d["confidence"] for d in labeled], correct) if labeled else {"ece": None}
    letters = Counter(d["choice"] for d in decided)
    ranks = Counter(d["letter_rank"].get(d["choice"]) for d in decided if d["choice"] not in (None, "none"))
    return {
        "sent": len(run), "not_run": len(decisions) - len(run), "decided": len(decided),
        "fallback": len(run) - len(decided),
        "first_try_ok": sum(bool(d.get("first_try_ok")) for d in run),
        "repaired": sum(1 for d in decided if d["attempts"] == 2),
        "first_try_format_errors": sum(bool(d.get("first_errors")) for d in run),
        "first_try_contradictions": sum(bool(d.get("first_contradictions")) for d in run),
        "chose_none": letters.get("none", 0), "letters": dict(sorted(letters.items())),
        "splink_rank_of_choice": {str(k): v for k, v in sorted(ranks.items())},
        "llm_accuracy_labeled": sum(correct) / len(correct) if correct else None,
        "llm_accuracy_focus": (lambda xs: sum(xs) / len(xs) if xs else None)(
            [ok for d, ok in zip(labeled, correct) if focus_of.get(d["record_id"])]),
        "ece": cal["ece"], "calibration_bins": cal.get("bins"),
    }


def flips(before: dict, after: dict) -> dict:
    keys = before.keys() & after.keys()
    out = {}
    for scope in ("all", "focus"):
        ks = [k for k in keys if scope == "all" or before[k][0]]
        out[scope] = {"fixed": sum(1 for k in ks if not before[k][1] and after[k][1]),
                      "broken": sum(1 for k in ks if before[k][1] and not after[k][1]),
                      "still_wrong": sum(1 for k in ks if not before[k][1] and not after[k][1])}
    return out


def _pct(x):
    return "–" if x is None else f"{x:.1%}"


def _rng(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "–"
    lo, hi = min(xs), max(xs)
    return _pct(lo) if hi - lo < 0.0005 else f"{_pct(lo)}–{_pct(hi)}"


def to_markdown(s: dict) -> str:
    sp, cb = s["results"]["splink"], s["results"]["splink_llm"]
    rec = lambda r, g="overall": r["records"]["focus"].get(g, {})
    llm = s["llm"]
    lines = [f"# Phase 4: LLM adjudication of the uncertain band ({s['split']} {s['state']}, subset {s['subset']})", "",
             f"Run `{s['run_id']}` over baseline `{s['baseline']}`. Model `{llm['model']}` ({llm['provider']}, profile "
             f"`{s['profile']}`, prompt `{s['prompt_sha']}`), {s['repeats']} repeat(s) with reshuffled candidate order. Band: top Splink score in "
             f"[{s['band'][0]}, {s['band'][1]}) and cardiology in the record's or a candidate's own fields; "
             f"{s['n_band']:,} records. Thresholds fixed from Phase 3 (record {s['thresholds']['record']}). Aggregates only.", ""]
    if s["stopped"]:
        lines += [f"**Partial run:** {s['stopped']}. Records not sent keep Splink's decision.", ""]
    lines += ["## Cardiologists' records, resolved to the registry", "",
              "| Method | Accuracy | Precision | Recall |", "|---|---|---|---|",
              f"| Splink | {_pct(rec(sp)['accuracy'])} | {_pct(rec(sp)['precision'])} | {_pct(rec(sp)['recall'])} |",
              f"| Splink + LLM | {_rng([rec(r)['accuracy'] for r in cb])} | {_rng([rec(r)['precision'] for r in cb])} | {_rng([rec(r)['recall'] for r in cb])} |",
              "", "## On the band records themselves (labeled ones)", "",
              "| Method | All | Cardiologists |", "|---|---|---|",
              f"| Splink | {_pct(s['band_accuracy']['splink']['all'])} | {_pct(s['band_accuracy']['splink']['focus'])} |",
              f"| Splink + LLM | {_rng([b['all'] for b in s['band_accuracy']['splink_llm']])} | {_rng([b['focus'] for b in s['band_accuracy']['splink_llm']])} |",
              "", "Changes against Splink (cardiologists, per repeat): " +
              ", ".join(f"fixed {f['focus']['fixed']}, broke {f['focus']['broken']}, still wrong {f['focus']['still_wrong']}" for f in s["flips"]),
              "", "## By slice (cardiologists)", "", "| Slice | Records | Splink | Splink + LLM |", "|---|---|---|---|"]
    for g in sorted(k for k in sp["records"]["focus"] if k.startswith("slice:")):
        lines.append(f"| {g[6:]} | {rec(sp, g).get('n', 0):,} | {_pct(rec(sp, g).get('accuracy'))} | {_rng([rec(r, g).get('accuracy') for r in cb])} |")
    lines += ["", "## Agent behaviour", "", "| Repeat | Sent | Decided | Fallback | First try OK | Repaired | Format errors | Contradictions | Chose none | LLM accuracy | ECE |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, p in enumerate(s["process"]):
        ece = "–" if p["ece"] is None else f"{p['ece']:.3f}"
        lines.append(f"| {i} | {p['sent']:,} | {p['decided']:,} | {p['fallback']:,} | {p['first_try_ok']:,} | {p['repaired']:,} | "
                     f"{p['first_try_format_errors']:,} | {p['first_try_contradictions']:,} | {p['chose_none']:,} | "
                     f"{_pct(p['llm_accuracy_labeled'])} | {ece} |")
    p0 = s["process"][0]
    lines += ["", f"Letters chosen (repeat 0): {p0['letters']}. Splink rank of the chosen candidate (0 = Splink's top): "
              f"{p0['splink_rank_of_choice']}.", "",
              f"Model calls: {llm['calls']:,} ({llm['live_calls']:,} live, {llm['cache_hits']:,} cached); tokens "
              f"{llm['input_tokens']:,} in / {llm['output_tokens']:,} out; cost ${llm['cost_usd']:.2f}; wall {s['wall_s']:.0f}s."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
