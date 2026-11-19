"""Build the fine-tuning set: Texas records like the adjudicator's band, labeled from the NPIs, as chat JSONL.

  python -m finetune.prepare --count                 # pool sizes and token estimate, writes nothing
  python -m finetune.prepare --n 1000                # data/finetune/train.jsonl and val.jsonl

Pool: Texas records whose best registry candidate scores in [0.001, 0.999) with cardiology on either side (the
band's rule, widened), minus every record of the reported Texas evaluation band. California is never read. So the
fine-tuned model is scored on exactly the records the other methods are, and none of them is in its training data.

Each example is the adjudicator's own rendering (shuffled letters, no scores) with the compact system prompt. The
target answer is the gold choice (a letter, or "none" when the right person isn't among the candidates), with
evidence taken from the verifier's own field comparison, so no target ever contradicts the records. Confidence is
a constant: the training labels carry no uncertainty, so the fine-tuned model's confidence isn't calibrated.

Writes text derived from real records; data/ is git-ignored and nothing here goes into reports beyond counts.
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import duckdb

from adjudicate import prompt, select, verify
from adjudicate.run import _rows
from matching import evaluate

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "finetune"
POOL_BAND = (0.001, 0.999)
EVAL_BAND = (0.01, 0.99)
TRAIN_SAMPLE = 1000  # letter shuffles distinct from the evaluation runs' samples 0-2
CONFIDENCE = 0.9
EVIDENCE_FIELDS = ("last_name", "first_name", "middle_initial", "city", "zip")


def target(choice: str, source_row: dict, cand_row: dict | None) -> str:
    evidence = []
    if cand_row is not None:
        rel = verify.actual_relations(source_row, cand_row)
        evidence = [{"field": f, "relation": rel[f]} for f in EVIDENCE_FIELDS if f in rel]
    return json.dumps({"choice": choice, "confidence": CONFIDENCE, "evidence": evidence, "reason": ""},
                      separators=(",", ":"))


def _baseline() -> Path:
    return sorted((ROOT / "runs").glob("*_matching_dev_TX_baseline"))[-1]


def build(n: int | None, val_frac: float, seed: int, count_only: bool) -> dict:
    data = ROOT / "data" / "dev" / "TX"
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"CREATE TABLE splink AS SELECT * FROM read_parquet('{_baseline() / 'splink_0.parquet'}')")
    select.load_sources(con, data / "stripped")
    select.registry_candidates(con, "splink")
    excluded = {i["record_id"] for i in select.band(con, *EVAL_BAND, split="eval")}
    pool = [i for i in select.band(con, *POOL_BAND) if i["record_id"] not in excluded]

    evaluate.load_gold(con, data / "gold")  # labels, read after the pool is fixed
    truth = dict(con.execute("SELECT record_id, true_registry FROM g_records").fetchall())
    pool = [i for i in pool if i["record_id"] in truth]
    for i in pool:
        regs = [c["registry"] for c in i["candidates"]]
        i["label"] = truth[i["record_id"]] if truth[i["record_id"]] in regs else None
        i["top_right"] = i["label"] == regs[0] and i["p_top"] >= 0.5

    rng = random.Random(seed)
    rng.shuffle(pool)
    chosen = pool if n is None else pool[:n]
    stats = {"excluded_eval_band": len(excluded), "pool_labeled": len(pool), "chosen": len(chosen),
             "label_none": sum(i["label"] is None for i in chosen),
             "splink_top_already_right": sum(i["top_right"] for i in chosen),
             "candidates": dict(Counter(len(i["candidates"]) for i in chosen))}

    ids = [i["record_id"] for i in chosen]
    sources = {**_rows(con, "src_open_payments", [r for r in ids if r.startswith("op_")]),
               **_rows(con, "src_part_d", [r for r in ids if not r.startswith("op_")])}
    registry = _rows(con, "src_nppes", sorted({c["registry"] for i in chosen for c in i["candidates"]}))
    rows, chars = [], 0
    for i in chosen:
        src = sources[i["record_id"]]
        cands = {c["registry"]: registry[c["registry"]] for c in i["candidates"]}
        user, mapping = prompt.render(i, src, cands, sample=TRAIN_SAMPLE)
        letter = next((k for k, v in mapping.items() if v == i["label"]), "none")
        answer = target(letter, src, cands.get(i["label"]))
        rows.append({"messages": [{"role": "system", "content": prompt.SYSTEM_COMPACT},
                                  {"role": "user", "content": user},
                                  {"role": "assistant", "content": answer}]})
        chars += len(prompt.SYSTEM_COMPACT) + len(user) + len(answer)
    stats["est_tokens_per_example"] = round(chars / max(1, len(rows)) / 3.5)  # ~3.5 chars/token on this text
    stats["est_tokens_total"] = round(chars / 3.5)
    if count_only:
        return stats

    OUT.mkdir(parents=True, exist_ok=True)
    held = lambda r: int(hashlib.md5(f"ft-val:{r}".encode()).hexdigest(), 16) % 100 < val_frac * 100
    with (OUT / "train.jsonl").open("w") as tr, (OUT / "val.jsonl").open("w") as va:
        for i, row in zip(chosen, rows):
            (va if held(i["record_id"]) else tr).write(json.dumps(row) + "\n")
    stats["system_sha"] = hashlib.sha256(prompt.SYSTEM_COMPACT.encode()).hexdigest()[:12]
    stats["seed"] = seed
    (OUT / "meta.json").write_text(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, help="examples to keep (default: the whole pool)")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", action="store_true", help="report pool size and token estimate only")
    args = parser.parse_args()
    print(json.dumps(build(args.n, args.val_frac, args.seed, args.count), indent=2))


if __name__ == "__main__":
    main()
