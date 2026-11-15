"""Generate the question set for one state, with gold answers from the hidden-NPI truth.

  python -m qa.questions                       # Texas dev, the reported set -> data/dev/TX/questions/questions.jsonl
  python -m qa.questions --set tune            # a separate set for iterating on the agent -> questions_tune.jsonl
  python -m qa.questions --split heldout --state CA

The tune set uses another seed and leaves out every provider (and question) in the reported set, so prompt changes
are judged on questions the final number never sees.

Gold answers join records by their true provider (gold membership), never through the graph, so a linkage mistake
in the graph shows up as a wrong answer. Questions name real providers, so the output stays in data/ (git-ignored);
reports carry counts only.

Types (targets for 60 questions):
  single_source  9  one source; the answer may span several of the provider's records in it (name/address variants)
  cross_source  10  identified in one source, answered from another: the link is the point
  multi_hop      9  a chain: top prescriber of a drug -> what one payer paid them; or a ZIP + threshold -> a set
  ambiguous      8  a name several providers share, nothing else: the right move is to decline
  conflict       8  the sources disagree on a provider's city: surface the disagreeing records
  unanswerable   8  invented provider, a year or dataset that isn't loaded: abstain
  decoy          8  a name plus a detail that rules out a busier namesake
"""

import argparse
import json
import random
import tomllib
from dataclasses import asdict
from pathlib import Path

import duckdb

from evals.contract import GoldItem, canonical_pair

ROOT = Path(__file__).resolve().parent.parent
TARGETS = {"single_source": 9, "cross_source": 10, "multi_hop": 9, "ambiguous": 8, "conflict": 8, "unanswerable": 8, "decoy": 8}
HARD_SLICES = ("first_name_differs", "last_name_matches_other_name", "address_differs_zip", "address_differs_street")
YEAR = 2024
STATE_NAMES = {"TX": "Texas", "CA": "California"}


def _name(first, last, middle=None) -> str:
    parts = [p for p in (first, (middle[:1] + ".") if middle else None, last) if p]
    return " ".join(p.title() if len(p) > 2 else p for p in parts)


def _sql_list(values) -> str:
    return "[" + ", ".join("'" + v.replace("'", "''") + "'" for v in values) + "]"


def load(con, data: Path, scope: dict) -> None:
    s, g = data / "stripped", data / "gold"
    con.execute(f"CREATE OR REPLACE TABLE m AS SELECT * FROM read_parquet('{g / 'membership.parquet'}')")
    con.execute(f"CREATE OR REPLACE TABLE focus AS SELECT * FROM read_parquet('{g / 'focus_entities.parquet'}')")
    con.execute(f"CREATE OR REPLACE TABLE sl AS SELECT * FROM read_parquet('{g / 'slices.parquet'}')")
    con.execute(f"CREATE OR REPLACE TABLE pay AS SELECT * FROM read_parquet('{s / 'open_payments_payments.parquet'}')")
    con.execute(f"CREATE OR REPLACE TABLE rx AS SELECT * FROM read_parquet('{s / 'part_d_drugs.parquet'}')")
    taxonomy = _sql_list(scope["nppes"]["taxonomy_codes"])
    # Every record (labeled or not) with its identifying fields; key = true provider, or the record itself if unlabeled.
    con.execute(f"""
        CREATE OR REPLACE TABLE recs AS
        WITH r AS (
            SELECT record_id, 'nppes' AS source, upper(last_name) AS last, upper(first_name) AS first, middle_name AS middle,
                   credential, practice_address_1 AS address, upper(practice_city) AS city, left(practice_zip, 5) AS zip,
                   list_has_any(taxonomy_codes, {taxonomy}) AS registry_cardiology, NULL AS prescriber_type
            FROM read_parquet('{s / 'nppes.parquet'}')
            UNION ALL
            SELECT record_id, 'open_payments', upper(last_name), upper(first_name), middle_name, NULL, address_1, upper(city),
                   left(zip, 5), false, NULL
            FROM read_parquet('{s / 'open_payments_recipients.parquet'}')
            UNION ALL
            SELECT record_id, 'part_d', upper(last_name), upper(first_name), middle_initial, credential, address_1, upper(city),
                   left(zip, 5), false, prescriber_type
            FROM read_parquet('{s / 'part_d_prescribers.parquet'}')
        )
        SELECT r.*, m.entity_id, coalesce(m.entity_id, r.record_id) AS who, m.entity_id IN (SELECT entity_id FROM focus) AS is_focus
        FROM r LEFT JOIN m USING (record_id)
        WHERE r.last IS NOT NULL AND r.first IS NOT NULL
    """)
    con.execute("CREATE OR REPLACE TABLE name_n AS SELECT first, last, count(DISTINCT who) AS n FROM recs GROUP BY ALL")
    con.execute("CREATE OR REPLACE TABLE name_city_n AS SELECT first, last, city, count(DISTINCT who) AS n FROM recs GROUP BY ALL")
    con.execute("""CREATE OR REPLACE TABLE ent_op AS
                   SELECT m.entity_id, sum(p.amount_usd) AS paid, count(*) AS n_pay, count(DISTINCT p.record_id) AS n_records
                   FROM pay p JOIN m USING (record_id) GROUP BY 1""")
    con.execute("CREATE OR REPLACE TABLE ent_payer AS SELECT m.entity_id, p.payer, sum(p.amount_usd) AS usd FROM pay p JOIN m USING (record_id) GROUP BY ALL")
    con.execute(f"""CREATE OR REPLACE TABLE ent_pd AS
                    SELECT m.entity_id, sum(d.total_claims) AS claims, sum(d.total_drug_cost) AS cost, count(DISTINCT d.record_id) AS n_records
                    FROM read_parquet('{s / 'part_d_prescribers.parquet'}') d JOIN m USING (record_id) GROUP BY 1""")
    con.execute("""CREATE OR REPLACE TABLE ent_drug AS
                   SELECT m.entity_id, coalesce(r.generic_name, r.brand_name) AS drug, sum(r.total_claims) AS claims
                   FROM rx r JOIN m USING (record_id) GROUP BY ALL""")


def _rows(con, sql: str, params=()) -> list[dict]:
    cur = con.execute(sql, list(params))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _slices(con, entity: str) -> list[str]:
    return [s for (s,) in con.execute("SELECT DISTINCT slice FROM sl JOIN m USING (record_id) WHERE m.entity_id = ? ORDER BY 1", [entity]).fetchall()]


def _links(con, record: str, entity: str, sources: list[str]) -> list:
    others = [r for (r,) in con.execute("SELECT record_id FROM m WHERE entity_id = ? AND source IN (SELECT unnest(?::VARCHAR[])) AND record_id <> ?",
                                        [entity, sources, record]).fetchall()]
    return [canonical_pair(record, o) for o in sorted(others)]


def _pick(rng, pool: list, k: int, key=lambda x: x) -> list:
    pool = sorted(pool, key=key)
    return rng.sample(pool, min(k, len(pool)))


def single_source(con, rng, state: str, k: int) -> list[GoldItem]:
    """Unique name + city; the answer totals all of the provider's records in that source."""
    items = []
    op = _rows(con, """
        SELECT r.record_id, r.first, r.last, r.city, r.entity_id, e.paid, e.n_records
        FROM recs r JOIN ent_op e USING (entity_id) JOIN name_city_n n USING (first, last, city)
        WHERE r.source = 'open_payments' AND r.is_focus AND n.n = 1 AND e.paid >= 100
        QUALIFY row_number() OVER (PARTITION BY r.entity_id ORDER BY r.record_id) = 1""")
    for x in _pick(rng, op, k - k // 2, key=lambda x: x["record_id"]):
        items.append(GoldItem(id="", question=f"How much did Open Payments report in general payments to {_name(x['first'], x['last'])} "
                              f"of {x['city'].title()}, {STATE_NAMES[state]}, in program year {YEAR}?",
                              question_type="single_source", answer_kind="money", required_sources=["open_payments"],
                              gold_value=round(x["paid"], 2), subject_entity=x["entity_id"], slices=_slices(con, x["entity_id"]),
                              notes=f"{x['n_records']} Open Payments recipient record(s) for this provider"))
    pd = _rows(con, """
        SELECT r.record_id, r.first, r.last, r.city, r.entity_id, e.claims
        FROM recs r JOIN ent_pd e USING (entity_id) JOIN name_city_n n USING (first, last, city)
        WHERE r.source = 'part_d' AND r.is_focus AND n.n = 1
        QUALIFY row_number() OVER (PARTITION BY r.entity_id ORDER BY r.record_id) = 1""")
    for x in _pick(rng, pd, k // 2, key=lambda x: x["record_id"]):
        items.append(GoldItem(id="", question=f"How many Medicare Part D claims did {_name(x['first'], x['last'])} of "
                              f"{x['city'].title()}, {STATE_NAMES[state]}, have in {YEAR}?",
                              question_type="single_source", answer_kind="count", required_sources=["part_d"],
                              gold_value=float(x["claims"]), subject_entity=x["entity_id"], slices=_slices(con, x["entity_id"])))
    return items


def cross_source(con, rng, state: str, k: int) -> list[GoldItem]:
    """Identified by one source's record (its own spelling and address), answered from the other source."""
    items = []
    base = """
        SELECT r.record_id, r.first, r.last, r.middle, r.city, r.address, r.prescriber_type, r.entity_id, op.paid, pd.claims,
               (SELECT count(*) FROM sl s JOIN m mm USING (record_id) WHERE mm.entity_id = r.entity_id
                AND s.slice IN (SELECT unnest(?::VARCHAR[]))) > 0 AS hard
        FROM recs r JOIN ent_op op USING (entity_id) JOIN ent_pd pd USING (entity_id) JOIN name_city_n n USING (first, last, city)
        WHERE r.source = ? AND r.is_focus AND n.n = 1 AND op.paid >= 100
        QUALIFY row_number() OVER (PARTITION BY r.entity_id ORDER BY r.record_id) = 1"""
    for source, n in (("open_payments", k - k // 2), ("part_d", k // 2)):
        pool = _rows(con, base, [list(HARD_SLICES), source])
        hard, easy = [x for x in pool if x["hard"]], [x for x in pool if not x["hard"]]
        chosen = _pick(rng, hard, (n + 1) // 2, key=lambda x: x["record_id"])
        chosen += _pick(rng, easy, n - len(chosen), key=lambda x: x["record_id"])
        for x in chosen:
            who = _name(x["first"], x["last"], x["middle"])
            if source == "open_payments":
                q = (f"The Open Payments recipient {who} at {x['address'].title()}, {x['city'].title()}, {STATE_NAMES[state]}: "
                     f"how many Medicare Part D claims did this provider have in {YEAR}?")
                kind, value, other = "count", float(x["claims"]), ["part_d"]
            else:
                q = (f"The Medicare Part D prescriber {who} of {x['city'].title()}, {STATE_NAMES[state]} ({x['prescriber_type']}): "
                     f"how much did this provider receive in Open Payments general payments in {YEAR}?")
                kind, value, other = "money", round(x["paid"], 2), ["open_payments"]
            items.append(GoldItem(id="", question=q, question_type="cross_source", answer_kind=kind,
                                  required_sources=[source] + other, gold_value=value, subject_entity=x["entity_id"],
                                  required_links=_links(con, x["record_id"], x["entity_id"], other),
                                  slices=_slices(con, x["entity_id"])))
    return items


def multi_hop(con, rng, state: str, k: int) -> list[GoldItem]:
    items = []
    # Drug -> its top prescriber (by a clear margin, a cardiologist) -> what that provider's top payer paid them.
    drugs = _rows(con, """
        WITH ranked AS (SELECT drug, entity_id, claims, row_number() OVER (PARTITION BY drug ORDER BY claims DESC, entity_id) AS rk FROM ent_drug),
        top2 AS (SELECT drug, max(claims) FILTER (WHERE rk = 1) AS c1, max(claims) FILTER (WHERE rk = 2) AS c2,
                        any_value(entity_id) FILTER (WHERE rk = 1) AS entity_id FROM ranked WHERE rk <= 2 GROUP BY drug)
        SELECT t.drug, t.entity_id, p.payer, p.usd
        FROM top2 t JOIN focus USING (entity_id)
        JOIN (SELECT *, row_number() OVER (PARTITION BY entity_id ORDER BY usd DESC, payer) AS rp FROM ent_payer) p
          ON p.entity_id = t.entity_id AND p.rp = 1
        WHERE t.c1 >= 50 AND (t.c2 IS NULL OR t.c1 >= 1.2 * t.c2) AND p.usd >= 100""")
    for x in _pick(rng, drugs, k - k * 4 // 9, key=lambda x: x["drug"]):
        items.append(GoldItem(id="", question=f"How much did {x['payer']} pay, in Open Payments general payments for {YEAR}, to the "
                              f"{STATE_NAMES[state]} provider with the most Medicare Part D claims for {x['drug'].lower()}?",
                              question_type="multi_hop", answer_kind="money", required_sources=["part_d", "open_payments"],
                              gold_value=round(x["usd"], 2), subject_entity=x["entity_id"], slices=_slices(con, x["entity_id"]),
                              notes="top prescriber leads the next by at least 20%"))
    # ZIP + threshold -> the set of registry cardiologists there paid more than the threshold (no one within 10% of it).
    zips = _rows(con, """
        WITH c AS (SELECT r.zip, r.entity_id, coalesce(op.paid, 0) AS paid FROM recs r LEFT JOIN ent_op op USING (entity_id)
                   WHERE r.source = 'nppes' AND r.registry_cardiology AND r.entity_id IS NOT NULL AND r.zip IS NOT NULL),
        t AS (SELECT unnest([1000, 5000, 10000, 25000, 50000]) AS threshold)
        SELECT zip, threshold, list(entity_id ORDER BY entity_id) FILTER (WHERE paid > threshold) AS above,
               count(*) FILTER (WHERE paid BETWEEN 0.9 * threshold AND 1.1 * threshold) AS near
        FROM c, t GROUP BY ALL""")
    zips = [z for z in zips if z["above"] and 2 <= len(z["above"]) <= 5 and z["near"] == 0]
    seen, chosen = set(), []
    for z in _pick(rng, zips, len(zips), key=lambda z: (z["zip"], z["threshold"])):
        if z["zip"] not in seen and len(chosen) < k * 4 // 9:
            seen.add(z["zip"])
            chosen.append(z)
    for z in chosen:
        items.append(GoldItem(id="", question=f"Which cardiologists whose NPPES registry practice address is in ZIP {z['zip']} received "
                              f"more than ${z['threshold']:,} in Open Payments general payments in {YEAR}? List each provider once.",
                              question_type="multi_hop", answer_kind="set", required_sources=["nppes", "open_payments"],
                              gold_entities=list(z["above"]), notes=f"{len(z['above'])} providers; none within 10% of the threshold"))
    return items


def ambiguous(con, rng, state: str, k: int) -> list[GoldItem]:
    names = _rows(con, """
        SELECT r.first, r.last, count(DISTINCT r.entity_id) AS n
        FROM recs r JOIN ent_op USING (entity_id)
        WHERE r.source = 'open_payments'
        GROUP BY ALL HAVING n >= 2 AND bool_or(r.is_focus)""")
    return [GoldItem(id="", question=f"How much did {_name(x['first'], x['last'])} receive in Open Payments general payments in {YEAR}?",
                     question_type="ambiguous", answer_kind="abstain", required_sources=["open_payments"],
                     notes=f"{x['n']} providers with Open Payments records share this name in {STATE_NAMES[state]}")
            for x in _pick(rng, names, k, key=lambda x: (x["last"], x["first"]))]


def conflict(con, rng, state: str, k: int) -> list[GoldItem]:
    """The registry and another source disagree on the city. The name is unique statewide, so the provider is clear."""
    pool = _rows(con, """
        WITH reg AS (SELECT entity_id, record_id, city FROM recs WHERE source = 'nppes' AND entity_id IS NOT NULL),
        other AS (SELECT r.entity_id, r.record_id, r.city FROM recs r JOIN reg g USING (entity_id)
                  WHERE r.source <> 'nppes' AND r.city IS NOT NULL AND regexp_replace(r.city, '[^A-Z]', '', 'g') <> regexp_replace(g.city, '[^A-Z]', '', 'g'))
        SELECT g.entity_id, g.record_id AS registry_record, list(o.record_id ORDER BY o.record_id) AS differing,
               any_value(r.first) AS first, any_value(r.last) AS last
        FROM reg g JOIN other o USING (entity_id) JOIN recs r ON r.record_id = g.record_id JOIN name_n n ON n.first = r.first AND n.last = r.last
        WHERE r.is_focus AND n.n = 1
        GROUP BY ALL""")
    return [GoldItem(id="", question=f"Do the NPPES registry, Open Payments and Medicare Part D agree on the city where "
                     f"{_name(x['first'], x['last'])} practises? If they disagree, identify the records that disagree.",
                     question_type="conflict", answer_kind="conflict", required_sources=["nppes", "open_payments", "part_d"],
                     gold_conflict_entity=x["entity_id"], subject_entity=x["entity_id"],
                     disagreeing_records=[x["registry_record"]] + list(x["differing"]), slices=_slices(con, x["entity_id"]))
            for x in _pick(rng, pool, k, key=lambda x: x["entity_id"])]


def unanswerable(con, rng, state: str, k: int) -> list[GoldItem]:
    items = []
    firsts = [f for (f,) in con.execute("SELECT first FROM recs WHERE is_focus GROUP BY 1 HAVING count(*) >= 50 ORDER BY 1").fetchall()]
    lasts = [l for (l,) in con.execute("SELECT last FROM recs WHERE is_focus GROUP BY 1 HAVING count(*) >= 20 ORDER BY 1").fetchall()]
    cities = [c for (c,) in con.execute("SELECT city FROM recs WHERE source = 'nppes' AND city IS NOT NULL GROUP BY 1 ORDER BY count(*) DESC LIMIT 10").fetchall()]
    existing = {(f, l) for f, l in con.execute("SELECT DISTINCT first, last FROM recs").fetchall()}
    n_fake = k - 2 * (k // 3)
    while sum(1 for i in items if "invented" in i.notes) < n_fake:
        f, l, c = rng.choice(firsts), rng.choice(lasts), rng.choice(cities)
        if (f, l) in existing:
            continue
        existing.add((f, l))
        items.append(GoldItem(id="", question=f"How much did {_name(f, l)} of {c.title()}, {STATE_NAMES[state]}, receive in Open Payments "
                              f"general payments in {YEAR}?", question_type="unanswerable", answer_kind="abstain",
                              required_sources=["open_payments"], notes="invented provider: no record in any source"))
    real = _rows(con, """SELECT r.first, r.last, r.city, r.entity_id FROM recs r JOIN ent_pd USING (entity_id) JOIN name_city_n n USING (first, last, city)
                         WHERE r.source = 'part_d' AND r.is_focus AND n.n = 1
                         QUALIFY row_number() OVER (PARTITION BY r.entity_id ORDER BY r.record_id) = 1""")
    picks = _pick(rng, real, 2 * (k // 3), key=lambda x: x["entity_id"])
    for i, x in enumerate(picks):
        who = f"{_name(x['first'], x['last'])} of {x['city'].title()}, {STATE_NAMES[state]}"
        if i < k // 3:
            q, note = f"How many Medicare Part D claims did {who} have in {YEAR - 3}?", f"only {YEAR} is loaded"
        else:
            q, note = f"How much did Medicare Part B pay {who} in {YEAR}?", "Medicare Part B isn't loaded"
        items.append(GoldItem(id="", question=q, question_type="unanswerable", answer_kind="abstain",
                              required_sources=[], subject_entity=x["entity_id"], notes=note))
    return items


def decoy(con, rng, state: str, k: int) -> list[GoldItem]:
    """Two providers share a name; the question's city picks the smaller one, and the namesake was paid at least 25% more."""
    pool = _rows(con, """
        WITH c AS (
            SELECT r.first, r.last, r.city, r.entity_id, r.is_focus, op.paid,
                   (SELECT n FROM name_city_n n WHERE n.first = r.first AND n.last = r.last AND n.city = r.city) AS n_city
            FROM recs r JOIN ent_op op USING (entity_id)
            WHERE r.source = 'open_payments'
            QUALIFY row_number() OVER (PARTITION BY r.entity_id ORDER BY r.record_id) = 1)
        SELECT t.first, t.last, t.city, t.entity_id, t.paid, d.entity_id AS decoy, d.city AS decoy_city, d.paid AS decoy_paid
        FROM c t JOIN c d ON d.first = t.first AND d.last = t.last AND d.entity_id <> t.entity_id
        WHERE t.is_focus AND t.n_city = 1 AND d.city <> t.city AND d.paid >= 1.25 * t.paid AND t.paid >= 100
        QUALIFY row_number() OVER (PARTITION BY t.entity_id ORDER BY d.paid DESC, d.entity_id) = 1""")
    return [GoldItem(id="", question=f"How much did {_name(x['first'], x['last'])} of {x['city'].title()}, {STATE_NAMES[state]}, "
                     f"receive in Open Payments general payments in {YEAR}?", question_type="decoy", answer_kind="money",
                     required_sources=["open_payments"], gold_value=round(x["paid"], 2), subject_entity=x["entity_id"],
                     slices=_slices(con, x["entity_id"]),
                     notes=f"a namesake in {x['decoy_city'].title()} received {x['decoy_paid'] / x['paid']:.1f}x as much")
            for x in _pick(rng, pool, k, key=lambda x: x["entity_id"])]


GENERATORS = {"single_source": single_source, "cross_source": cross_source, "multi_hop": multi_hop, "ambiguous": ambiguous,
              "conflict": conflict, "unanswerable": unanswerable, "decoy": decoy}


def _balanced(items: list[GoldItem], k: int) -> list[GoldItem]:
    """Take k items, alternating across answer kinds, so trimming an oversized pool keeps the mix."""
    groups: dict[str, list[GoldItem]] = {}
    for i in items:
        groups.setdefault(i.answer_kind, []).append(i)
    out, queues = [], list(groups.values())
    while len(out) < k and any(queues):
        for q in queues:
            if q and len(out) < k:
                out.append(q.pop(0))
    return out


def generate(con, state: str, split: str, seed: int, name: str = "eval", exclude: list[GoldItem] = ()) -> list[GoldItem]:
    """`exclude`: items of another set whose providers and questions this set must not reuse."""
    banned_entities = {e for x in exclude for e in [x.subject_entity, x.gold_conflict_entity, *x.gold_entities] if e}
    banned_questions = {x.question for x in exclude}
    items = []
    for qtype, k in TARGETS.items():
        rng = random.Random(f"{seed}:{state}:{qtype}" + ("" if name == "eval" else f":{name}"))
        made = [i for i in GENERATORS[qtype](con, rng, state, k + (k if exclude else 0))
                if i.question not in banned_questions and not ({i.subject_entity, i.gold_conflict_entity, *i.gold_entities} & banned_entities)]
        made = _balanced(made, k)
        prefix = state if name == "eval" else f"{state}-{name}"
        for i, item in enumerate(made, 1):
            item.id, item.split = f"{prefix}-{qtype}-{i:02d}", split
        items += made
    return items


def report(items: list[GoldItem], state: str, split: str) -> str:
    from collections import Counter

    by_type = Counter(i.question_type for i in items)
    by_kind = Counter(i.answer_kind for i in items)
    slices = Counter(s for i in items for s in i.slices)
    lines = [f"# Question set ({split} {state})", "", f"{len(items)} questions. Counts only: the questions name real providers "
             "and stay in data/ (git-ignored).", "", "| Type | Questions | Target |", "|---|---|---|"]
    lines += [f"| {t} | {by_type.get(t, 0)} | {k} |" for t, k in TARGETS.items()]
    lines += ["", "| Answer kind | Questions |", "|---|---|"] + [f"| {k} | {n} |" for k, n in sorted(by_kind.items())]
    lines += ["", "Hard-case slices touched by question subjects:", "", "| Slice | Questions |", "|---|---|"]
    lines += [f"| {s} | {n} |" for s, n in sorted(slices.items())]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--state", default="TX")
    parser.add_argument("--set", default="eval", choices=["eval", "tune"])
    args = parser.parse_args()
    scope = tomllib.loads((ROOT / "config" / "scope.toml").read_text())
    data = ROOT / "data" / args.split / args.state
    con = duckdb.connect()
    load(con, data, scope)
    out = data / "questions"
    out.mkdir(exist_ok=True)
    exclude = []
    if args.set != "eval":
        from qa.run import load_questions
        exclude = load_questions(out / "questions.jsonl")
    items = generate(con, args.state, args.split, scope["seed"], args.set, exclude)
    path = out / ("questions.jsonl" if args.set == "eval" else f"questions_{args.set}.jsonl")
    with path.open("w") as f:
        for item in items:
            f.write(json.dumps(asdict(item)) + "\n")
    text = report(items, args.state, args.split)
    if args.set == "eval":
        (ROOT / "reports" / f"questions_{args.split}_{args.state}.md").write_text(text)
    print(text)
    short = {t: TARGETS[t] - sum(1 for i in items if i.question_type == t) for t in TARGETS}
    if any(short.values()):
        print("Short of target:", {t: n for t, n in short.items() if n})


if __name__ == "__main__":
    main()
