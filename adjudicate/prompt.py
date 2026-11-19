"""Render one source record and its registry candidates as text. No record ids, no Splink scores by default.

Candidates get letters in an order shuffled per record and sample, so a model that favours "A" can't ride the
Splink ranking. `show_scores` is an ablation switch.
"""

import hashlib
import random
import tomllib
from functools import cache
from pathlib import Path

from pipeline.reference import nucc_names

ROOT = Path(__file__).resolve().parent.parent
LETTERS = "ABCDEFGH"
CARDIOLOGY_CODES = set(tomllib.loads((ROOT / "config" / "scope.toml").read_text())["nppes"]["taxonomy_codes"])

SYSTEM = """You match healthcare provider records across US government datasets.

You get one SOURCE record, from CMS Open Payments or Medicare Part D, and up to six CANDIDATE records from the NPPES \
provider registry. Decide which candidate is the same individual person as the source, or "none" if no candidate is.

Usually one candidate is the right person. Answer "none" only when no candidate is plausibly the same person, or when two candidates are equally plausible and nothing separates them.

Normal for the same person: nicknames (BOB/ROBERT), a first initial only, typos, a missing middle name, ZIP+4 versus 5-digit ZIP, and a payer listing a broader specialty (Internal Medicine for a cardiologist).
Registry addresses are often out of date: residents and fellows (taxonomy "Student in an Organized Health Care Education/Training Program") move after training, and doctors change practices. A full-name match with a consistent specialty is strong evidence even at a different address, unless another candidate shares that name.
A different surname with the same first name, the same specialty and the same practice address usually means a name change (marriage); the registry may also list the former name as "Other name on file".
Points to different people: a different middle initial when both have one, a different credential type (MD vs NP), a clearly different specialty, or a different city with nothing else connecting them. Several people can share a name in one city, and many doctors share one hospital address, so a name alone or an address alone isn't enough.

Reply with a JSON object only:
{"choice": "<candidate letter or none>",
 "confidence": <number 0 to 1: the probability your answer is right>,
 "evidence": [{"field": "<last_name|first_name|middle_initial|credential|address|city|zip|specialty>",
               "relation": "<same|compatible|different|missing>"}],
 "reason": "<one short sentence>"}
Evidence compares the source with your chosen candidate; leave it empty for "none"."""

# For a model fine-tuned on the task: the instructions above are learned from examples, so it gets the output
# contract only. Saves ~550 tokens per call, at training and at inference.
SYSTEM_COMPACT = """Match the SOURCE provider record to the CANDIDATE registry record that is the same individual, or "none". \
Reply with a JSON object only: {"choice": "<letter or none>", "confidence": <0 to 1>, \
"evidence": [{"field": ..., "relation": "same|compatible|different|missing"}], "reason": "<short>"}"""

SYSTEMS = {"full": SYSTEM, "compact": SYSTEM_COMPACT}


@cache
def _taxonomy_names() -> dict[str, str]:
    return nucc_names()


def _join(*parts) -> str:
    return " ".join(str(p).strip() for p in parts if p not in (None, "") and str(p).strip())


def _address(a1, a2, city, state, zip_) -> str:
    return ", ".join(p for p in [_join(a1), _join(a2), _join(city), _join(state, zip_)] if p) or "(none)"


def source_card(source: str, row: dict) -> str:
    label = {"open_payments": "Open Payments recipient", "part_d": "Medicare Part D prescriber"}[source]
    middle = row.get("middle_name", row.get("middle_initial"))
    lines = [f"SOURCE ({label})",
             f"  Name: {_join(row['last_name'] + ',' if row.get('last_name') else '', row.get('first_name'), middle, row.get('suffix'))}"]
    if row.get("credential"):
        lines.append(f"  Credential: {row['credential']}")
    lines.append(f"  Address: {_address(row.get('address_1'), row.get('address_2'), row.get('city'), row.get('state'), row.get('zip'))}")
    if source == "open_payments":
        specialties = [s.split("|")[-1].strip() for s in (row.get("specialties") or [])]
        lines.append(f"  Specialty: {'; '.join(specialties) or '(none)'}")
    else:
        lines.append(f"  Specialty: {row.get('prescriber_type') or '(none)'}")
    return "\n".join(lines)


def candidate_card(letter: str, row: dict, p: float | None = None) -> str:
    lines = [f"CANDIDATE {letter}" + (f" (similarity score {p:.2f})" if p is not None else ""),
             f"  Name: {_join(row['last_name'] + ',' if row.get('last_name') else '', row.get('first_name'), row.get('middle_name'), row.get('suffix'))}"]
    if row.get("credential"):
        lines.append(f"  Credential: {row['credential']}")
    other = _join(row.get("other_last_name") and row["other_last_name"] + ",", row.get("other_first_name"), row.get("other_middle_name"))
    if other:
        lines.append(f"  Other name on file: {other}")
    lines.append(f"  Practice address: {_address(row.get('practice_address_1'), row.get('practice_address_2'), row.get('practice_city'), row.get('practice_state'), row.get('practice_zip'))}")
    lines.append(f"  Mailing address: {_address(row.get('mailing_address_1'), row.get('mailing_address_2'), row.get('mailing_city'), row.get('mailing_state'), row.get('mailing_zip'))}")
    codes = [c for c in (row.get("taxonomy_codes") or []) if c]
    if codes:
        names = _taxonomy_names()
        shown = []
        for c in dict.fromkeys(codes):  # dedupe, keep order
            label = names.get(c, c)
            shown.append(f"{label} (cardiology)" if c in CARDIOLOGY_CODES else label)
        lines.append(f"  Taxonomy: {'; '.join(shown)}")
    return "\n".join(lines)


ORDERS = ("shuffled", "splink", "reversed")


def letter_order(record_id: str, n: int, sample: int, mode: str = "shuffled") -> list[int]:
    """Which candidate (by Splink rank) each letter shows. "shuffled" (the method) is seeded by record and sample, so
    stable and cacheable; "splink" (A = Splink's top) and "reversed" (Splink's top last) are position-bias ablations."""
    order = list(range(n))
    if mode == "shuffled":
        random.Random(int(hashlib.md5(f"{record_id}:{sample}".encode()).hexdigest(), 16)).shuffle(order)
    elif mode == "reversed":
        order.reverse()
    elif mode != "splink":
        raise ValueError(f"order must be one of {ORDERS}")
    return order


def render(item: dict, source_row: dict, candidate_rows: dict[str, dict], sample: int = 0,
           show_scores: bool = False, order: str = "shuffled") -> tuple[str, dict[str, str]]:
    """User message and the letter -> registry record mapping."""
    source = "open_payments" if item["record_id"].startswith("op_") else "part_d"
    cands = item["candidates"]
    mapping, cards = {}, []
    for letter, idx in zip(LETTERS, letter_order(item["record_id"], len(cands), sample, order)):
        cand = cands[idx]
        mapping[letter] = cand["registry"]
        cards.append(candidate_card(letter, candidate_rows[cand["registry"]], cand["p"] if show_scores else None))
    return source_card(source, source_row) + "\n\n" + "\n\n".join(cards), mapping
