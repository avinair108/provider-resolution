"""Deterministic checks on a model reply: the shape, then each factual claim against the records themselves.

Format errors: not a JSON object, a choice that isn't a shown letter or "none", confidence outside [0, 1], an unknown
field or relation. Contradictions: the reply says a field agrees (same or compatible) when the records differ, or that it
differs when the records are identical. Only fields with an unambiguous normalized value are checked (names, middle initial, ZIP, city);
address, credential and specialty are judgment calls and pass through.
"""

import re
from difflib import SequenceMatcher
from dataclasses import dataclass, field

from pipeline.perturb import NICKNAMES

FIELDS = {"last_name", "first_name", "middle_initial", "credential", "address", "city", "zip", "specialty"}
RELATIONS = {"same", "compatible", "different", "missing"}
FORMAL = {nick: formal for formal, nicks in NICKNAMES.items() for nick in nicks}


@dataclass
class Verdict:
    choice: str | None = None  # letter, "none", or None when unusable
    registry: str | None = None
    confidence: float | None = None
    reason: str = ""
    evidence: list = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and not self.contradictions

    def problems(self) -> str:
        return "; ".join(self.errors + self.contradictions)


def _letters(value) -> str:
    return re.sub(r"[^A-Z]", "", str(value or "").upper())


def _zip5(value) -> str:
    return str(value or "")[:5]


def _similar(a: str, b: str) -> bool:
    """Spelling variants (MUHAMMAD/MOHAMMED, typos): close enough that calling them compatible isn't false."""
    return a[:1] == b[:1] and SequenceMatcher(None, a, b).ratio() >= 0.7


LOCATION_FIELDS = {"zip", "city"}  # "compatible" here can mean nearby, which the records can't refute


def actual_relations(source: dict, cand: dict) -> dict[str, str]:
    """The relation each checkable field really has: same, compatible, different or missing."""
    rel = {}
    s_last, c_last, c_other = _letters(source.get("last_name")), _letters(cand.get("last_name")), _letters(cand.get("other_last_name"))
    if not s_last or not c_last:
        rel["last_name"] = "missing"
    elif s_last == c_last:
        rel["last_name"] = "same"
    elif s_last == c_other or s_last in c_last or c_last in s_last or _similar(s_last, c_last):
        rel["last_name"] = "compatible"
    else:
        rel["last_name"] = "different"
    s_first, c_first = _letters(source.get("first_name")), _letters(cand.get("first_name"))
    if not s_first or not c_first:
        rel["first_name"] = "missing"
    elif s_first == c_first:
        rel["first_name"] = "same"
    elif FORMAL.get(s_first, s_first) == FORMAL.get(c_first, c_first) or \
            (s_first[0] == c_first[0] and min(len(s_first), len(c_first)) == 1) or _similar(s_first, c_first) \
            or s_first.startswith(c_first) or c_first.startswith(s_first):
        rel["first_name"] = "compatible"
    else:
        rel["first_name"] = "different"
    s_mid = _letters(source.get("middle_name", source.get("middle_initial")))[:1]
    c_mid = _letters(cand.get("middle_name"))[:1]
    rel["middle_initial"] = "missing" if not s_mid or not c_mid else ("same" if s_mid == c_mid else "different")
    s_zip = _zip5(source.get("zip"))
    c_zips = {z for z in (_zip5(cand.get("practice_zip")), _zip5(cand.get("mailing_zip"))) if z}
    rel["zip"] = "missing" if not s_zip or not c_zips else ("same" if s_zip in c_zips else "different")
    s_city = _letters(source.get("city"))
    c_cities = {c for c in (_letters(cand.get("practice_city")), _letters(cand.get("mailing_city"))) if c}
    rel["city"] = "missing" if not s_city or not c_cities else ("same" if s_city in c_cities else "different")
    return rel


def _contradicts(claimed: str, actual: str) -> bool:
    """Only a claim of agreement where the fields differ, or of a difference where they're identical. Same versus
    compatible, or compatible versus missing, are wording, not facts worth a repair."""
    if claimed in ("same", "compatible"):
        return actual == "different"
    if claimed == "different":
        return actual == "same"
    return False


def check(reply: dict | None, mapping: dict[str, str], source: dict, candidates: dict[str, dict],
          parse_error: str | None = None) -> Verdict:
    v = Verdict()
    if reply is None:
        v.errors.append(f"reply is not a JSON object ({parse_error})" if parse_error else "reply is not a JSON object")
        return v
    choice = str(reply.get("choice", "")).strip()
    choice = "none" if choice.lower() == "none" else choice.upper()
    if choice != "none" and choice not in mapping:
        v.errors.append(f"choice {choice!r} is not one of {sorted(mapping)} or 'none'")
    else:
        v.choice = choice
        v.registry = mapping.get(choice)
    try:
        v.confidence = float(reply.get("confidence"))
        if not 0 <= v.confidence <= 1:
            raise ValueError
    except (TypeError, ValueError):
        v.errors.append("confidence must be a number from 0 to 1")
        v.confidence = None
    v.reason = str(reply.get("reason", ""))[:500]
    evidence = reply.get("evidence") or []
    if not isinstance(evidence, list):
        v.errors.append("evidence must be a list")
        evidence = []
    for e in evidence:
        if not isinstance(e, dict) or e.get("field") not in FIELDS or e.get("relation") not in RELATIONS:
            v.errors.append(f"bad evidence item {e!r}: field must be one of {sorted(FIELDS)}, relation one of {sorted(RELATIONS)}")
            continue
        v.evidence.append({"field": e["field"], "relation": e["relation"]})
    if v.registry and not v.errors:
        actual = actual_relations(source, candidates[v.registry])
        for e in v.evidence:
            truth = actual.get(e["field"])
            if e["field"] in LOCATION_FIELDS and e["relation"] == "compatible":
                continue
            if truth is not None and _contradicts(e["relation"], truth):
                v.contradictions.append(f"{e['field']} is {truth}, not {e['relation']}, between the source and candidate {choice}")
    return v


def repair_message(original_user: str, previous_reply: str, verdict: Verdict) -> str:
    return (f"{original_user}\n\nYour previous reply was:\n{previous_reply.strip()[:1500]}\n\n"
            f"It was rejected by an automatic check: {verdict.problems()}. "
            "Look at the records again and reply with a corrected JSON object only.")
