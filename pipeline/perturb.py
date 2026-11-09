"""Synthetic perturbations of identity records. Each function returns a changed copy, or None if it doesn't apply."""

import random
import string

# Formal first name -> common nicknames. Names only; applied in both directions.
NICKNAMES = {
    "ALEXANDER": ["ALEX"], "ANDREW": ["ANDY", "DREW"], "ANTHONY": ["TONY"], "BENJAMIN": ["BEN"], "CHARLES": ["CHUCK", "CHARLIE"],
    "CHRISTOPHER": ["CHRIS"], "DANIEL": ["DAN", "DANNY"], "DAVID": ["DAVE"], "DONALD": ["DON"], "EDWARD": ["ED", "TED"],
    "ELIZABETH": ["LIZ", "BETH", "BETSY"], "GREGORY": ["GREG"], "JAMES": ["JIM", "JIMMY"], "JEFFREY": ["JEFF"],
    "JENNIFER": ["JEN", "JENNY"], "JOHN": ["JACK"], "JONATHAN": ["JON"], "JOSEPH": ["JOE"], "JOSHUA": ["JOSH"],
    "KATHERINE": ["KATE", "KATIE"], "KENNETH": ["KEN"], "MATTHEW": ["MATT"], "MICHAEL": ["MIKE"], "NICHOLAS": ["NICK"],
    "PATRICIA": ["PAT", "TRISH"], "PATRICK": ["PAT"], "PETER": ["PETE"], "RAYMOND": ["RAY"], "REBECCA": ["BECKY"],
    "RICHARD": ["RICK", "DICK", "RICH"], "ROBERT": ["BOB", "ROB", "BOBBY"], "RONALD": ["RON"], "SAMUEL": ["SAM"],
    "STEPHEN": ["STEVE"], "STEVEN": ["STEVE"], "SUSAN": ["SUE"], "THOMAS": ["TOM"], "TIMOTHY": ["TIM"], "VINCENT": ["VINCE"],
    "WILLIAM": ["BILL", "WILL"],
}
_FORMAL_OF = {}
for _formal, _nicks in NICKNAMES.items():
    for _nick in _nicks:
        _FORMAL_OF.setdefault(_nick, []).append(_formal)

_ADDRESS_WORDS = {
    "STREET": "ST", "ST": "STREET", "AVENUE": "AVE", "AVE": "AVENUE", "BOULEVARD": "BLVD", "BLVD": "BOULEVARD",
    "DRIVE": "DR", "DR": "DRIVE", "ROAD": "RD", "RD": "ROAD", "SUITE": "STE", "STE": "SUITE", "HIGHWAY": "HWY",
    "HWY": "HIGHWAY", "PARKWAY": "PKWY", "PKWY": "PARKWAY", "LANE": "LN", "LN": "LANE", "NORTH": "N", "SOUTH": "S",
    "EAST": "E", "WEST": "W",
}


def typo(record: dict, rng: random.Random) -> dict | None:
    fields = [f for f in ("last_name", "first_name") if len(record.get(f) or "") >= 4]
    if not fields:
        return None
    field = rng.choice(fields)
    name = record[field]
    i = rng.randrange(1, len(name) - 1)  # keep the first letter: typos there are rarer and are a separate problem
    kind = rng.choice(["delete", "transpose", "substitute", "double"])
    if kind == "delete":
        changed = name[:i] + name[i + 1:]
    elif kind == "transpose":
        changed = name[:i] + name[i + 1] + name[i] + name[i + 2:]
    elif kind == "substitute":
        alphabet = string.ascii_uppercase if name.isupper() else string.ascii_lowercase
        changed = name[:i] + rng.choice([c for c in alphabet if c != name[i].upper() and c != name[i].lower()]) + name[i + 1:]
    else:
        changed = name[:i] + name[i] + name[i:]
    return {**record, field: changed} if changed != name else None


def nickname(record: dict, rng: random.Random) -> dict | None:
    first = (record.get("first_name") or "").upper()
    options = NICKNAMES.get(first) or _FORMAL_OF.get(first)
    return {**record, "first_name": rng.choice(options)} if options else None


def initial_only(record: dict, rng: random.Random) -> dict | None:
    first = record.get("first_name") or ""
    if len(first) < 2:
        return None
    middle = record.get("middle_name") or ""
    return {**record, "first_name": first[0], "middle_name": middle[:1] or None}


def swapped_name(record: dict, rng: random.Random) -> dict | None:
    first, last = record.get("first_name"), record.get("last_name")
    if not first or not last or first.upper() == last.upper():
        return None
    return {**record, "first_name": last, "last_name": first}


def address_abbreviation(record: dict, rng: random.Random) -> dict | None:
    changed = dict(record)
    any_change = False
    for field in ("address_1", "address_2"):
        value = record.get(field)
        if not value:
            continue
        words = [_ADDRESS_WORDS.get(w.upper().rstrip("."), w) for w in value.split()]
        joined = " ".join(words)
        if joined.upper() != value.upper():
            changed[field] = joined
            any_change = True
    return changed if any_change else None


def old_address(record: dict, rng: random.Random, donors: list[dict]) -> dict | None:
    """Replace the address with a different provider's in the same state. Records can't be aged honestly without
    history, so this simulates "the source still has where they used to work"."""
    usable = [d for d in donors if d.get("zip") and d.get("zip") != record.get("zip") and d.get("address_1")]
    if not usable:
        return None
    donor = rng.choice(usable)
    return {**record, **{k: donor.get(k) for k in ("address_1", "address_2", "city", "state", "zip")}}


PERTURBATIONS = {
    "typo": typo,
    "nickname": nickname,
    "initial_only": initial_only,
    "swapped_name": swapped_name,
    "address_abbreviation": address_abbreviation,
}
ALL_TYPES = [*PERTURBATIONS, "old_address"]


def generate(records: list[dict], donors: list[dict], per_type: int, seed: int) -> list[dict]:
    """Up to `per_type` perturbed records per type, each `{"type", "origin", "record"}`. Deterministic for a seed."""
    out = []
    for kind in ALL_TYPES:
        rng = random.Random(f"{seed}:{kind}")
        candidates = list(records)
        rng.shuffle(candidates)
        made = 0
        for record in candidates:
            if made == per_type:
                break
            if kind == "old_address":
                sample = rng.sample(donors, min(len(donors), 25))
                changed = old_address(record, rng, [d for d in sample if d["record_id"] != record["record_id"]])
            else:
                changed = PERTURBATIONS[kind](record, rng)
            if changed is None or changed == record:
                continue
            out.append({"type": kind, "origin": record["record_id"], "record": changed})
            made += 1
    return out
