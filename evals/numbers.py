"""Number extraction with units and display precision. Parser carried over from the 10-K project."""

import re
from dataclasses import dataclass

_SCALE = {
    "trillion": 1e12, "tn": 1e12, "t": 1e12,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "million": 1e6, "mn": 1e6, "mm": 1e6, "m": 1e6,
    "thousand": 1e3, "k": 1e3,
}

_NUM_RE = re.compile(
    r"(?P<neg>[-−])?\$?\s?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s?(?P<unit>%|percent\b|trillion\b|billion\b|million\b|thousand\b|tn\b|bn\b|mn\b|mm\b|[tbmk]\b)?",
    re.IGNORECASE,
)


@dataclass
class Number:
    value: float
    kind: str  # "percent" | "scaled" | "plain"
    text: str
    precision: float = 0.0  # half a unit of the last displayed digit, in the value's scale ("$0.1M" -> 50,000)


def extract_numbers(text: str) -> list[Number]:
    found = []
    for m in _NUM_RE.finditer(text):
        num = m.group("num").replace(",", "")
        value = float(num)
        decimals = len(num.split(".")[1]) if "." in num else 0
        precision = 0.5 * 10**-decimals
        if m.group("neg"):
            value = -value
        unit = (m.group("unit") or "").lower()
        if unit in ("%", "percent"):
            kind = "percent"
        elif unit in _SCALE:
            value *= _SCALE[unit]
            precision *= _SCALE[unit]
            kind = "scaled"
        else:
            kind = "plain"
        found.append(Number(value, kind, m.group(0).strip(), precision))
    return found
