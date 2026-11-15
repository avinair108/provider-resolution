"""The agent's output contract and the gold item schema that scorers work from.

The agent answers in structure, not only prose, so scorers don't have to guess which number or provider in a
paragraph is the answer. The 10-K numeric scorer passed if *any* number in the text matched; here `value` holds only
the asserted answer, and a hedge across several values fails.
"""

from dataclasses import dataclass, field

Pair = tuple[str, str]


def canonical_pair(a: str, b: str) -> Pair:
    """Links are unordered: (a, b) and (b, a) are the same claim."""
    if a == b:
        raise ValueError(f"A record cannot link to itself: {a}")
    return (a, b) if a < b else (b, a)


@dataclass
class Claim:
    text: str
    cited_links: list[Pair] = field(default_factory=list)


@dataclass
class AgentOutput:
    answer_text: str = ""
    value: str | None = None  # numeric answers: only the asserted figure, e.g. "$12.3K"
    providers: list[str] = field(default_factory=list)  # record IDs for set answers
    abstained: bool = False
    conflict_records: list[str] = field(default_factory=list)  # records the answer reports as disagreeing
    evidence_records: list[str] = field(default_factory=list)  # records the answer says it relies on
    sources_queried: list[str] = field(default_factory=list)
    links: list[Pair] = field(default_factory=list)  # every cross-source link the answer relies on
    claims: list[Claim] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0
    latency_s: float = 0.0


@dataclass
class GoldItem:
    id: str
    question: str
    question_type: str  # single_source | cross_source | multi_hop | ambiguous | conflict | unanswerable | decoy
    answer_kind: str  # money | count | percent | set | conflict | abstain
    required_sources: list[str]
    gold_value: float | None = None
    gold_entities: list[str] = field(default_factory=list)  # set answers: entity IDs, never raw NPIs in reports
    gold_conflict_entity: str | None = None  # conflict answers: the entity whose records disagree
    required_links: list[Pair] = field(default_factory=list)
    slices: list[str] = field(default_factory=list)
    split: str = "dev"  # dev | heldout
    subject_entity: str | None = None  # the gold entity the question is about, if one
    disagreeing_records: list[str] = field(default_factory=list)  # conflict answers: records whose values differ
    notes: str = ""  # why the gold answer is what it is (e.g. "3 providers share this name")
