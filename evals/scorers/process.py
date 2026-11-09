"""Routing and claim-level citation scorers."""

from collections.abc import Iterable

from evals.contract import Claim, Pair, canonical_pair


def score_routing(queried: Iterable[str], required: Iterable[str]) -> dict:
    queried, required = set(queried), set(required)
    hit = queried & required
    return {
        "precision": len(hit) / len(queried) if queried else 0.0,
        "recall": len(hit) / len(required) if required else 1.0,
        "missed": sorted(required - queried),
        "extra": sorted(queried - required),
    }


def score_citations(claims: Iterable[Claim], required_links: Iterable[Pair], true_pairs: set[Pair],
                    unlabeled: frozenset[str] = frozenset()) -> dict:
    """Citation precision: of the links cited across claims, how many gold supports.
    Citation recall: of the links the question requires, how many some claim cites.

    A cited link touching a record with no known gold is left out of precision instead of counted as wrong. It is
    reported in `unscored`, since an answer resting on unlabeled links can't be vouched for either way."""
    cited = {canonical_pair(*link) for claim in claims for link in claim.cited_links}
    required = {canonical_pair(*link) for link in required_links}
    unscored = {p for p in cited if p[0] in unlabeled or p[1] in unlabeled}
    scored = cited - unscored
    supported = scored & true_pairs
    uncited_claims = sum(1 for claim in claims if not claim.cited_links)
    return {
        "precision": len(supported) / len(scored) if scored else None,
        "recall": len(cited & required) / len(required) if required else None,
        "unsupported": sorted(scored - supported),
        "unscored": sorted(unscored),
        "uncited_claims": uncited_claims,
    }
