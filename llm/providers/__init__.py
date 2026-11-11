"""Provider adapters. Each has a `name` and `send(Request) -> Response`; nothing else in the repo talks to an API."""

from llm.providers.fake import FakeProvider
from llm.providers.groq import GroqProvider


def make(name: str):
    from llm.env import require

    if name == "groq":
        return GroqProvider(require("GROQ_API_KEY"))
    raise ValueError(f"unknown provider {name!r}; add an adapter in llm/providers/")


__all__ = ["FakeProvider", "GroqProvider", "make"]
