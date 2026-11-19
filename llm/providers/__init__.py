"""Provider adapters. Each has a `name` and `send(Request) -> Response`; nothing else in the repo talks to an API."""

from llm.providers.fake import FakeProvider
from llm.providers.fireworks import FireworksProvider
from llm.providers.groq import GroqProvider
from llm.providers.openrouter import OpenRouterProvider


def make(settings: dict):
    """Build the adapter a profile in config/llm.toml names. Keys come from .env, never from the config."""
    from llm.env import require

    name = settings["provider"]
    if name == "groq":
        return GroqProvider(require("GROQ_API_KEY"))
    if name == "openrouter":
        return OpenRouterProvider(require("OPENROUTER_API_KEY"), host=settings["host"],
                                  quantization=settings.get("quantization"), omit=settings.get("omit"))
    if name == "fireworks":
        return FireworksProvider(require("FIREWORKS_API_KEY"))
    raise ValueError(f"unknown provider {name!r}; add an adapter in llm/providers/")


__all__ = ["FakeProvider", "FireworksProvider", "GroqProvider", "OpenRouterProvider", "make"]
