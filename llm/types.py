"""Provider-neutral request and response shapes. Every provider adapter speaks only these."""

import json
import re
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Request:
    model: str
    user: str
    system: str = ""
    max_tokens: int = 512
    temperature: float = 0.0
    json: bool = False  # ask for a JSON object; the caller still validates it
    effort: str | None = None  # reasoning effort for reasoning models (low | medium | high); None = provider default
    sample: int = 0  # repeat index: part of the cache key, so repeats are separate samples, not cache hits

    def key_payload(self, provider: str) -> dict:
        return {"provider": provider, **asdict(self)}


@dataclass
class Response:
    text: str
    provider: str
    model: str  # the model the provider says served the request
    input_tokens: int
    output_tokens: int
    latency_s: float
    finish_reason: str | None = None
    cached: bool = False
    meta: dict = field(default_factory=dict)  # e.g. system_fingerprint, to spot silent model changes

    def parsed(self) -> dict:
        """The response as a JSON object, tolerating a ```json fence. Raises ValueError if it isn't one."""
        text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", self.text.strip())
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value


class ProviderError(RuntimeError):
    """A request the provider rejected, or that failed after every retry."""


class QuotaExhausted(ProviderError):
    """The provider's limit resets too far ahead to wait for (e.g. the free tier's daily cap)."""


class BudgetExceeded(RuntimeError):
    """This run hit its own cap on live (uncached) calls."""
