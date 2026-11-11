"""Model calls behind one interface: `LLM.complete(Request) -> Response`, with provider adapters in llm.providers."""

from llm.client import LLM
from llm.types import BudgetExceeded, ProviderError, QuotaExhausted, Request, Response

__all__ = ["LLM", "BudgetExceeded", "ProviderError", "QuotaExhausted", "Request", "Response"]
