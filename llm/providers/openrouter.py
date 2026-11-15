"""OpenRouter: one key, many hosts. Each profile pins one host and precision, with fallbacks off.

The same open model can be served by different hosts at different precisions (fp4, fp8, bf16), which changes the
answers. So a profile names `host` and `quantization`, routing never falls back to another host, and only hosts that
support every parameter we send (JSON mode, seed, reasoning effort) are eligible; a profile's `omit` list drops
parameters its host doesn't take, such as temperature and seed on Anthropic's endpoint. The host is part of the provider
name, so it is part of every cache key: results from different hosts never mix.

Cost: OpenRouter reports the charged cost of each request, which the client records in place of its own estimate.
Running out of credit (HTTP 402) stops the run cleanly; a top-up and rerun resume from the cache.
"""

import json

from llm.providers.openai_compat import OpenAICompatible, _snippet
from llm.types import ProviderError, Request


class OpenRouterProvider(OpenAICompatible):
    label = "OpenRouter"
    base_url = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, host: str, quantization: str | None = None, omit: list[str] | None = None,
                 **kwargs):
        super().__init__(api_key, **kwargs)
        self.host = host
        self.quantization = quantization
        self.omit = set(omit or [])  # parameters the pinned host rejects (Anthropic: temperature, seed)
        self.name = f"openrouter:{host}" + (f"/{quantization}" if quantization else "")

    def headers(self) -> dict[str, str]:
        return {**super().headers(), "X-Title": "provider-resolution"}

    def payload(self, request: Request) -> dict:
        body = super().payload(request)
        effort = body.pop("reasoning_effort", None)
        if effort:
            body["reasoning"] = {"effort": effort, "exclude": True}  # reasoning still runs; we just don't need its text
        body["provider"] = {"order": [self.host], "allow_fallbacks": False, "require_parameters": True,
                            **({"quantizations": [self.quantization]} if self.quantization else {})}
        body["usage"] = {"include": True}
        for key in self.omit:
            body.pop(key, None)
        return body

    def response_meta(self, data: dict, usage: dict) -> dict:
        meta = super().response_meta(data, usage)
        meta["served_by"] = data.get("provider")
        if usage.get("cost") is not None:
            meta["cost_usd"] = float(usage["cost"])
        return meta

    def out_of_quota(self, status: int, raw: bytes) -> str | None:
        return f"OpenRouter credit exhausted ({_snippet(raw)})" if status == 402 else None

    def endpoints(self, model: str) -> list[dict]:
        """Hosts serving a model, with price per 1M tokens, precision and supported parameters."""
        status, _, raw = self._http("GET", f"/models/{model}/endpoints")
        if status != 200:
            raise ProviderError(f"OpenRouter returned {status}: {_snippet(raw)}")
        out = []
        for e in json.loads(raw)["data"]["endpoints"]:
            pricing = e.get("pricing", {})
            out.append({"host": e.get("provider_name"), "tag": e.get("tag"), "quantization": e.get("quantization"),
                        "input_per_mtok": float(pricing.get("prompt", 0)) * 1e6,
                        "output_per_mtok": float(pricing.get("completion", 0)) * 1e6,
                        "parameters": e.get("supported_parameters", [])})
        return out

    def credit(self) -> dict | None:
        """Account balance (/credits) and this key's own spending cap (/key). The cap is not money in the account."""
        out = {}
        status, _, raw = self._http("GET", "/credits")
        if status == 200:
            data = json.loads(raw).get("data", {})
            total, used = data.get("total_credits"), data.get("total_usage")
            if total is not None and used is not None:
                out.update(account_credit=float(total), account_used=float(used), account_remaining=float(total) - float(used))
        status, _, raw = self._http("GET", "/key")
        if status == 200:
            data = json.loads(raw).get("data", {})
            out.update(key_limit=data.get("limit"), key_limit_remaining=data.get("limit_remaining"))
        return out or None
