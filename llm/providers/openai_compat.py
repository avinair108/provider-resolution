"""Shared adapter for OpenAI-compatible chat endpoints (Groq, OpenRouter, ...), standard library only.

Rate limits: on 429 or 5xx it waits for `retry-after` (or backs off) and retries. If the wait would be longer than
`max_wait_s`, which is what a daily cap looks like, it raises QuotaExhausted rather than sleeping for hours, so a
run stops cleanly and resumes from the cache. The latest x-ratelimit-* headers are kept on `last_limits`.

JSON mode: like OpenAI, these endpoints expect the word "JSON" somewhere in the prompt when response_format is set.
"""

import json
import re
import time
import urllib.error
import urllib.request

from llm.types import ProviderError, QuotaExhausted, Request, Response


class OpenAICompatible:
    name = "openai-compatible"
    label = "API"
    base_url = ""

    def __init__(self, api_key: str, timeout_s: float = 90, max_retries: int = 6, max_wait_s: float = 90,
                 sleep=time.sleep):
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.max_wait_s = max_wait_s
        self.sleep = sleep
        self.last_limits: dict[str, str] = {}

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                "User-Agent": "provider-resolution/0.1"}  # the default urllib agent can be refused

    def _http(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict[str, str], bytes]:
        request = urllib.request.Request(self.base_url + path, method=method, headers=self.headers(),
                                         data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as r:
                return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
        except urllib.error.HTTPError as e:
            return e.code, {k.lower(): v for k, v in (e.headers or {}).items()}, e.read()

    def payload(self, request: Request) -> dict:
        messages = ([{"role": "system", "content": request.system}] if request.system else []) + \
                   [{"role": "user", "content": request.user}]
        body = {"model": request.model, "messages": messages, "max_tokens": request.max_tokens,
                "temperature": request.temperature, "seed": request.sample}
        if request.json:
            body["response_format"] = {"type": "json_object"}
        if request.effort:
            body["reasoning_effort"] = request.effort
        return body

    def response_meta(self, data: dict, usage: dict) -> dict:
        return {**{k: data[k] for k in ("id", "system_fingerprint") if data.get(k)},
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")}

    def out_of_quota(self, status: int, raw: bytes) -> str | None:
        """A reason to stop the run (not retry), or None."""
        return None

    def send(self, request: Request) -> Response:
        body = self.payload(request)
        for attempt in range(self.max_retries + 1):
            self._pace(len(request.system) + len(request.user))
            start = time.monotonic()
            try:
                status, headers, raw = self._http("POST", "/chat/completions", body)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == self.max_retries:
                    raise ProviderError(f"network error after {attempt + 1} attempts: {e}") from e
                self.sleep(min(2 ** attempt, self.max_wait_s))
                continue
            latency = time.monotonic() - start
            self.last_limits = {k: v for k, v in headers.items() if k.startswith("x-ratelimit")}
            if status == 200:
                data = json.loads(raw)
                if "error" in data and not data.get("choices"):  # some routers report upstream errors with 200
                    if attempt == self.max_retries:
                        raise ProviderError(f"{self.label} upstream error: {_snippet(raw)}")
                    self.sleep(min(2 ** attempt, self.max_wait_s))
                    continue
                choice, usage = data["choices"][0], data.get("usage") or {}
                return Response(text=choice["message"].get("content") or "", provider=self.name,
                                model=data.get("model", request.model), input_tokens=usage.get("prompt_tokens", 0),
                                output_tokens=usage.get("completion_tokens", 0), latency_s=round(latency, 3),
                                finish_reason=choice.get("finish_reason"), meta=self.response_meta(data, usage))
            if (reason := self.out_of_quota(status, raw)) is not None:
                raise QuotaExhausted(reason)
            if status == 429 or status >= 500:
                wait = _retry_after(headers) or min(2 ** attempt, self.max_wait_s)
                if wait > self.max_wait_s:
                    raise QuotaExhausted(f"{self.label} limit resets in {wait:.0f}s ({_snippet(raw)})")
                if attempt == self.max_retries:
                    break
                self.sleep(wait)
                continue
            if status == 400 and (failed := _failed_json(raw)) is not None:
                # JSON mode rejected the model's own output: that's a bad answer, not an API failure. Hand it back
                # so the caller's parser fails and its repair step runs.
                return Response(text=failed, provider=self.name, model=request.model, input_tokens=0, output_tokens=0,
                                latency_s=round(latency, 3), finish_reason="json_validate_failed")
            raise ProviderError(f"{self.label} returned {status}: {_snippet(raw)}")
        raise ProviderError(f"{self.label} still failing after {self.max_retries + 1} attempts")

    def _pace(self, prompt_chars: int) -> None:
        """Wait out a per-minute token window if the last response says it can't fit this prompt."""
        remaining = self.last_limits.get("x-ratelimit-remaining-tokens")
        reset = parse_duration(self.last_limits.get("x-ratelimit-reset-tokens", ""))
        if remaining is None or reset is None:
            return
        needed = prompt_chars / 3 + 400  # rough: about 3 characters a token, plus the reply and reasoning
        if float(remaining) < needed:
            self.sleep(min(reset + 0.25, self.max_wait_s))
            self.last_limits = {k: v for k, v in self.last_limits.items() if k != "x-ratelimit-remaining-tokens"}

    def list_models(self) -> list[str]:
        status, _, raw = self._http("GET", "/models")
        if status != 200:
            raise ProviderError(f"{self.label} returned {status}: {_snippet(raw)}")
        return sorted(m["id"] for m in json.loads(raw)["data"])


def parse_duration(text: str) -> float | None:
    """Reset headers look like "1.522s", "1m26.4s" or "250ms"."""
    parts = re.findall(r"([0-9.]+)(ms|h|m|s)", text or "")
    if not parts:
        return None
    scale = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}
    return sum(float(n) * scale[u] for n, u in parts)


def _failed_json(raw: bytes) -> str | None:
    try:
        error = json.loads(raw).get("error", {})
    except (ValueError, AttributeError):
        return None
    return error.get("failed_generation", "") if error.get("code") == "json_validate_failed" else None


def _retry_after(headers: dict[str, str]) -> float | None:
    try:
        return float(headers["retry-after"])
    except (KeyError, ValueError):
        return None


def _snippet(raw: bytes) -> str:
    """Error text for messages and reports: short, with account and organization ids removed."""
    text = raw[:400].decode(errors="replace").replace("\n", " ")
    text = re.sub(r"org_[A-Za-z0-9]+", "org_<redacted>", text)
    text = re.sub(r"user_[A-Za-z0-9]+", "user_<redacted>", text)
    return text[:300]
