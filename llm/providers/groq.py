"""Groq, through its OpenAI-compatible REST endpoint, using the standard library only.

Rate limits: on 429 or 5xx it waits for `retry-after` (or backs off) and retries. If the wait would be longer than
`max_wait_s`, which is what the free tier's daily cap looks like, it raises QuotaExhausted rather than sleeping
for hours. The latest x-ratelimit-* headers are kept on `last_limits` so runs can report remaining quota.

JSON mode: Groq, like OpenAI, expects the word "JSON" somewhere in the prompt when response_format is set.
"""

import json
import time
import urllib.error
import urllib.request

from llm.types import ProviderError, QuotaExhausted, Request, Response

BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider:
    name = "groq"

    def __init__(self, api_key: str, timeout_s: float = 60, max_retries: int = 6, max_wait_s: float = 90,
                 sleep=time.sleep):
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.max_wait_s = max_wait_s
        self.sleep = sleep
        self.last_limits: dict[str, str] = {}

    def _http(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict[str, str], bytes]:
        request = urllib.request.Request(
            BASE_URL + path, method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                     "User-Agent": "provider-resolution/0.1"},  # the default urllib agent can be refused
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as r:
                return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
        except urllib.error.HTTPError as e:
            return e.code, {k.lower(): v for k, v in (e.headers or {}).items()}, e.read()

    @staticmethod
    def payload(request: Request) -> dict:
        messages = ([{"role": "system", "content": request.system}] if request.system else []) + \
                   [{"role": "user", "content": request.user}]
        body = {"model": request.model, "messages": messages, "max_tokens": request.max_tokens,
                "temperature": request.temperature, "seed": request.sample}
        if request.json:
            body["response_format"] = {"type": "json_object"}
        if request.effort:
            body["reasoning_effort"] = request.effort
        return body

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
                choice, usage = data["choices"][0], data.get("usage", {})
                return Response(text=choice["message"].get("content") or "", provider=self.name,
                                model=data.get("model", request.model), input_tokens=usage.get("prompt_tokens", 0),
                                output_tokens=usage.get("completion_tokens", 0), latency_s=round(latency, 3),
                                finish_reason=choice.get("finish_reason"),
                                meta={**{k: data[k] for k in ("id", "system_fingerprint") if data.get(k)},
                                      "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")})
            if status == 429 or status >= 500:
                wait = _retry_after(headers) or min(2 ** attempt, self.max_wait_s)
                if wait > self.max_wait_s:
                    raise QuotaExhausted(f"Groq limit resets in {wait:.0f}s ({_snippet(raw)})")
                if attempt == self.max_retries:
                    break
                self.sleep(wait)
                continue
            if status == 400 and (failed := _failed_json(raw)) is not None:
                # JSON mode rejected the model's own output: that's a bad answer, not an API failure. Hand it back
                # so the caller's parser fails and its repair step runs.
                return Response(text=failed, provider=self.name, model=request.model, input_tokens=0, output_tokens=0,
                                latency_s=round(latency, 3), finish_reason="json_validate_failed")
            raise ProviderError(f"Groq returned {status}: {_snippet(raw)}")
        raise ProviderError(f"Groq still failing after {self.max_retries + 1} attempts")

    def _pace(self, prompt_chars: int) -> None:
        """Wait out the per-minute token window if the last response says it can't fit this prompt.

        The free tier allows a few thousand tokens a minute, so waiting here is cheaper than a refused request."""
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
            raise ProviderError(f"Groq returned {status}: {_snippet(raw)}")
        return sorted(m["id"] for m in json.loads(raw)["data"])


def parse_duration(text: str) -> float | None:
    """Groq reset headers look like "1.522s", "1m26.4s" or "250ms"."""
    import re

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
    """Error text for messages and reports: short, with the account's organization id removed."""
    import re

    text = raw[:400].decode(errors="replace").replace("\n", " ")
    return re.sub(r"org_[A-Za-z0-9]+", "org_<redacted>", text)[:300]
