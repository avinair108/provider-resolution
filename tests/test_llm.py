"""The model-call layer, offline: cache, budget, leak gate, env parsing and the Groq adapter's retries."""

import json
import os

import pytest

from llm.cache import Cache, request_key
from llm.client import LLM
from llm.env import load_env
from llm.providers import FakeProvider, GroqProvider
from llm.providers.groq import parse_duration
from llm.types import BudgetExceeded, ProviderError, QuotaExhausted, Request, Response
from pipeline.leak import LeakError, is_valid_npi


def make_llm(tmp_path, **kw):
    provider = FakeProvider()
    return LLM(provider, model="m", cache=Cache(tmp_path / "cache"), **kw), provider


def test_cache_key_is_stable_and_covers_every_field():
    base = Request(model="m", user="u", system="s")
    assert request_key(base, "groq") == request_key(Request(model="m", user="u", system="s"), "groq")
    variants = [Request(model="m2", user="u", system="s"), Request(model="m", user="u2", system="s"),
                Request(model="m", user="u", system="s", sample=1), Request(model="m", user="u", system="s", json=True),
                Request(model="m", user="u", system="s", temperature=0.5)]
    keys = {request_key(v, "groq") for v in variants} | {request_key(base, "other")}
    assert len(keys) == len(variants) + 1 and request_key(base, "groq") not in keys


def test_second_identical_call_is_served_from_cache(tmp_path):
    llm, provider = make_llm(tmp_path)
    first = llm.complete(llm.request("hello"))
    second = llm.complete(llm.request("hello"))
    assert len(provider.calls) == 1 and not first.cached and second.cached and second.text == first.text
    assert llm.stats["cache_hits"] == 1 and llm.stats["live_calls"] == 1
    # A new LLM over the same cache directory (a rerun) also hits.
    rerun, provider2 = make_llm(tmp_path)
    assert rerun.complete(rerun.request("hello")).cached and not provider2.calls


def test_repeats_are_separate_samples(tmp_path):
    llm, provider = make_llm(tmp_path)
    for sample in range(3):
        llm.complete(llm.request("same prompt", sample=sample))
    assert len(provider.calls) == 3


def test_budget_counts_live_calls_only(tmp_path):
    llm, _ = make_llm(tmp_path, max_live_calls=1)
    llm.complete(llm.request("a"))
    llm.complete(llm.request("a"))  # cached: free
    with pytest.raises(BudgetExceeded):
        llm.complete(llm.request("b"))


def test_prompt_with_npi_like_value_is_never_sent(tmp_path):
    assert is_valid_npi("1999999901")
    llm, provider = make_llm(tmp_path)
    with pytest.raises(LeakError):
        llm.complete(llm.request("record: SMITH 1999999901"))
    with pytest.raises(LeakError):
        llm.complete(llm.request("fine", system="also checked: 1999999901"))
    assert not provider.calls and llm.stats["calls"] == 0


def test_parsed_tolerates_a_fence_and_rejects_non_objects():
    r = Response(text='```json\n{"match": true}\n```', provider="p", model="m", input_tokens=0, output_tokens=0, latency_s=0)
    assert r.parsed() == {"match": True}
    with pytest.raises(ValueError):
        Response(text="[1]", provider="p", model="m", input_tokens=0, output_tokens=0, latency_s=0).parsed()


def test_cost_from_prices(tmp_path):
    llm, _ = make_llm(tmp_path, prices=(1.0, 2.0))
    r = llm.complete(llm.request("x" * 4000))
    assert llm.summary()["cost_usd"] == pytest.approx((r.input_tokens + 2 * r.output_tokens) / 1e6)


def test_env_file_parsing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nGROQ_API_KEY="abc"\nexport OTHER=x\nEMPTY=\nALREADY=file\n')
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OTHER", raising=False)
    monkeypatch.setenv("ALREADY", "shell")
    values = load_env(env)
    assert values == {"GROQ_API_KEY": "abc", "OTHER": "x", "ALREADY": "file"}
    assert os.environ["GROQ_API_KEY"] == "abc" and os.environ["ALREADY"] == "shell"  # the shell wins


OK_BODY = json.dumps({"model": "served-model", "choices": [{"message": {"content": '{"a": 1}'}, "finish_reason": "stop"}],
                      "usage": {"prompt_tokens": 10, "completion_tokens": 3}, "system_fingerprint": "fp_1"}).encode()


def groq_with(responses):
    waits = []
    provider = GroqProvider("test-key", sleep=waits.append)
    sent = []

    def fake_http(method, path, body=None):
        sent.append(body)
        return responses.pop(0)

    provider._http = fake_http
    return provider, sent, waits


def test_groq_payload_and_response():
    provider, sent, _ = groq_with([(200, {"x-ratelimit-remaining-requests": "999"}, OK_BODY)])
    r = provider.send(Request(model="m", user="u", system="s", json=True, sample=2))
    assert sent[0]["messages"] == [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert sent[0]["response_format"] == {"type": "json_object"} and sent[0]["seed"] == 2
    assert (r.model, r.input_tokens, r.output_tokens, r.meta["system_fingerprint"]) == ("served-model", 10, 3, "fp_1")
    assert provider.last_limits == {"x-ratelimit-remaining-requests": "999"}


def test_groq_waits_on_rate_limit_then_succeeds():
    provider, _, waits = groq_with([(429, {"retry-after": "2"}, b"slow down"), (503, {}, b""), (200, {}, OK_BODY)])
    assert provider.send(Request(model="m", user="u")).text == '{"a": 1}'
    assert waits == [2.0, 2]


def test_groq_long_reset_raises_quota_exhausted():
    provider, _, waits = groq_with([(429, {"retry-after": "3600"}, b"daily limit")])
    with pytest.raises(QuotaExhausted):
        provider.send(Request(model="m", user="u"))
    assert not waits


def test_groq_client_errors_are_not_retried():
    provider, sent, _ = groq_with([(400, {}, b"bad model")])
    with pytest.raises(ProviderError, match="400"):
        provider.send(Request(model="m", user="u"))
    assert len(sent) == 1


def test_reset_durations():
    assert parse_duration("1m26.4s") == pytest.approx(86.4)
    assert parse_duration("1.522s") == pytest.approx(1.522) and parse_duration("250ms") == pytest.approx(0.25)
    assert parse_duration("") is None


def test_groq_paces_when_token_window_is_nearly_used():
    low = {"x-ratelimit-remaining-tokens": "100", "x-ratelimit-reset-tokens": "3s"}
    provider, _, waits = groq_with([(200, low, OK_BODY), (200, {}, OK_BODY)])
    provider.send(Request(model="m", user="u"))
    assert waits == []
    provider.send(Request(model="m", user="u"))
    assert waits == [3.25]


def test_groq_json_validation_failure_comes_back_as_a_bad_answer():
    body = json.dumps({"error": {"code": "json_validate_failed", "failed_generation": '{"confidence": 0. nine}'}}).encode()
    provider, sent, _ = groq_with([(400, {}, body)])
    r = provider.send(Request(model="m", user="u", json=True))
    assert r.finish_reason == "json_validate_failed" and r.text == '{"confidence": 0. nine}' and len(sent) == 1
    with pytest.raises(ValueError):
        r.parsed()


def test_error_text_hides_the_organization_id():
    from llm.providers.groq import _snippet
    assert "org_abc123" not in _snippet(b'{"error": "limit for organization `org_abc123` reached"}')


def test_openrouter_pins_one_host_and_reads_charged_cost(tmp_path):
    from llm.providers import OpenRouterProvider
    provider = OpenRouterProvider("k", host="deepinfra", quantization="bf16", sleep=lambda s: None)
    body = json.dumps({"model": "openai/gpt-oss-120b", "provider": "DeepInfra",
                       "choices": [{"message": {"content": '{"a": 1}'}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 10, "completion_tokens": 3, "cost": 0.0000123}}).encode()
    sent = []
    provider._http = lambda method, path, b=None: (sent.append(b), (200, {}, body))[1]
    llm = LLM(provider, model="openai/gpt-oss-120b", cache=Cache(tmp_path), prices=(100.0, 100.0))
    r = llm.complete(llm.request("u", json=True, effort="low"))
    routing = sent[0]["provider"]
    assert routing == {"order": ["deepinfra"], "allow_fallbacks": False, "require_parameters": True, "quantizations": ["bf16"]}
    assert sent[0]["reasoning"]["effort"] == "low" and "reasoning_effort" not in sent[0]
    assert r.meta["served_by"] == "DeepInfra" and llm.stats["cost_usd"] == pytest.approx(0.0000123)  # charged, not estimated
    assert provider.name == "openrouter:deepinfra/bf16"  # the host is in every cache key


def test_openrouter_out_of_credit_stops_the_run():
    from llm.providers import OpenRouterProvider
    provider = OpenRouterProvider("k", host="deepinfra")
    provider._http = lambda method, path, b=None: (402, {}, b'{"error": "Insufficient credits"}')
    with pytest.raises(QuotaExhausted):
        provider.send(Request(model="m", user="u"))


def test_parallel_calls_respect_the_budget_exactly(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    llm, provider = make_llm(tmp_path, max_live_calls=20)

    def call(i):
        try:
            return llm.complete(llm.request(f"prompt {i}"))
        except BudgetExceeded:
            return None

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(call, range(50)))
    assert sum(r is not None for r in results) == 20 == len(provider.calls) == llm.stats["live_calls"]


def test_openrouter_omits_parameters_the_host_rejects():
    from llm.providers import OpenRouterProvider
    provider = OpenRouterProvider("k", host="anthropic", omit=["temperature", "seed"])
    body = provider.payload(Request(model="anthropic/claude-haiku-5.5", user="u", json=True, effort="low", sample=2))
    assert "temperature" not in body and "seed" not in body
    assert body["response_format"] == {"type": "json_object"} and body["provider"]["order"] == ["anthropic"]


def test_openrouter_credit_reports_account_balance_not_key_cap():
    from llm.providers import OpenRouterProvider
    provider = OpenRouterProvider("k", host="deepinfra")
    replies = {"/credits": json.dumps({"data": {"total_credits": 10, "total_usage": 3.5}}).encode(),
               "/key": json.dumps({"data": {"limit": 100, "limit_remaining": 96.5}}).encode()}
    provider._http = lambda method, path, b=None: (200, {}, replies[path])
    c = provider.credit()
    assert c["account_remaining"] == pytest.approx(6.5) and c["key_limit"] == 100


def test_routed_json_failure_is_a_bad_answer_not_an_error():
    from llm.providers import OpenRouterProvider
    provider = OpenRouterProvider("k", host="groq", sleep=lambda s: None)
    body = json.dumps({"error": {"message": "Upstream error from Groq: Failed to generate JSON. Please adjust your prompt."}}).encode()
    provider._http = lambda method, path, b=None: (200, {}, body)
    r = provider.send(Request(model="m", user="u", json=True))
    assert r.finish_reason == "json_validate_failed"
    with pytest.raises(ValueError):
        r.parsed()


def test_dollar_cap_stops_live_calls_but_not_cached_ones(tmp_path):
    llm, provider = make_llm(tmp_path, prices=(1e6, 0.0), max_cost_usd=2.5)  # $1 per input token: easy arithmetic
    for i in range(3):
        llm.complete(llm.request("abcd"))       # 1 input token: $1; the repeats are cached and free
    llm.complete(llm.request("efgh"))           # $2 spent
    llm.complete(llm.request("ijkl"))           # still under the cap when sent: $3 spent
    with pytest.raises(BudgetExceeded, match="max_cost_usd"):
        llm.complete(llm.request("mnop"))
    assert llm.complete(llm.request("abcd")).cached and len(provider.calls) == 3
