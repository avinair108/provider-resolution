"""The one entry point for model calls: leak check, cache, live-call budget, then the provider.

    llm = LLM.from_config()                  # config/llm.toml [default]
    r = llm.complete(llm.request(user="...", system="...", json=True))
    r.parsed()

Order matters. The leak check runs before the cache, so a prompt carrying an NPI-like value fails even if a response
is cached. The budgets (live calls, dollars) count live calls only: cached responses are free and never refused.
"""

import json
import threading
import time
import tomllib
from pathlib import Path

from llm.cache import Cache, request_key
from llm.types import BudgetExceeded, Request, Response
from pipeline.leak import LeakError, npi_like_values

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "llm.toml"


def project_capped(run_cap: float | None, provider, budget: dict) -> float | None:
    """The run's dollar cap, lowered to what's left of the project cap ([budget] project_cap_usd in config/llm.toml).

    Spent-so-far comes from the provider's account (OpenRouter /credits), so it counts every run, past and parallel,
    not only this process. A provider that can't report spend (free tiers, the fake) leaves the run cap as it is."""
    cap = budget.get("project_cap_usd")
    credit = getattr(provider, "credit", None)
    if cap is None or credit is None:
        return run_cap
    info = credit() or {}
    if "account_used" not in info:
        return run_cap
    left = max(0.0, float(cap) - info["account_used"])
    if left <= 0:
        raise BudgetExceeded(f"project cap ${cap:.2f} reached (${info['account_used']:.2f} spent on this account)")
    return left if run_cap is None else min(run_cap, left)


class LLM:
    def __init__(self, provider, model: str, cache: Cache | None = None, max_live_calls: int | None = None,
                 max_cost_usd: float | None = None,
                 defaults: dict | None = None, prices: tuple[float, float] = (0.0, 0.0), usage_log: Path | None = None):
        self.provider = provider
        self.model = model
        self.cache = cache
        self.max_live_calls = max_live_calls
        self.max_cost_usd = max_cost_usd  # dollars of live calls per run; cached answers cost nothing
        self.defaults = defaults or {}
        self.prices = prices  # $ per 1M input and output tokens; 0 on the free tier
        self.usage_log = usage_log
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "cache_hits": 0, "live_calls": 0, "input_tokens": 0, "output_tokens": 0,
                      "live_latency_s": 0.0, "cost_usd": 0.0}

    @classmethod
    def from_config(cls, profile: str = "default", path: Path = CONFIG, provider=None, **overrides) -> "LLM":
        from llm.providers import make

        config = tomllib.loads(path.read_text())
        settings = {**config[profile], **overrides}
        provider = provider or make(settings)
        return cls(
            provider=provider,
            model=settings["model"],
            cache=Cache(ROOT / settings.get("cache_dir", ".cache/llm")),
            max_live_calls=settings.get("max_live_calls"),
            max_cost_usd=project_capped(settings.get("max_cost_usd"), provider, config.get("budget", {})),
            defaults={k: settings[k] for k in ("max_tokens", "temperature", "effort") if k in settings},
            prices=(settings.get("price_input_per_mtok", 0.0), settings.get("price_output_per_mtok", 0.0)),
            usage_log=ROOT / settings["usage_log"] if settings.get("usage_log") else None,
        )

    def request(self, user: str, **fields) -> Request:
        return Request(model=self.model, user=user, **{**self.defaults, **fields})

    def complete(self, request: Request) -> Response:
        """Safe to call from several threads: bookkeeping is locked, the network call isn't."""
        found = npi_like_values(request.system + "\n" + request.user)
        if found:
            raise LeakError(f"{len(found)} NPI-like value(s) in a prompt; nothing was sent")
        key = request_key(request, self.provider.name)
        hit = self.cache.get(key) if self.cache else None
        with self._lock:
            self.stats["calls"] += 1
            if hit:
                self.stats["cache_hits"] += 1
                return hit
            if self.max_live_calls is not None and self.stats["live_calls"] >= self.max_live_calls:
                raise BudgetExceeded(f"{self.max_live_calls} live calls used this run (max_live_calls)")
            if self.max_cost_usd is not None and self.stats["cost_usd"] >= self.max_cost_usd:
                # Checked before each live call, so the overshoot is at most the calls already in flight.
                raise BudgetExceeded(f"${self.stats['cost_usd']:.2f} spent this run, cap ${self.max_cost_usd:.2f} (max_cost_usd)")
            self.stats["live_calls"] += 1  # counted when attempted, so parallel workers can't overshoot the cap
        response = self.provider.send(request)
        if self.cache:
            self.cache.put(key, request, self.provider.name, response)
        charged = response.meta.get("cost_usd")  # what the provider billed, when it says; else the config estimate
        cost = charged if charged is not None else \
            (response.input_tokens * self.prices[0] + response.output_tokens * self.prices[1]) / 1e6
        with self._lock:
            self.stats["input_tokens"] += response.input_tokens
            self.stats["output_tokens"] += response.output_tokens
            self.stats["live_latency_s"] = round(self.stats["live_latency_s"] + response.latency_s, 3)
            self.stats["cost_usd"] += cost
            if self.usage_log:
                self.usage_log.parent.mkdir(parents=True, exist_ok=True)
                with self.usage_log.open("a") as f:
                    f.write(json.dumps({"t": round(time.time()), "key": key, "provider": response.provider,
                                        "model": response.model, "input_tokens": response.input_tokens,
                                        "output_tokens": response.output_tokens, "latency_s": response.latency_s,
                                        "cost_usd": cost}) + "\n")
        return response

    def summary(self) -> dict:
        """For a run's summary.json: which model, and what it cost."""
        return {"provider": self.provider.name, "model": self.model, **self.stats,
                "limits": getattr(self.provider, "last_limits", {})}
