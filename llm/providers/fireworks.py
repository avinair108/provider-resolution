"""Fireworks' OpenAI-compatible inference endpoint, for the fine-tuned adjudicator (a LoRA on GLM 5.3 Flash).

Fireworks has no balance endpoint this repo uses, so spend is counted locally: every live Fireworks call in the usage
log (priced from the profile) plus each training job, which `finetune.train` appends to the ledger. That total is
held to [budget] fireworks_cap_usd, separately from the OpenRouter cap.

GLM 5.3 Flash is a reasoning model that thinks at maximum effort unless reasoning_effort is set, so profiles set it.
The first request after an adapter loads can take a minute, hence the longer timeout.
"""

import json
from pathlib import Path

from llm.providers.openai_compat import OpenAICompatible

ROOT = Path(__file__).resolve().parents[2]
USAGE_LOG = ROOT / ".cache" / "llm" / "usage.jsonl"
LEDGER = ROOT / ".cache" / "llm" / "fireworks_ledger.jsonl"


def spent(usage_log: Path = USAGE_LOG, ledger: Path = LEDGER) -> float:
    total = 0.0
    if usage_log.exists():
        for line in usage_log.read_text().splitlines():
            row = json.loads(line)
            if str(row.get("provider", "")).startswith("fireworks"):
                total += row.get("cost_usd") or 0.0
    if ledger.exists():
        total += sum(json.loads(line).get("cost_usd", 0.0) for line in ledger.read_text().splitlines())
    return total


class FireworksProvider(OpenAICompatible):
    name = "fireworks"
    label = "Fireworks"
    base_url = "https://api.fireworks.ai/inference/v1"
    budget_key = "fireworks_cap_usd"

    def __init__(self, api_key: str, usage_log: Path = USAGE_LOG, ledger: Path = LEDGER, **kw):
        kw.setdefault("timeout_s", 300)
        super().__init__(api_key, **kw)
        self.usage_log, self.ledger = usage_log, ledger

    def credit(self) -> dict:
        return {"account_used": spent(self.usage_log, self.ledger)}
