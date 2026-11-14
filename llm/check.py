"""Check a key and a profile's model with one tiny synthetic request (no project data).

  python -m llm.check                      # the [default] profile in config/llm.toml
  python -m llm.check --profile or-120b

Skips the cache so it really reaches the API. For OpenRouter it also confirms the pinned host serves the model at
the pinned precision with every parameter we send, and shows remaining credit.
"""

import argparse
import tomllib

from llm.client import CONFIG, LLM
from llm.providers import make

NEEDED = {"response_format", "seed", "reasoning", "max_tokens"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default="default")
    args = parser.parse_args()
    settings = tomllib.loads(CONFIG.read_text())[args.profile]
    provider = make(settings)
    if hasattr(provider, "endpoints"):
        hosts = [e for e in provider.endpoints(settings["model"])
                 if settings["host"].lower() in (e["host"].lower().replace(" ", ""), (e["tag"] or "").lower())
                 and (not settings.get("quantization") or e["quantization"] == settings["quantization"])]
        if not hosts:
            raise SystemExit(f"{settings['host']} ({settings.get('quantization')}) doesn't serve {settings['model']} on OpenRouter.")
        e = hosts[0]
        missing = NEEDED - set(settings.get("omit", [])) - set(e["parameters"])
        print(f"Host {e['tag']}: ${e['input_per_mtok']:.3f} in / ${e['output_per_mtok']:.3f} out per 1M tokens; "
              f"missing parameters: {sorted(missing) or 'none'}")
        if missing:
            raise SystemExit(1)
        credit = provider.credit()
        if credit:
            print(f"Key works. Credit used ${credit['usage'] or 0:.4f}" +
                  (f", remaining ${credit['limit_remaining']:.2f}" if credit.get("limit_remaining") is not None else ""))
    elif hasattr(provider, "list_models"):
        models = provider.list_models()
        print(f"Key works. {len(models)} models available on {provider.name}.")
        if settings["model"] not in models:
            print(f"Configured model {settings['model']!r} is NOT available. Available:\n  " + "\n  ".join(models))
            raise SystemExit(1)
    llm = LLM.from_config(args.profile, provider=provider)
    llm.cache = None
    r = llm.complete(llm.request(
        system="Reply with a JSON object only.",
        user='Return JSON {"match": true or false} for whether "JON SMITH, 12 OAK ST" and "JONATHAN SMITH, 12 OAK STREET" '
             "could be the same person.",
        json=True, max_tokens=300))
    print(f"Model {r.model} via {r.meta.get('served_by') or r.provider}: {r.text.strip()!r} in {r.latency_s:.2f}s, "
          f"{r.input_tokens} in / {r.output_tokens} out tokens" +
          (f", ${r.meta['cost_usd']:.6f}" if r.meta.get("cost_usd") is not None else "") + ".")
    r.parsed()
    print("JSON parsed.")
    limits = llm.summary()["limits"]
    for k, v in sorted(limits.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
