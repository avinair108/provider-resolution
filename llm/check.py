"""Check the key and the configured model with one tiny synthetic request (no project data).

  python -m llm.check              # the [default] profile in config/llm.toml
  python -m llm.check --profile small

Skips the cache so it really reaches the API, and prints remaining rate limits.
"""

import argparse
import tomllib

from llm.client import CONFIG, LLM
from llm.providers import make


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default="default")
    args = parser.parse_args()
    settings = tomllib.loads(CONFIG.read_text())[args.profile]
    provider = make(settings["provider"])
    if hasattr(provider, "list_models"):
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
        json=True, max_tokens=50))
    print(f"Model {r.model}: {r.text.strip()!r} in {r.latency_s:.2f}s, {r.input_tokens} in / {r.output_tokens} out tokens.")
    r.parsed()
    print("JSON parsed. Rate limits now:")
    for k, v in sorted(llm.summary()["limits"].items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
