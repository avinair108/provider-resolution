"""Load the trained adapter for serverless inference, unload it, or show what's loaded. Loading and keeping a LoRA
loaded is free on Fireworks; calls are billed per token at the base model's price. At most 3 adapters per account.

  python -m finetune.serve load pr-adjudicator-v1      # then: python -m adjudicate.run --profile ft-glm ...
  python -m finetune.serve status
  python -m finetune.serve unload

`load` writes the adapter's model id into the ft-glm profile in config/llm.toml. The account id is read from the
API and never printed.
"""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

from llm.env import require

ROOT = Path(__file__).resolve().parent.parent
API = "https://api.fireworks.ai/v1"
STATE = ROOT / ".cache" / "llm" / "fireworks_deployed.json"


def _call(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {require('FIREWORKS_API_KEY')}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def account() -> str:
    accounts = _call("GET", "/accounts?pageSize=5").get("accounts", [])
    if len(accounts) != 1:
        raise SystemExit(f"expected one account on this key, found {len(accounts)}")
    return accounts[0]["name"].split("/", 1)[1]


def set_profile_model(model: str) -> None:
    path = ROOT / "config" / "llm.toml"
    text = path.read_text()
    new = re.sub(r'(\[ft-glm\]\nprovider = "fireworks"\nmodel = )"[^"]*"', rf'\1"{model}"', text)
    if new == text and f'"{model}"' not in text:
        raise SystemExit("couldn't find the [ft-glm] profile's model line")
    path.write_text(new)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["load", "unload", "status"])
    parser.add_argument("adapter", nargs="?", default="pr-adjudicator-v1")
    args = parser.parse_args()
    acct = account()
    if args.action == "load":
        model = f"accounts/{acct}/models/{args.adapter}"
        print("adapter state:", _call("GET", f"/accounts/{acct}/models/{args.adapter}").get("state"))
        dep = _call("POST", f"/accounts/{acct}/deployedModels", {"model": model, "serverless": True})
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps({"name": dep.get("name"), "model": model}))
        for _ in range(60):
            state = _call("GET", f"/{dep['name']}").get("state")  # name is accounts/<acct>/deployedModels/<id>
            print("deployed model state:", state, flush=True)
            if state == "DEPLOYED":
                break
            time.sleep(10)
        set_profile_model(model)
        print("ft-glm profile now points at the adapter")
    elif args.action == "unload":
        dep = json.loads(STATE.read_text())
        _call("DELETE", f"/{dep['name']}")
        print("unloaded")
    else:
        deps = _call("GET", f"/accounts/{acct}/deployedModels").get("deployedModels", [])
        for d in deps:
            print(d.get("model", "").rsplit("/", 1)[-1], d.get("state"))
        print(f"{len(deps)} loaded")


if __name__ == "__main__":
    main()
