"""Train the LoRA adjudicator on Fireworks serverless training. Run with the cookbook's interpreter, not the project's:

  ~/Projects/fw-cookbook/.venv/bin/python -m finetune.train_fireworks --dry-run    # exact tokens and cost, no job
  ~/Projects/fw-cookbook/.venv/bin/python -m finetune.train_fireworks              # train, promote, record cost

Fireworks retired REST fine-tuning; the documented path is its training cookbook (github.com/fw-ai-external/cookbook,
recipes/sft_loop.py with serverless=True). Tokens are counted with GLM 5.3 Flash's own chat template, every token of
every example (billing for masked prompt tokens isn't documented, so they're assumed billed), times epochs. The job
is refused if that estimate would take Fireworks spend past [budget] fireworks_cap_usd; afterwards the estimate goes
in the ledger the inference cap also reads. The validation file isn't sent: the evaluation is our own runner.
"""

import argparse
import json
import os
import time
import tomllib
from pathlib import Path

from llm.env import require
from llm.providers.fireworks import LEDGER, spent

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "finetune"
BASE_MODEL = "accounts/fireworks/models/glm-5p3-flash"
TOKENIZER = "zai-org/GLM-5.3-Flash"
RENDERER = "glm53_flash"
TRAIN_PRICE_PER_MTOK = 8.89


def count_tokens(path: Path) -> tuple[int, int, int]:
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")  # public tokenizer: don't send a stored (maybe stale) token
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    def n_tokens(messages: list[dict]) -> int:
        out = tok.apply_chat_template(messages, tokenize=True)
        return len(out["input_ids"] if hasattr(out, "keys") else out)  # transformers 5 returns a BatchEncoding

    lengths = [n_tokens(r["messages"]) for r in rows]
    return len(rows), sum(lengths), max(lengths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-model-id", default="pr-adjudicator-v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train = DATA / "train.jsonl"
    n, tokens, longest = count_tokens(train)
    estimate = tokens * args.epochs * TRAIN_PRICE_PER_MTOK / 1e6
    cap = tomllib.loads((ROOT / "config" / "llm.toml").read_text())["budget"]["fireworks_cap_usd"]
    used = spent()
    print(f"{n} examples, {tokens:,} tokens (longest {longest}), {args.epochs} epoch(s): about ${estimate:.2f}. "
          f"Fireworks spent so far ${used:.2f} of ${cap:.2f}.", flush=True)
    if used + estimate > cap:
        raise SystemExit("Refused: this job would pass the Fireworks cap. Fewer examples or epochs.")
    if args.dry_run:
        return

    os.environ["FIREWORKS_API_KEY"] = require("FIREWORKS_API_KEY")
    import training.recipes.sft_loop as sft_loop

    log_path = ROOT / "runs" / (time.strftime("%Y%m%d-%H%M%S") + "_finetune_fireworks")
    cfg = sft_loop.Config(
        serverless=True, log_path=str(log_path), base_model=BASE_MODEL, tokenizer_model=TOKENIZER,
        renderer_name=RENDERER, dataset=str(train), lora_rank=args.lora_rank, max_seq_len=max(1024, longest + 64),
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        output_model_id=args.output_model_id, close_session_on_exit=True,
    )
    t0 = time.time()
    try:
        sft_loop.main(cfg)
    finally:  # charge the estimate even when a job fails partway: better to over-count than to overrun
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a") as f:
            f.write(json.dumps({"t": round(time.time()), "job": "sft", "output_model_id": args.output_model_id,
                                "examples": n, "tokens": tokens, "epochs": args.epochs,
                                "cost_usd": round(estimate, 4), "wall_s": round(time.time() - t0)}) + "\n")
    meta = {"base_model": BASE_MODEL, "output_model_id": args.output_model_id, "examples": n, "tokens": tokens,
            "epochs": args.epochs, "lora_rank": args.lora_rank, "learning_rate": args.learning_rate,
            "batch_size": args.batch_size, "est_cost_usd": round(estimate, 4), "wall_s": round(time.time() - t0)}
    log_path.mkdir(parents=True, exist_ok=True)
    (log_path / "finetune.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
