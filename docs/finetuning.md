# Fine-tuning a small adjudicator

The Phase 4 adjudicator is a large open model (gpt-oss-120b) given a long prompt of matching rules. This experiment
asks whether a much smaller model, fine-tuned on labeled Texas records, can do the same job with a two-line prompt,
and how much of any gain comes from the fine-tuning rather than from the base model.

Three numbers on the same records answer that:

| | Model | Prompt | What it shows |
|---|---|---|---|
| Reference | gpt-oss-120b, medium effort (the frozen Phase 4 method) | full rules | the number to beat |
| Base | GLM 5.3 Flash, untuned | full rules | what the small model does zero-shot |
| Fine-tuned | GLM 5.3 Flash + LoRA | compact (output format only) | what training on labels adds |

The supervised classifier (`matching/supervised.py`) is the other labeled baseline: if the fine-tuned model doesn't
beat a free gradient-boosted classifier trained on the same state, the LLM isn't earning its cost.

## Training data (`finetune/prepare.py`)

**Where the examples come from.** Texas only. The pool is every labeled Texas record whose best Splink candidate scores
in [0.001, 0.999) with cardiology on either side: the adjudicator's band rule, widened from [0.01, 0.99) so there are
enough examples. **Every record of the reported Texas evaluation band is removed** (1,909 records), so the fine-tuned
model is scored on exactly the records every other method is, and none of them is in its training data. California is
never read; it stays held out.

**What an example looks like.** Each one is rendered by the adjudicator's own code (`adjudicate/prompt.py`), so
training and inference see identical text: the source record and up to six registry candidates, letters shuffled, no
Splink scores, no record IDs. The letter shuffle uses a sample number (1000) that no evaluation run uses.

**The target answer.** The gold choice, from the hidden NPI: the letter of the right registry record, or `"none"` when
the right person isn't among the candidates (8% of examples). The answer uses the same JSON the verifier checks, with
evidence filled in by the verifier's own field comparison (`verify.actual_relations`), so a target can never
contradict the records. A test checks that every target passes the verifier.

**What it can't learn.** Confidence is a constant 0.9: labels say what's right, not how sure to be. The fine-tuned
model's confidence is therefore not calibrated, and its calibration error isn't reported as a result.

**Size.** 1,200 examples drawn at random from the 2,460 in the pool (seed 0), split by hash into 1,069 for training and
131 for validation. Not reweighted toward hard cases: the training mix keeps the pool's share of easy records (87% where
Splink's top candidate is already right), so the model's prior matches what it will see.

**Where it lives.** `data/finetune/`, git-ignored: the examples are text from real records.

## Training (`finetune/train_fireworks.py`)

| Setting | Value | Why |
|---|---|---|
| Base model | GLM 5.3 Flash (`accounts/fireworks/models/glm-5p3-flash`) | the one base Fireworks serves LoRAs for serverless: per-token, no hourly GPU |
| Method | LoRA SFT, rank 16, alpha 32 | small adapter; serverless allows rank ≤ 64 |
| Loss | assistant answer only | system and user turns are masked |
| Epochs | 1 | budget; 1,069 examples × 1 epoch |
| Batch, learning rate | 16, 2e-4 constant | about 67 optimizer steps |
| Prompt | `SYSTEM_COMPACT`: the output format only | the matching rules are learned from examples; saves ~550 tokens per call |

Fireworks retired REST fine-tuning in 2026; the documented route is its training cookbook
([fw-ai-external/cookbook](https://github.com/fw-ai-external/cookbook), `training/recipes/sft_loop.py` with
`serverless=True`), which tokenizes locally and trains on Fireworks' shared pool. The cookbook lives in its own
environment, outside this repo's dependencies:

```bash
git clone https://github.com/fw-ai-external/cookbook ~/Projects/fw-cookbook
cd ~/Projects/fw-cookbook && uv venv -p 3.12 .venv
uv pip install --python .venv/bin/python -e ./training "duckdb>=2.0"
# Intel Macs: no torch >= 2.13 wheels exist, so this machine used --override with torch==2.2.2 and numpy<2.
# Training runs remotely; locally torch only handles batches. On Linux or Apple silicon, skip the override.
```

## Inference and scoring

`python -m finetune.serve load` loads the adapter for serverless inference (free to load and keep loaded; calls are
billed per token at the base model's price) and points the `ft-glm` profile at it. Scoring then uses the same runner,
verifier and repair as every other adjudicator: `python -m adjudicate.run --profile ft-glm --subset eval --repeats 3`.
Each profile names its prompt (`prompt = "full"` or `"compact"` in `config/llm.toml`), and the run records which, with
its hash.

## Budget

Fireworks spend has its own cap, `[budget] fireworks_cap_usd = 10` in `config/llm.toml`, separate from the $20
OpenRouter cap. Fireworks exposes no balance this repo reads, so spend is counted locally: every Fireworks call in the
usage log, plus each training job in a ledger (`.cache/llm/fireworks_ledger.jsonl`). The training script counts tokens
with GLM's own chat template before submitting, assumes every token is billed (Fireworks doesn't say whether masked
prompt tokens are), and refuses a job that would pass the cap. The estimate is charged to the ledger even if a job
fails.

| Item | Cost |
|---|---|
| Training: 493,681 tokens × 1 epoch × $8.89/M | $4.39 (estimate) |
| Base model, Texas evaluation band × 3 | ~$0.95 |
| Fine-tuned model, Texas evaluation band × 3 | ~$0.60 |

## Reproduce

```bash
python -m finetune.prepare --count          # pool sizes and a token estimate, writes nothing
python -m finetune.prepare --n 1200         # data/finetune/{train,val}.jsonl
~/Projects/fw-cookbook/.venv/bin/python -m finetune.train_fireworks --dry-run   # exact tokens and cost
~/Projects/fw-cookbook/.venv/bin/python -m finetune.train_fireworks             # train and promote the adapter
python -m finetune.serve load pr-adjudicator-v1
python -m adjudicate.run --profile ft-glm-base --subset eval --repeats 3 --workers 8 --tag glmbase_eval
python -m adjudicate.run --profile ft-glm --subset eval --repeats 3 --workers 8 --tag glmft_eval
python -m finetune.serve unload
```

## Results (Texas evaluation band, 1,909 records, 3 repeats)

| Adjudicator | Cardiologists' records statewide | Errors | Band, cardiologists | Band, all labeled | Fallbacks | Cost |
|---|---|---|---|---|---|---|
| Splink alone | 98.0% | 352 | 86.2% | 80.1% | – | – |
| gpt-oss-120b, medium effort, full prompt (frozen method) | 98.80–98.83% | 209–213 | 96.4–96.6% | 94.0–94.8% | 2–9 | $2.10 |
| GLM 5.3 Flash, untuned, low effort, full prompt | 98.74–98.78% | 218–224 | 95.6–96.0% | 93.2–93.7% | 13–16 | $1.01 |
| GLM 5.3 Flash + LoRA, compact prompt | not run | | | | | |

Untuned GLM 5.3 Flash lands within about 10 errors of the frozen method at half the cost, with similar calibration
(ECE 0.062–0.068 vs 0.058–0.064). It isn't small, though: Fireworks lists it as a 320-billion-parameter
mixture-of-experts model, so "Flash" means cheap per token, not a small model.

**Why the fine-tuned row is empty.** Training completed (67 steps, 8.5 minutes, adapter `pr-adjudicator-v1` READY).
Serving it did not: Fireworks' serverless LoRA serving is a preview that isn't enabled on this account (`403 serverless
LoRA is not enabled for this account`), and the alternative, an on-demand deployment of a 320B model, needs about eight
H100s, around $60 an hour. The adapter stays parked; if serverless serving is enabled, scoring it costs about $0.60.

Fireworks spend so far: $5.41 of the $10 cap ($4.39 training estimate, $1.01 base-model scoring, $0.003 smoke test).
