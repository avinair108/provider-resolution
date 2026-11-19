# Provider Identity Resolution: Agentic RAG + Eval

Three public CMS datasets describe the same US healthcare providers but, as this system sees them, share no key:
the **NPPES** provider registry, **Open Payments** (industry payments to doctors) and **Medicare Part D** (prescribing).
This project links their records into one identity graph, answers questions over it with an LLM agent, and measures
every step against ground truth the system never sees.

**The eval harness is the main deliverable.** Gold comes from the NPI, which is stripped from every record the system
can read; gold is loaded only at scoring time, and tests check that matching and agent code never import it.

Design rule: the LLM decides what to look up and how to phrase the answer; deterministic code does every comparison,
checks every claimed link against the records, and has the final say.

## Results

Scope: program year 2024, cardiology as the focus group, every individual provider in the state as the candidate pool.
**Texas is for development; California is held out**, scored once at the end with every method and threshold frozen.

| Linking cardiologists' records to the registry | Texas (dev) | California (held out) |
|---|---|---|
| Exact name | 84.3% | 80.2% |
| Splink (Fellegi-Sunter, unsupervised) | 98.0% | 97.7% |
| Supervised classifier (gradient boosting, Texas labels) | — | 98.9% |
| **Splink + LLM adjudicator on uncertain pairs** (gpt-oss-120b) | **98.8%** | **99.0%** |

| | Texas | California |
|---|---|---|
| Uncertain band, cardiologists: Splink → + LLM | 86.2% → 96.4–96.6% | 79.9% → 93.6–94.2% |
| Identity graph: B-cubed F1 / registry attachment | 98.50% / 99.06% | 98.87% / 99.02% |
| Questions answered correctly (60 × 3 repeats) | 90–92% | 95–97% |
| Adjudication cost | $2.10 | $3.46 |

Ranges are over 3 repeats. Full tables, ablations and model comparisons: **[reports/summary.md](reports/summary.md)**.
Total model spend for the project: about $17.

## How it works

1. **Data and gold** (`pipeline/`): download pinned CMS files, keep allowlisted columns, strip NPIs, shuffle record IDs
   per source, and build gold separately. The build fails if anything NPI-like leaks into what the system can read.
2. **Matching** (`matching/`): blocking plus Splink with EM, unsupervised. Exact-name baselines and a supervised
   classifier for comparison.
3. **Adjudication** (`adjudicate/`): records whose best candidate is uncertain (score 0.01–0.99) go to an LLM with up to
   six shuffled candidates. A deterministic verifier checks each claimed field agreement against the records.
4. **Identity graph** (`graph/`): Parquet files and DuckDB views, no server. One registry link per record, links kept
   only when consistent, provenance on every link.
5. **Question answering** (`qa/`): a tool-using agent over the graph. Every number and record it cites must appear in
   its tool results, and failures are attributed to the graph or to the agent.
6. **Fine-tuning** (`finetune/`, in progress): see below.

## Evaluation design

- **Hidden gold.** NPIs never reach matching, prompts or tools; a leak gate blocks NPI-like values in any prompt.
- **Dev and held-out.** All tuning on Texas (with a tune/eval split inside it); California run once, with `--final`.
- **Unknown is not "no match".** Records with no recoverable NPI are left out of metrics instead of counted as wrong.
- **Repeats and costs.** Every LLM result is 3 repeats with shuffled candidate order; every run records its cost.
- **Budget caps.** Per-run and per-project dollar caps are enforced in the client.

## Status and next steps

Paused with Phases 1–6 complete. Open items:

- **Fine-tuned small adjudicator.** A LoRA on GLM 5.3 Flash was trained on Texas labels ([procedure](docs/finetuning.md)),
  but serving it needs Fireworks' serverless LoRA (not enabled on this account) or an on-demand GPU deployment
  (about $60/hr). Untuned GLM 5.3 Flash already scores within about 10 errors of gpt-oss-120b at half the cost.
  Next: fine-tune a small open model (e.g. Qwen3-4B) on a free GPU and score it with the same harness.
- **Write-up**, and consolidating `reports/` into a few files.
- A third held-out state if new methods are compared, since California has now been used once.

## Setup

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e .
source .venv/bin/activate
pytest                          # synthetic data only; no downloads or API calls

python -m pipeline.download     # about 20 GB into data/raw (resumable)
python -m pipeline.build        # one build per state: strip, gold, leak check
python -m matching.run          # exact-name baselines and Splink

cp .env.example .env            # add API keys; profiles live in config/llm.toml
python -m adjudicate.run --subset eval --repeats 3 --profile or-120b-groq --effort medium
python -m graph.build && python -m graph.evaluate
python -m qa.questions && python -m qa.run
```

Scope is set in `config/scope.toml` and source files are pinned in `config/sources.toml`. Every module's docstring
gives its own commands; `--final` runs the held-out state.

## Layout

```
config/     scope, pinned sources, model profiles and budget caps (keys live in .env)
pipeline/   download, build (gold, leak check), perturbations, hand check, NUCC reference
matching/   blocking, Splink, exact baselines, supervised classifier, evaluation vs gold
llm/        one call path: leak gate, disk cache, cost caps; providers (Groq, OpenRouter, Fireworks)
adjudicate/ uncertain band, prompt, verifier, runs and comparisons
graph/      identity graph build, DuckDB views, lookups, evaluation
qa/         question sets with gold, tools, agent, failure attribution
finetune/   training data from Texas gold, Fireworks training and serving
evals/      answer, process and linkage scorers
docs/       method write-ups
reports/    aggregate results (safe to publish)
tests/      synthetic providers only
```

## Data and guardrails

The sources are public, but they describe real people, so this stays an evaluation, not a lookup tool.

- The repo holds code and aggregate metrics only. `data/`, `runs/` and `.cache/` are git-ignored; nothing here names
  a provider or pairs one with payments.
- No searchable interface over the joined data is published.
- Tests use synthetic providers.

Sources: [NPPES](https://download.cms.gov/nppes/NPI_Files.html) (December 2024 snapshot via NBER),
[Open Payments](https://openpaymentsdata.cms.gov/) and
[Medicare Part D Prescribers](https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers),
program year 2024.
