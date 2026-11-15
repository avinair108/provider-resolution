# Provider Identity Resolution: Agentic RAG + Eval

An agent that answers questions about US healthcare providers across public CMS datasets that, as the system sees them,
share no key. It has to work out which records are the same provider and cite every link it relies on. **The eval
harness is the main deliverable:** routing, cross-source joining, conflict handling and abstention are measured
separately, against gold derived from the NPI that the system never sees.

Core principle, carried over from the 10-K project: the LLM decides what to look up and how to phrase the answer;
deterministic code supplies every comparison, verifies every claimed link, and has the final say.

Spec: [Provider Identity Resolution: Agentic RAG + Eval Plan](https://claude.ai/artifact/9Uyzs5LTXsGXvEGffteFGG)

## Status

| Phase | State |
|---|---|
| 1. Data and gold | Built and validated for Texas and California; hand check round 1 passed (see reports) |
| 2. Scorers before any LLM | Written and tested on hand-made answers |
| 3. Splink baseline | Texas: 98.0–98.1% of cardiologists' records resolved correctly, vs 84.3% for exact name ([report](reports/phase3_matching_dev_TX.md)) |
| 4. Agent, resolver, verifier | Texas: on the uncertain band, cardiologists' records go from 86.2% (Splink) to 95.0–95.9% with Claude Haiku 5.5 ($2.02), 95.4–95.6% with gpt-oss-120b on Groq ($1.18) and 94.3–94.8% with the same model on DeepInfra ($0.41), over 3 repeats; 98.0% to 98.7–98.8% across all cardiologists ([Haiku](reports/phase4_adjudicate_dev_TX_haiku.md), [gpt-oss on Groq](reports/phase4_adjudicate_dev_TX_gpt-oss-120b-groq.md), [gpt-oss on DeepInfra](reports/phase4_adjudicate_dev_TX_gpt-oss-120b.md)). California not yet scored |
| 5. Cluster review | Not started |
| 6. Comparison runs and write-up | Not started |

## Setup

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e .
source .venv/bin/activate
pytest                      # synthetic data only; no downloads or API calls
python -m pipeline.download # about 20 GB into data/raw; resumable
python -m pipeline.build    # scope, strip, gold, slices, perturbations, leak check; one build per state
python -m pipeline.clean    # dry run; add --yes to delete the raw files
python -m pipeline.handcheck # score the hand check of 50 gold links against its hidden key
python -m matching.run      # Phase 3: exact-name baselines and Splink (3 repeats) on Texas -> runs/<run_id>/
python -m pipeline.reference   # NUCC taxonomy names for prompts -> data/reference/
cp .env.example .env           # then paste GROQ_API_KEY (free tier) into .env
python -m llm.check            # confirms the key and the model in config/llm.toml
python -m adjudicate.run --subset tune --limit 50 --profile small   # iterate on the prompt
python -m adjudicate.run --subset eval                              # Phase 4 number; rerun to resume after the daily cap
```

Change the slice in `config/scope.toml`; source files are pinned in `config/sources.toml`.

**Dev and held-out:** Texas is for all tuning. California is never used for tuning and builds into its own directory,
so it is a real test of whether the system generalizes.

**Location:** keep this project out of iCloud-synced folders such as the Desktop: `data/` holds named records.

**Disk:** the raw files take about 25 GB only while building, and the built tables about 370 MB. `pipeline.clean` then
removes the raw files; a rebuild downloads them again.

**NPPES date:** the registry is the CMS file of 2024-12-09, from NBER's archive, to line up with program year 2024.

## What the build enforces

- **The candidate pool is every individual provider in the state, never filtered through the NPI.** Filtering through
  the join would give every record a guaranteed match and leak the answer. Questions target cardiologists, a focus group
  recorded only in gold, and everyone else stays in the pool as realistic decoys. Cardiology alone made matching trivial:
  exact first and last name scored about 98% precision, and few namesakes existed.
- **Columns are allowlisted, not blocklisted,** so a new upstream ID column can't slip through.
- **Record IDs are shuffled per source.** IDs in NPI order would reveal links by position; the build measures the rank
  correlation across sources and fails if it is high.
- **The leak check fails the build** on a known NPI or any 10-digit value passing the NPI check digit. The same check
  gates tool results at run time. Its errors never print the value they found.
- **Unknown is not "no match".** Records with no recoverable NPI are unlabeled and left out of the metrics instead of
  being counted as non-matches.
- **People only.** Open Payments also lists teaching hospitals, with blank names and no NPI; these are excluded.
- **Perturbation types and origins live only in gold.**
- **Long digit runs are removed from free text.** Real records have NPIs typed into addresses and credentials.
- **Gold integrity is checked on every build:** one NPI per entity, unique record IDs, one registry record per provider.

## Guardrails

The data is public but describes real people, so this stays an eval and not a lookup tool.

- Publish aggregate metrics and synthetic or anonymized examples; never a table of named providers with their payments.
- Don't host a searchable interface over the joined data.
- Commit download scripts, not data. `data/` and `runs/` are git-ignored because they hold named records.
  `reports/` holds aggregates only and is safe to commit.
- Tests use synthetic providers only.

## Layout

```
config/          scope.toml (slice), sources.toml (pinned URLs), llm.toml (model profiles; keys live in .env)
pipeline/        download, build (scope, IDs, gold, slices), perturb, leak, handcheck, clean, reference (NUCC names)
matching/        inputs (standardized, no gold), candidates (blocking, Splink, exact baselines), evaluate (vs gold), run
llm/             one call path: leak gate, request-keyed disk cache, live-call budget; providers/ (groq, fake)
adjudicate/      select (uncertain band, no gold), prompt, verify (claims vs records, one repair), run (scores vs Phase 3)
evals/
  contract.py    the agent's structured output and the gold item schema
  scorers/       answers (numeric, set, conflict, abstention), process (routing, citations),
                 linkage (pairs, clusters, blocking, calibration)
  aggregate.py   repeat ranges, per-slice rollups, cost and latency
tests/           synthetic only
reports/         aggregate build and eval reports
data/            raw/, then <split>/<state>/stripped (what the system may query) and gold (truth); git-ignored
```
