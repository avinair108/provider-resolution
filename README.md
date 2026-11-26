# Provider Identity Resolution: Agentic RAG + Eval

## What this is

The US government publishes three public datasets about doctors and other healthcare providers:

- **NPPES**: the official list of every provider.
- **Open Payments**: money that drug and device companies pay to doctors.
- **Medicare Part D**: the drugs doctors prescribe to Medicare patients.

All three describe the same people, but they don't share an ID that links them. So the same doctor can appear three
times with small differences: a nickname, a typo, an old address.

This project works out which records belong to the same person. It joins them into one map of who is who (an
"identity graph"). Then an AI agent answers questions using that map.

**The main goal is honest measurement.** Each provider has a real ID number, the NPI. I removed it from everything the
system can see and kept it hidden as the answer key. Every result below is checked against that hidden key.

**One rule runs through the project:** the AI suggests, and plain code checks. The AI decides what to look up and how
to word an answer. Regular code does the comparisons, checks every claim against the records, and has the final say.

## Results

I focused on cardiologists (heart doctors) in 2024, with every provider in the state as a possible match. I built and
tuned everything on **Texas**. Then I tested once on **California**, with no changes allowed. That shows whether the
method works on data it has never seen.

**How often a cardiologist's record was linked to the right person in the official list:**

| Method | Texas | California (unseen) |
|---|---|---|
| Match on exact name | 84.3% | 80.2% |
| Splink, a standard matching tool that learns without answers | 98.0% | 97.7% |
| A classic machine-learning model trained on Texas answers | — | 98.9% |
| **Splink, plus an AI model for the unclear cases** | **98.8%** | **99.0%** |

**Other results:**

| | Texas | California |
|---|---|---|
| Unclear cases only: Splink alone → with the AI | 86.2% → 96.4–96.6% | 79.9% → 93.6–94.2% |
| Quality of the identity graph (B-cubed F1) | 98.50% | 98.87% |
| Questions the agent answered correctly | 90–92% | 95–97% |
| AI cost to check the unclear cases | $2.10 | $3.46 |

Each AI result was run 3 times, so some results are ranges. The whole project cost about $17 in AI usage. More detail
is in [reports/summary.md](reports/summary.md).

## How it works

1. **Prepare the data** (`pipeline/`). Download the files, remove the NPI, shuffle record IDs, and save the answer key
   separately. The build stops if anything that looks like an NPI is left behind.
2. **Match records** (`matching/`). Splink scores how likely each pair of records is the same person.
3. **Check unclear cases with AI** (`adjudicate/`). When Splink isn't sure, an AI model gets the record and up to six
   possible matches in random order, and picks one or says "none". Code then checks each reason the AI gives against
   the actual records.
4. **Build the identity graph** (`graph/`). Plain files and a small local database, with no server. Every link records
   where it came from.
5. **Answer questions** (`qa/`). An AI agent looks things up in the graph. Every number it reports must come from its
   lookups. When it gets one wrong, code works out whether the graph or the agent was to blame.
6. **Fine-tune a smaller model** (`finetune/`). In progress; see below.

## How it's tested

- **Hidden answer key.** The NPI never reaches the matching code, the AI prompts or the agent's tools.
- **Separate test state.** All tuning happened on Texas. California was used once, at the end.
- **Unknown isn't counted as wrong.** Records with no known answer are left out of the scores.
- **Repeated runs.** AI results are run 3 times, with possible matches shuffled each time.
- **Spending limits.** The code stops a run before it goes over a set budget.

## Status and next steps

Paused, with the main pipeline complete. Next:

- **A smaller, fine-tuned model.** I trained a smaller model on Texas answers to do the AI checking step
  ([how](docs/finetuning.md)). Running it needs a hosting feature not yet turned on for my account, or a server at
  about $60 an hour. The same model without training already comes close to the bigger model, at half the cost.
  Next step: train a small free model on a free cloud GPU and test it the same way.
- **A short write-up**, and merging the many report files into a few.
- **A third test state** for any new methods, since California has now been used once.

## Run it yourself

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e .
source .venv/bin/activate
pytest                          # tests use made-up data; no downloads or API calls

python -m pipeline.download     # about 20 GB of public data
python -m pipeline.build        # prepare one state
python -m matching.run          # name matching and Splink

cp .env.example .env            # add your API keys here
python -m adjudicate.run --subset eval --repeats 3 --profile or-120b-groq --effort medium
python -m graph.build && python -m graph.evaluate
python -m qa.questions && python -m qa.run
```

Settings are in `config/`. Each module's docstring lists its own commands.

## Folders

```
config/     what data to use, file sources, AI model settings and spending limits
pipeline/   download and prepare the data, build the answer key
matching/   name matching, Splink, the machine-learning model, scoring
llm/        the single path for every AI call: safety check, cache, cost limits
adjudicate/ the AI checking step for unclear cases
graph/      the identity graph and lookups
qa/         questions, the agent, and finding the cause of each wrong answer
finetune/   training data and training scripts for the smaller model
evals/      scoring code
docs/       longer write-ups
reports/    results (totals and percentages only)
tests/      tests, using made-up providers only
```

## Data and privacy

The data is public, but it is about real people. So this project only measures accuracy. It is not a tool for looking
people up.

- The repo holds code and overall results only. No provider names, and no tables of who was paid what.
- The downloaded data stays on my computer and is never uploaded.
- Tests use made-up providers.

Data sources (2024): [NPPES](https://download.cms.gov/nppes/NPI_Files.html),
[Open Payments](https://openpaymentsdata.cms.gov/),
[Medicare Part D Prescribers](https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers).

License: [MIT](LICENSE).
