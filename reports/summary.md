# Results: Texas (development) and California (held out)

Every method was built and tuned on Texas. California was scored once, at the end, with the method frozen and every
threshold taken from Texas. Ground truth is the NPI, which no method sees. Ranges are min–max over 3 repeats.
Aggregates only: no provider names or payment tables. Cardiologists are the focus group; the candidate pool is every
individual provider in the state.

## Linking Open Payments and Part D records to the NPPES registry (cardiologists' records, statewide)

| Method | Texas | California (held out) | Labels used | Model cost |
|---|---|---|---|---|
| Exact name | 84.3% | 80.2% | none | – |
| Exact name + ZIP | 76.3% | 77.6% | none | – |
| Splink (Fellegi-Sunter, EM) | 98.0–98.1% | 97.7% | none | – |
| Supervised classifier (gradient boosting on Splink and name/address features) | in-sample, not reported | **98.9%** | Texas NPIs | – |
| Splink + LLM adjudicator on the uncertain band (gpt-oss-120b, medium effort) | **98.8%** | **99.0%** | none | $2.10 / $3.46 |

## The uncertain band (top Splink score 0.01–0.99, cardiology by the sources' own fields)

Texas figures are on the 1,909 band records held back from all tuning; California's on all 2,995 of its band records.

| Method | Texas, cardiologists | Texas, all labeled | California, cardiologists | California, all labeled |
|---|---|---|---|---|
| Splink | 86.2% | 80.1% | 79.9% | 73.9% |
| Supervised classifier | 95.1% | 89.1% | 91.3% | 81.5% |
| + gpt-oss-120b, Groq, low effort | 95.4–95.6% | 92.7–93.1% | – | – |
| + Claude Haiku 5.5, low effort | 95.0–95.9% | 93.3–94.0% | – | – |
| + GLM 5.3 Flash, untuned, low effort (Fireworks) | 95.6–96.0% | 93.2–93.7% | – | – |
| **+ gpt-oss-120b, Groq, medium effort (frozen)** | **96.4–96.6%** | **94.0–94.8%** | **93.6–94.2%** | **89.9–90.2%** |

- The adjudicator fixes about 7 of Splink's errors for every one it breaks in Texas (155–159 fixed, 14–16 broken per
  repeat) and about 6.5 in California (276–282 fixed, 39–43 broken). Its confidence is calibrated: expected calibration
  error 0.06 in Texas and 0.01–0.02 in California. Haiku's was 0.15.
- The supervised classifier, free and trained in 100 seconds, closes most of Splink's gap on cardiologists, and nearly
  matches the LLM statewide in California (98.9% vs 99.0%) because it re-scores every candidate, not only the band. The
  LLM is still clearly ahead on the band itself, most of all across non-cardiologist records.
- The same open model scored about a point higher on Groq than on DeepInfra at full precision (95.4–95.6% vs
  94.3–94.8%, non-overlapping over 3 repeats), so each profile pins one host.

- An untuned GLM 5.3 Flash comes within about 10 errors of the frozen method statewide (218–224 vs 209–213 on Texas
  cardiologists' records) at half the cost ($1.01 vs $2.10). A fine-tuned version was trained but couldn't be served
  affordably; see [docs/finetuning.md](../docs/finetuning.md).

## Ablations (Texas tuning split, gpt-oss-120b on Groq)

| Variant | Errors on cardiologists' records statewide |
|---|---|
| Reference (low effort, shuffled candidates, scores hidden, verifier with one repair) | 320–324 |
| Medium effort | **311–314** |
| Splink scores shown | 317–318 |
| Medium effort and scores shown | 311–314 |
| Candidates in Splink's order / reversed | 325–329 / 322–325 |
| No repair after the verifier | 318–323 |
| Wider band (0.001–0.999) / narrower (0.05–0.9) | 316–323 / 323–328 |

Negative results kept: the verifier's repair step doesn't change accuracy (the verifier stays as a monitor: it flags
about 4% of answers that contradict the records), and candidate order barely matters once shuffled.

## Identity graph (cardiologists)

| | Texas, Splink | Texas, + LLM | California, Splink | California, + LLM |
|---|---|---|---|---|
| B-cubed F1 | 98.00% | **98.50%** | 98.17% | **98.87%** |
| Registry attachment | 98.02% | **99.06%** | 97.73% | **99.02%** |
| Providers split across entities | 257 of 3,054 | 210 | 253 of 4,077 | 174 |
| Entities merging two providers | 7 | 9 | 6 | 11 |

## Questions over the graph (60 per state × 3 repeats; gpt-oss-120b on DeepInfra)

| | Texas | California (held out) |
|---|---|---|
| All questions | 90–92% | **95–97%** |
| Ambiguous, unanswerable, conflict, cross-source | 100% | 100% |
| Decoy (a namesake to avoid) | 62% | 75% |
| Failures: graph / agent | 9 / 8 | 6 / 2 |
| Cost | $0.04 | $0.04 |

Gold answers come from the NPIs, not the graph, so linkage mistakes reach the score: most remaining failures are
providers the graph split. The agent's prompt and verifier were tuned on a separate 60-question Texas set with no
shared providers (97%) and frozen before either reported run.

## Cost

About $11.70 of model calls on OpenRouter plus $5.41 on Fireworks (fine-tuning experiment) over the whole project, including every ablation and model comparison. The final
frozen pipeline costs about $1.20 per 1,000 band records adjudicated (3,000 decisions with 3 repeats) and under a
cent per question answered.

Detailed reports: phase3_matching_*, phase4_adjudicate_*, phase4_ablations_dev_TX_tune, supervised_*,
phase5_graph_*, qa_*.
