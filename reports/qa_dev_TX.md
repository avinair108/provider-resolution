# Question answering (dev TX, eval set)

Run `20261115-171456_qa_dev_TX_eval_frozen`: 60 questions × 3 repeat(s); model `openai/gpt-oss-120b` (openrouter:deepinfra/bf16, effort medium, agent prompt `f93e50e4892f`) over graph `graph`. Ranges over repeats. Aggregates only.

| Question type | Pass rate |
|---|---|
| ambiguous | 100% |
| conflict | 100% |
| cross_source | 100% |
| decoy | 62% |
| multi_hop | 78% |
| single_source | 89%–100% |
| unanswerable | 100% |
| **all** | **90%–92%** |

| Behaviour | Value |
|---|---|
| Abstained when it should (unanswerable, ambiguous) | 100% |
| Abstained when it shouldn't | 2%–5% |
| Answer repaired after the grounding check | 3%–7% |
| Accepted answer still ungrounded | 0% |
| Ran out of steps | 0% |
| Required sources touched (routing recall) | 89%–92% |
| Steps per question | 1.6–1.7 |

Failures (repeat 0): decoy: false_abstention ×1, decoy: wrong_value ×2, multi_hop: failed ×2

Model calls 486 (28 live, 458 cached); tokens 51,486 in / 10,156 out; cost $0.00; wall 252s.

## Where the failures come from

Failures over 3 repeat(s): 17. agent 8, graph 9.

| Cause | Question type | Reason | Failures |
|---|---|---|---|
| agent | decoy | abstained on an answerable question | 3 |
| agent | multi_hop | wrong providers listed | 3 |
| agent | single_source | abstained on an answerable question | 2 |
| graph | decoy | graph entity's total differs from gold | 3 |
| graph | decoy | provider split across 2 graph entities | 3 |
| graph | multi_hop | a listed provider is split in the graph | 3 |


Agent prompt and verifier were frozen on a separate 60-question tuning set (no shared providers); tuning-set pass rate 97% over 3 repeats.
