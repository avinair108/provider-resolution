# Question answering (heldout CA, eval set)

Run `20261115-181145_qa_heldout_CA_eval_final`: 60 questions × 3 repeat(s); model `openai/gpt-oss-120b` (openrouter:deepinfra/bf16, effort medium, agent prompt `f93e50e4892f`) over graph `graph`. Ranges over repeats. Aggregates only.

| Question type | Pass rate |
|---|---|
| ambiguous | 100% |
| conflict | 100% |
| cross_source | 100% |
| decoy | 75% |
| multi_hop | 89%–100% |
| single_source | 89%–100% |
| unanswerable | 100% |
| **all** | **95%–97%** |

| Behaviour | Value |
|---|---|
| Abstained when it should (unanswerable, ambiguous) | 100% |
| Abstained when it shouldn't | 2%–5% |
| Answer repaired after the grounding check | 7%–8% |
| Accepted answer still ungrounded | 0% |
| Ran out of steps | 0% |
| Required sources touched (routing recall) | 92% |
| Steps per question | 1.6 |

Failures (repeat 0): decoy: false_abstention ×1, decoy: wrong_value ×1

Model calls 482 (482 live, 0 cached); tokens 529,771 in / 99,891 out; cost $0.04; wall 596s.

## Where the failures come from

Failures over 3 repeat(s): 8. agent 2, graph 6.

| Cause | Question type | Reason | Failures |
|---|---|---|---|
| agent | multi_hop | wrong providers listed | 1 |
| agent | single_source | abstained on an answerable question | 1 |
| graph | decoy | provider split across 2 graph entities | 3 |
| graph | decoy | provider split across 3 graph entities | 3 |


Held-out state: the agent, prompt, verifier and graph method were frozen on Texas before this single run.
