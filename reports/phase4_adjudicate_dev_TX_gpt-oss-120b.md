# Phase 4: LLM adjudication of the uncertain band (dev TX, subset eval)

Run `20261114-000823_adjudicate_dev_TX_eval_or_eval_v2` over baseline `20261109-002635_matching_dev_TX_baseline`. Model `openai/gpt-oss-120b` (openrouter:deepinfra/bf16, profile `or-120b`, prompt `f93518b0575b`), 3 repeat(s) with reshuffled candidate order. Band: top Splink score in [0.01, 0.99) and cardiology in the record's or a candidate's own fields; 1,909 records. Thresholds fixed from Phase 3 (record 0.05). Aggregates only.

## Cardiologists' records, resolved to the registry

| Method | Accuracy | Precision | Recall |
|---|---|---|---|
| Splink | 98.0% | 99.2% | 98.4% |
| Splink + LLM | 98.6% | 99.6%–99.6% | 98.8%–98.8% |

## On the band records themselves (labeled ones)

| Method | All | Cardiologists |
|---|---|---|
| Splink | 80.1% | 86.2% |
| Splink + LLM | 91.9%–92.5% | 94.3%–94.8% |

Changes against Splink (cardiologists, per repeat): fixed 157, broke 42, still wrong 32, fixed 156, broke 38, still wrong 33, fixed 151, broke 40, still wrong 38

## By slice (cardiologists)

| Slice | Records | Splink | Splink + LLM |
|---|---|---|---|
| address_differs_street | 4,130 | 94.7% | 96.3%–96.5% |
| address_differs_zip | 3,072 | 93.5% | 95.6%–95.8% |
| first_name_differs | 285 | 81.8% | 85.3%–86.0% |
| initial_collision_same_city | 2,903 | 95.2% | 96.5%–96.6% |
| last_name_differs_unexplained | 144 | 68.8% | 64.6%–68.1% |
| last_name_matches_other_name | 48 | 83.3% | 85.4%–87.5% |
| namesake_same_city | 292 | 75.3% | 80.1%–81.5% |
| namesake_same_state | 2,337 | 91.6% | 95.0%–95.3% |
| no_registry_counterpart | 714 | 90.1% | 95.4%–96.5% |
| same_surname_same_address | 1,412 | 96.0% | 96.5%–96.5% |
| shared_group_address | 3,295 | 99.0% | 99.0%–99.1% |

## Agent behaviour

| Repeat | Sent | Decided | Fallback | First try OK | Repaired | Format errors | Contradictions | Chose none | LLM accuracy | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1,909 | 1,904 | 5 | 1,850 | 54 | 10 | 49 | 334 | 92.6% | 0.027 |
| 1 | 1,909 | 1,904 | 5 | 1,851 | 53 | 8 | 50 | 320 | 92.3% | 0.026 |
| 2 | 1,909 | 1,904 | 5 | 1,855 | 49 | 7 | 47 | 324 | 92.0% | 0.024 |

Letters chosen (repeat 0): {'A': 848, 'B': 330, 'C': 144, 'D': 117, 'E': 72, 'F': 59, 'none': 334}. Splink rank of the chosen candidate (0 = Splink's top): {'0': 1415, '1': 93, '2': 29, '3': 17, '4': 10, '5': 6}.

Model calls: 5,898 (5,898 live, 0 cached); tokens 4,718,386 in / 1,368,135 out; cost $0.41; wall 2150s.
