# Phase 4: LLM adjudication of the uncertain band (dev TX, subset eval)

Run `20261115-170051_adjudicate_dev_TX_eval_frozen_eval` over baseline `20261109-002635_matching_dev_TX_baseline`. Model `openai/gpt-oss-120b` (openrouter:groq, profile `or-120b-groq`, prompt `f93518b0575b`), 3 repeat(s) with reshuffled candidate order. Band: top Splink score in [0.01, 0.99) and cardiology in the record's or a candidate's own fields; 1,909 records. Thresholds fixed from Phase 3 (record 0.05). Aggregates only.

## Cardiologists' records, resolved to the registry

| Method | Accuracy | Precision | Recall |
|---|---|---|---|
| Splink | 98.0% | 99.2% | 98.4% |
| Splink + LLM | 98.8% | 99.5% | 99.0% |

## On the band records themselves (labeled ones)

| Method | All | Cardiologists |
|---|---|---|
| Splink | 80.1% | 86.2% |
| Splink + LLM | 94.0%–94.8% | 96.4%–96.6% |

Changes against Splink (cardiologists, per repeat): fixed 156, broke 14, still wrong 33, fixed 155, broke 16, still wrong 34, fixed 159, broke 16, still wrong 30

## By slice (cardiologists)

| Slice | Records | Splink | Splink + LLM |
|---|---|---|---|
| address_differs_street | 4,130 | 94.7% | 97.0%–97.1% |
| address_differs_zip | 3,072 | 93.5% | 96.5%–96.6% |
| first_name_differs | 285 | 81.8% | 85.6%–86.3% |
| initial_collision_same_city | 2,903 | 95.2% | 96.8%–96.9% |
| last_name_differs_unexplained | 144 | 68.8% | 71.5%–72.9% |
| last_name_matches_other_name | 48 | 83.3% | 85.4%–87.5% |
| namesake_same_city | 292 | 75.3% | 80.8%–81.5% |
| namesake_same_state | 2,337 | 91.6% | 95.5%–95.6% |
| no_registry_counterpart | 714 | 90.1% | 94.5%–94.8% |
| same_surname_same_address | 1,412 | 96.0% | 96.5%–96.6% |
| shared_group_address | 3,295 | 99.0% | 99.3%–99.3% |

## Agent behaviour

| Repeat | Sent | Decided | Fallback | First try OK | Repaired | Format errors | Contradictions | Chose none | LLM accuracy | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1,909 | 1,907 | 2 | 1,853 | 54 | 6 | 50 | 250 | 94.7% | 0.064 |
| 1 | 1,909 | 1,900 | 9 | 1,853 | 47 | 3 | 53 | 252 | 94.0% | 0.058 |
| 2 | 1,909 | 1,903 | 6 | 1,846 | 57 | 2 | 61 | 252 | 94.9% | 0.063 |

Letters chosen (repeat 0): {'A': 879, 'B': 353, 'C': 156, 'D': 130, 'E': 74, 'F': 65, 'none': 250}. Splink rank of the chosen candidate (0 = Splink's top): {'0': 1477, '1': 99, '2': 39, '3': 24, '4': 10, '5': 8}.

Model calls: 5,902 (5,902 live, 0 cached); tokens 4,952,033 in / 2,642,870 out; cost $2.10; wall 755s.
