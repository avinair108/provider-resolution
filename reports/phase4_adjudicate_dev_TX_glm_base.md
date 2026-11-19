# Phase 4: LLM adjudication of the uncertain band (dev TX, subset eval)

Run `20261115-233756_adjudicate_dev_TX_eval_glmbase_eval` over baseline `20261109-002635_matching_dev_TX_baseline`. Model `accounts/fireworks/models/glm-5p3-flash` (fireworks, profile `ft-glm-base`, prompt `f93518b0575b`), 3 repeat(s) with reshuffled candidate order. Band: top Splink score in [0.01, 0.99) and cardiology in the record's or a candidate's own fields; 1,909 records. Thresholds fixed from Phase 3 (record 0.05). Aggregates only.

## Cardiologists' records, resolved to the registry

| Method | Accuracy | Precision | Recall |
|---|---|---|---|
| Splink | 98.0% | 99.2% | 98.4% |
| Splink + LLM | 98.7% | 99.4% | 99.0% |

## On the band records themselves (labeled ones)

| Method | All | Cardiologists |
|---|---|---|
| Splink | 80.1% | 86.2% |
| Splink + LLM | 93.2%–93.7% | 95.6%–96.0% |

Changes against Splink (cardiologists, per repeat): fixed 151, broke 17, still wrong 38, fixed 147, broke 17, still wrong 42, fixed 146, broke 18, still wrong 43

## By slice (cardiologists)

| Slice | Records | Splink | Splink + LLM |
|---|---|---|---|
| address_differs_street | 4,130 | 94.7% | 97.1%–97.2% |
| address_differs_zip | 3,072 | 93.5% | 96.6%–96.7% |
| first_name_differs | 285 | 81.8% | 86.0% |
| initial_collision_same_city | 2,903 | 95.2% | 96.9%–97.0% |
| last_name_differs_unexplained | 144 | 68.8% | 73.6%–75.0% |
| last_name_matches_other_name | 48 | 83.3% | 85.4% |
| namesake_same_city | 292 | 75.3% | 81.8%–82.2% |
| namesake_same_state | 2,337 | 91.6% | 95.7%–95.9% |
| no_registry_counterpart | 714 | 90.1% | 91.9%–92.7% |
| same_surname_same_address | 1,412 | 96.0% | 96.6%–96.7% |
| shared_group_address | 3,295 | 99.0% | 99.3%–99.4% |

## Agent behaviour

| Repeat | Sent | Decided | Fallback | First try OK | Repaired | Format errors | Contradictions | Chose none | LLM accuracy | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1,909 | 1,896 | 13 | 1,875 | 21 | 0 | 34 | 213 | 94.0% | 0.064 |
| 1 | 1,909 | 1,895 | 14 | 1,875 | 20 | 0 | 34 | 198 | 93.6% | 0.062 |
| 2 | 1,909 | 1,893 | 16 | 1,874 | 19 | 0 | 35 | 200 | 93.4% | 0.068 |

Letters chosen (repeat 0): {'A': 889, 'B': 357, 'C': 165, 'D': 130, 'E': 74, 'F': 68, 'none': 213}. Splink rank of the chosen candidate (0 = Splink's top): {'0': 1489, '1': 105, '2': 38, '3': 28, '4': 11, '5': 12}.

Model calls: 5,830 (5,830 live, 0 cached); tokens 4,489,065 in / 679,921 out; cost $1.01; wall 73866s.
