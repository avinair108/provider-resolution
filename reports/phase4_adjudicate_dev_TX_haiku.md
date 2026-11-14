# Phase 4: LLM adjudication of the uncertain band (dev TX, subset eval)

Run `20261114-001205_adjudicate_dev_TX_eval_haiku_eval_v2` over baseline `20261109-002635_matching_dev_TX_baseline`. Model `anthropic/claude-haiku-5.5` (openrouter:anthropic, profile `or-haiku`, prompt `f93518b0575b`), 3 repeat(s) with reshuffled candidate order. Band: top Splink score in [0.01, 0.99) and cardiology in the record's or a candidate's own fields; 1,909 records. Thresholds fixed from Phase 3 (record 0.05). Aggregates only.

## Cardiologists' records, resolved to the registry

| Method | Accuracy | Precision | Recall |
|---|---|---|---|
| Splink | 98.0% | 99.2% | 98.4% |
| Splink + LLM | 98.7%–98.8% | 99.5% | 98.9%–99.0% |

## On the band records themselves (labeled ones)

| Method | All | Cardiologists |
|---|---|---|
| Splink | 80.1% | 86.2% |
| Splink + LLM | 93.3%–94.0% | 95.0%–95.9% |

Changes against Splink (cardiologists, per repeat): fixed 146, broke 26, still wrong 43, fixed 149, broke 16, still wrong 40, fixed 147, broke 18, still wrong 42

## By slice (cardiologists)

| Slice | Records | Splink | Splink + LLM |
|---|---|---|---|
| address_differs_street | 4,130 | 94.7% | 96.7%–96.9% |
| address_differs_zip | 3,072 | 93.5% | 96.0%–96.3% |
| first_name_differs | 285 | 81.8% | 85.6%–86.3% |
| initial_collision_same_city | 2,903 | 95.2% | 96.7% |
| last_name_differs_unexplained | 144 | 68.8% | 72.9%–73.6% |
| last_name_matches_other_name | 48 | 83.3% | 85.4%–87.5% |
| namesake_same_city | 292 | 75.3% | 81.5%–82.2% |
| namesake_same_state | 2,337 | 91.6% | 95.5% |
| no_registry_counterpart | 714 | 90.1% | 93.8%–94.3% |
| same_surname_same_address | 1,412 | 96.0% | 96.5%–96.6% |
| shared_group_address | 3,295 | 99.0% | 99.2%–99.3% |

## Agent behaviour

| Repeat | Sent | Decided | Fallback | First try OK | Repaired | Format errors | Contradictions | Chose none | LLM accuracy | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1,909 | 1,899 | 10 | 1,866 | 33 | 2 | 41 | 257 | 93.3% | 0.146 |
| 1 | 1,909 | 1,901 | 8 | 1,868 | 33 | 2 | 39 | 251 | 94.0% | 0.146 |
| 2 | 1,909 | 1,899 | 10 | 1,861 | 38 | 3 | 45 | 260 | 93.7% | 0.149 |

Letters chosen (repeat 0): {'A': 865, 'B': 358, 'C': 159, 'D': 124, 'E': 75, 'F': 61, 'none': 257}. Splink rank of the chosen candidate (0 = Splink's top): {'0': 1463, '1': 102, '2': 34, '3': 20, '4': 10, '5': 13}.

Model calls: 5,859 (5,859 live, 0 cached); tokens 6,908,947 in / 2,658,426 out; cost $2.02; wall 1288s.
