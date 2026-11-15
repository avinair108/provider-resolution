# Phase 4: LLM adjudication of the uncertain band (dev TX, subset eval)

Run `20261114-013729_adjudicate_dev_TX_eval_groq_host_eval_v2` over baseline `20261109-002635_matching_dev_TX_baseline`. Model `openai/gpt-oss-120b` (openrouter:groq, profile `or-120b-groq`, prompt `f93518b0575b`), 3 repeat(s) with reshuffled candidate order. Band: top Splink score in [0.01, 0.99) and cardiology in the record's or a candidate's own fields; 1,909 records. Thresholds fixed from Phase 3 (record 0.05). Aggregates only.

## Cardiologists' records, resolved to the registry

| Method | Accuracy | Precision | Recall |
|---|---|---|---|
| Splink | 98.0% | 99.2% | 98.4% |
| Splink + LLM | 98.7% | 99.6% | 98.9% |

## On the band records themselves (labeled ones)

| Method | All | Cardiologists |
|---|---|---|
| Splink | 80.1% | 86.2% |
| Splink + LLM | 92.7%–93.1% | 95.4%–95.6% |

Changes against Splink (cardiologists, per repeat): fixed 155, broke 26, still wrong 34, fixed 159, broke 33, still wrong 30, fixed 157, broke 28, still wrong 32

## By slice (cardiologists)

| Slice | Records | Splink | Splink + LLM |
|---|---|---|---|
| address_differs_street | 4,130 | 94.7% | 96.6%–96.7% |
| address_differs_zip | 3,072 | 93.5% | 96.0%–96.1% |
| first_name_differs | 285 | 81.8% | 85.3%–86.7% |
| initial_collision_same_city | 2,903 | 95.2% | 96.6%–96.8% |
| last_name_differs_unexplained | 144 | 68.8% | 68.1%–68.8% |
| last_name_matches_other_name | 48 | 83.3% | 85.4%–87.5% |
| namesake_same_city | 292 | 75.3% | 81.2%–81.8% |
| namesake_same_state | 2,337 | 91.6% | 95.3%–95.3% |
| no_registry_counterpart | 714 | 90.1% | 95.7%–95.9% |
| same_surname_same_address | 1,412 | 96.0% | 96.5%–96.5% |
| shared_group_address | 3,295 | 99.0% | 99.2% |

## Agent behaviour

| Repeat | Sent | Decided | Fallback | First try OK | Repaired | Format errors | Contradictions | Chose none | LLM accuracy | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1,909 | 1,906 | 3 | 1,830 | 76 | 3 | 76 | 299 | 93.1% | 0.036 |
| 1 | 1,909 | 1,902 | 7 | 1,824 | 78 | 5 | 80 | 302 | 93.2% | 0.038 |
| 2 | 1,909 | 1,902 | 7 | 1,825 | 77 | 5 | 79 | 303 | 92.7% | 0.032 |

Letters chosen (repeat 0): {'A': 859, 'B': 339, 'C': 153, 'D': 128, 'E': 70, 'F': 58, 'none': 299}. Splink rank of the chosen candidate (0 = Splink's top): {'0': 1442, '1': 93, '2': 32, '3': 21, '4': 12, '5': 7}.

Model calls: 5,975 (17 live, 5,958 cached); tokens 4,370 in / 959 out; cost $0.00; wall time not meaningful for this cached rerun (the machine slept during scoring; the original run took 554s).
