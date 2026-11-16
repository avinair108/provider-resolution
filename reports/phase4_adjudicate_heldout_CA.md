# Phase 4: LLM adjudication of the uncertain band (heldout CA, subset all)

Run `20261115-173702_adjudicate_heldout_CA_all_final` over baseline `20261115-170134_matching_heldout_CA_baseline`. Model `openai/gpt-oss-120b` (openrouter:groq, profile `or-120b-groq`, prompt `f93518b0575b`), 3 repeat(s) with reshuffled candidate order. Band: top Splink score in [0.01, 0.99) and cardiology in the record's or a candidate's own fields; 2,995 records. Thresholds fixed from Phase 3 (record 0.05). Aggregates only.

## Cardiologists' records, resolved to the registry

| Method | Accuracy | Precision | Recall |
|---|---|---|---|
| Splink | 97.7% | 99.3% | 98.0% |
| Splink + LLM | 99.0%–99.0% | 99.6% | 99.2% |

## On the band records themselves (labeled ones)

| Method | All | Cardiologists |
|---|---|---|
| Splink | 73.9% | 79.9% |
| Splink + LLM | 89.9%–90.2% | 93.6%–94.2% |

Changes against Splink (cardiologists, per repeat): fixed 282, broke 39, still wrong 59, fixed 277, broke 42, still wrong 64, fixed 276, broke 43, still wrong 65

## By slice (cardiologists)

| Slice | Records | Splink | Splink + LLM |
|---|---|---|---|
| address_differs_street | 4,223 | 92.2% | 97.0%–97.1% |
| address_differs_zip | 3,278 | 90.3% | 96.3%–96.4% |
| first_name_differs | 199 | 76.4% | 89.4%–91.0% |
| initial_collision_same_city | 2,662 | 94.7% | 97.9%–98.0% |
| last_name_differs_unexplained | 74 | 50.0% | 56.8%–60.8% |
| last_name_matches_other_name | 14 | 92.9% | 92.9% |
| namesake_same_city | 289 | 79.9% | 89.3%–90.0% |
| namesake_same_state | 3,454 | 91.2% | 97.3%–97.4% |
| no_registry_counterpart | 486 | 88.7% | 91.2%–92.2% |
| same_surname_same_address | 1,967 | 93.1% | 97.6%–97.8% |
| shared_group_address | 3,196 | 97.5% | 99.2%–99.3% |

## Agent behaviour

| Repeat | Sent | Decided | Fallback | First try OK | Repaired | Format errors | Contradictions | Chose none | LLM accuracy | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2,995 | 2,989 | 6 | 2,939 | 50 | 4 | 52 | 398 | 90.1% | 0.009 |
| 1 | 2,995 | 2,990 | 5 | 2,925 | 65 | 2 | 68 | 402 | 89.9% | 0.013 |
| 2 | 2,995 | 2,991 | 4 | 2,926 | 65 | 11 | 58 | 393 | 90.2% | 0.018 |

Letters chosen (repeat 0): {'A': 951, 'B': 541, 'C': 329, 'D': 309, 'E': 236, 'F': 225, 'none': 398}. Splink rank of the chosen candidate (0 = Splink's top): {'0': 2188, '1': 200, '2': 93, '3': 50, '4': 35, '5': 25}.

Model calls: 9,180 (9,180 live, 0 cached); tokens 8,413,293 in / 4,250,468 out; cost $3.46; wall 1067s.
