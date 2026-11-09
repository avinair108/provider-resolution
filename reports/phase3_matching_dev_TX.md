# Phase 3: matching baselines (dev TX)

Run `20261109-002635_matching_dev_TX_baseline`. Splink repeats: 3. Ranges are min–max over repeats. Aggregates only.

## Resolving each Open Payments and Part D record to its registry record

Cardiologists' records. A record resolves to its top candidate if unique and above the threshold.

| Method | Threshold | Accuracy | Precision | Recall | Wrong links | Missed |
|---|---|---|---|---|---|---|
| exact_name | 0.999 | 84.3% | 99.8% | 83.8% | 33 | 2,765 |
| exact_name_zip | 0.999 | 76.3% | 99.8% | 75.4% | 24 | 4,191 |
| splink | 0.05 | 98.0% | 99.2% | 98.4% | 131 | 221 |

Same, all providers in the pool:

| Method | Accuracy | Precision | Recall |
|---|---|---|---|
| exact_name | 81.8% | 99.6% | 81.4% |
| exact_name_zip | 79.1% | 99.9% | 78.5% |
| splink | 97.9% | 99.2% | 98.3%–98.3% |

## Pairs across sources (cardiologists)

| Method | Threshold | Precision | Recall | F1 | Blocking recall | B-cubed F1 (all) | Over-merged clusters |
|---|---|---|---|---|---|---|---|
| exact_name | 0.999 | 68.9% | 97.1% | 80.6% | 97.1% | 95.5% | 13,534 |
| exact_name_zip | 0.999 | 99.0% | 73.8% | 84.6% | 73.8% | 92.3% | 659 |
| splink | 0.85 | 95.5%–96.0% | 95.4%–96.0% | 95.7%–95.8% | 99.7% | 98.0%–98.1% | 3,017 |

## By slice (cardiologists' records, record resolution, first repeat)

| Slice | Records | exact_name accuracy | exact_name_zip accuracy | splink accuracy |
|---|---|---|---|---|
| address_differs_street | 4,130 | 84.3% | 23.5% | 94.7% |
| address_differs_zip | 3,072 | 83.4% | 0.0% | 93.5% |
| first_name_differs | 285 | 0.0% | 0.0% | 81.8% |
| initial_collision_same_city | 2,903 | 57.5% | 72.1% | 95.2% |
| last_name_differs_unexplained | 144 | 0.0% | 0.0% | 68.8% |
| last_name_matches_other_name | 48 | 0.0% | 0.0% | 83.3% |
| namesake_same_city | 292 | 0.0% | 41.8% | 75.3% |
| namesake_same_state | 2,337 | 0.0% | 74.1% | 91.6% |
| no_registry_counterpart | 714 | 97.3% | 98.9% | 90.1% |
| same_surname_same_address | 1,412 | 67.2% | 76.3% | 96.0% |
| shared_group_address | 3,295 | 87.7% | 75.3% | 99.0% |

## Calibration of the top candidate (splink)

Expected calibration error 0.012 over 351,739 records.

| p | Records | Mean p | Correct |
|---|---|---|---|
| 0.0–0.1 | 5,394 | 0.015 | 0.181 |
| 0.1–0.2 | 1,043 | 0.149 | 0.618 |
| 0.2–0.3 | 916 | 0.252 | 0.735 |
| 0.3–0.4 | 896 | 0.350 | 0.767 |
| 0.4–0.5 | 951 | 0.448 | 0.816 |
| 0.5–0.6 | 1,153 | 0.554 | 0.860 |
| 0.6–0.7 | 1,347 | 0.652 | 0.898 |
| 0.7–0.8 | 2,319 | 0.755 | 0.917 |
| 0.8–0.9 | 4,061 | 0.857 | 0.943 |
| 0.9–1.0 | 333,659 | 0.998 | 0.998 |
