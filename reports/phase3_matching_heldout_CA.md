# Phase 3: matching baselines (heldout CA)

Run `20261115-170134_matching_heldout_CA_baseline`. Splink repeats: 3. Ranges are min–max over repeats. Aggregates only.

## Resolving each Open Payments and Part D record to its registry record

Cardiologists' records. A record resolves to its top candidate if unique and above the threshold.

| Method | Threshold | Accuracy | Precision | Recall | Wrong links | Missed |
|---|---|---|---|---|---|---|
| exact_name | 0.999 | 80.2% | 99.7% | 79.8% | 41 | 3,691 |
| exact_name_zip | 0.999 | 77.6% | 99.8% | 77.1% | 25 | 4,198 |
| splink | 0.05 | 97.7% | 99.3% | 98.0% | 124 | 304 |

Same, all providers in the pool:

| Method | Accuracy | Precision | Recall |
|---|---|---|---|
| exact_name | 77.6% | 99.6% | 77.2% |
| exact_name_zip | 80.9% | 99.9% | 80.4% |
| splink | 97.6% | 99.1% | 97.9% |

## Pairs across sources (cardiologists)

| Method | Threshold | Precision | Recall | F1 | Blocking recall | B-cubed F1 (all) | Over-merged clusters |
|---|---|---|---|---|---|---|---|
| exact_name | 0.999 | 51.1% | 98.3% | 67.3% | 98.3% | 95.1% | 23,521 |
| exact_name_zip | 0.999 | 99.0% | 75.2% | 85.5% | 75.2% | 95.5% | 921 |
| splink | 0.85 | 95.1%–95.7% | 94.9%–95.0% | 95.1%–95.3% | 99.8% | 98.5%–98.5% | 5,248 |

## By slice (cardiologists' records, record resolution, first repeat)

| Slice | Records | exact_name accuracy | exact_name_zip accuracy | splink accuracy |
|---|---|---|---|---|
| address_differs_street | 4,223 | 79.9% | 20.9% | 92.2% |
| address_differs_zip | 3,278 | 80.5% | 0.0% | 90.3% |
| first_name_differs | 199 | 0.0% | 0.0% | 76.4% |
| initial_collision_same_city | 2,662 | 44.1% | 75.1% | 94.7% |
| last_name_differs_unexplained | 74 | 0.0% | 0.0% | 50.0% |
| last_name_matches_other_name | 14 | 0.0% | 0.0% | 92.9% |
| namesake_same_city | 289 | 0.0% | 45.0% | 79.9% |
| namesake_same_state | 3,454 | 0.0% | 76.8% | 91.2% |
| no_registry_counterpart | 486 | 95.3% | 99.4% | 88.7% |
| same_surname_same_address | 1,967 | 54.3% | 72.9% | 93.1% |
| shared_group_address | 3,196 | 79.1% | 74.6% | 97.5% |

## Calibration of the top candidate (splink)

Expected calibration error 0.010 over 391,470 records.

| p | Records | Mean p | Correct |
|---|---|---|---|
| 0.0–0.1 | 5,498 | 0.016 | 0.179 |
| 0.1–0.2 | 1,100 | 0.151 | 0.602 |
| 0.2–0.3 | 1,037 | 0.250 | 0.688 |
| 0.3–0.4 | 967 | 0.351 | 0.740 |
| 0.4–0.5 | 1,118 | 0.448 | 0.834 |
| 0.5–0.6 | 1,273 | 0.551 | 0.848 |
| 0.6–0.7 | 1,572 | 0.653 | 0.872 |
| 0.7–0.8 | 2,216 | 0.756 | 0.894 |
| 0.8–0.9 | 4,062 | 0.855 | 0.927 |
| 0.9–1.0 | 372,627 | 0.998 | 0.998 |
