# Phase 1 data build

Scope: all individual providers in TX, program year 2024, with 3,054 cardiology providers as the focus group. Aggregates only; no provider-level data.

## Records

| Source | Records | Without known NPI |
|---|---|---|
| nppes | 391,011 | 0 |
| open_payments | 266,293 | 674 |
| part_d | 95,720 | 0 |

Open Payments rows in scope: 1,452,682. Rows with no NPI: 1,458 (0.25% of dollars). NPIs recovered from the profile file: 0 rows. Long digit runs removed: 2 Open Payments rows (address), 6 NPPES records (any text field). Recipient records per provider: mean 3.05, max 27.

## Providers by which sources contain them

| Sources | All providers | Focus providers |
|---|---|---|
| nppes | 273,440 | 282 |
| nppes+open_payments+part_d | 57,504 | 2,101 |
| nppes+part_d | 35,924 | 204 |
| nppes+open_payments | 24,143 | 161 |
| open_payments | 5,021 | 242 |
| part_d | 1,789 | 33 |
| open_payments+part_d | 503 | 31 |

## Hard-case slices (focus providers' records)

| Slice | Records |
|---|---|
| absent_from_open_payments | 486 |
| absent_from_part_d | 443 |
| address_differs_street | 4,130 |
| address_differs_zip | 3,072 |
| first_name_differs | 285 |
| initial_collision_same_city | 3,352 |
| last_name_differs_unexplained | 144 |
| last_name_matches_other_name | 48 |
| namesake_same_city | 340 |
| namesake_same_state | 2,709 |
| no_registry_counterpart | 714 |
| same_surname_same_address | 1,655 |
| shared_group_address | 3,877 |

## Perturbations

| Type | Records |
|---|---|
| address_abbreviation | 200 |
| initial_only | 200 |
| nickname | 200 |
| old_address | 200 |
| swapped_name | 200 |
| typo | 200 |

Leak check: passed over 7 tables. Cross-source rank correlation of record IDs: -0.0007.
