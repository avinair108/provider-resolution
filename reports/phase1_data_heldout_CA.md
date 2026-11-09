# Phase 1 data build

Scope: all individual providers in CA, program year 2024, with 4,077 cardiology providers as the focus group. Aggregates only; no provider-level data.

## Records

| Source | Records | Without known NPI |
|---|---|---|
| nppes | 851,800 | 0 |
| open_payments | 260,477 | 901 |
| part_d | 143,204 | 0 |

Open Payments rows in scope: 1,378,845. Rows with no NPI: 2,602 (0.46% of dollars). NPIs recovered from the profile file: 0 rows. Long digit runs removed: 288 Open Payments rows (address), 233 NPPES records (any text field). Recipient records per provider: mean 2.64, max 34.

## Providers by which sources contain them

| Sources | All providers | Focus providers |
|---|---|---|
| nppes | 684,778 | 622 |
| nppes+part_d | 74,353 | 605 |
| nppes+open_payments+part_d | 65,546 | 2,411 |
| nppes+open_payments | 27,123 | 219 |
| open_payments | 4,958 | 158 |
| part_d | 2,701 | 29 |
| open_payments+part_d | 604 | 33 |

## Hard-case slices (focus providers' records)

| Slice | Records |
|---|---|
| absent_from_open_payments | 1,227 |
| absent_from_part_d | 841 |
| address_differs_street | 4,223 |
| address_differs_zip | 3,278 |
| first_name_differs | 199 |
| initial_collision_same_city | 3,198 |
| last_name_differs_unexplained | 74 |
| last_name_matches_other_name | 14 |
| namesake_same_city | 350 |
| namesake_same_state | 4,162 |
| no_registry_counterpart | 486 |
| same_surname_same_address | 2,421 |
| shared_group_address | 3,968 |

## Perturbations

| Type | Records |
|---|---|
| address_abbreviation | 200 |
| initial_only | 200 |
| nickname | 200 |
| old_address | 200 |
| swapped_name | 200 |
| typo | 200 |

Leak check: passed over 7 tables. Cross-source rank correlation of record IDs: 0.0016.
