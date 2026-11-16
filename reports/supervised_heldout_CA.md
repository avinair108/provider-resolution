# Supervised baseline (heldout CA)

Run `20261115-181059_supervised_heldout_CA_final`. Gradient-boosted classifier over record-registry candidates, trained on Texas NPI labels outside the Phase 4 evaluation band; record threshold 0.25 picked on held-back Texas records. Aggregates only.

## Band records

| Method | All | Cardiologists |
|---|---|---|
| Splink | 73.9% | 79.9% |
| Supervised | 81.5% | 91.3% |

## Statewide, cardiologists' records

Accuracy 98.9%, precision 99.4%, recall 99.1%; 211 errors of 18,879.
