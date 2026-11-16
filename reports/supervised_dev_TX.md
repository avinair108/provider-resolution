# Supervised baseline (dev TX)

Run `20261115-174058_supervised_dev_TX`. Gradient-boosted classifier over record-registry candidates, trained on Texas NPI labels outside the Phase 4 evaluation band; record threshold 0.25 picked on held-back Texas records. Aggregates only.

Texas numbers come only from the 1,909 evaluation-band records, which training never saw; statewide figures would be in-sample.

## Band records

| Method | All | Cardiologists |
|---|---|---|
| Splink | 80.1% | 86.2% |
| Supervised | 89.1% | 95.1% |
