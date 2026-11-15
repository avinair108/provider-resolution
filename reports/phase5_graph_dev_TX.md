# Phase 5: identity graph vs gold (dev TX)

Entity-level metrics over labeled records. Aggregates only.

- `graph_splink`: built 2026-11-15T05:42:31+00:00 from `20261109-002635_matching_dev_TX_baseline`, Splink alone; 405,134 entities over 753,024 records.
- `graph`: built 2026-11-15T05:42:44+00:00 from `20261109-002635_matching_dev_TX_baseline`, LLM decisions from 20261114-013729_adjudicate_dev_TX_eval_groq_host_eval_v2, 20261114-234242_adjudicate_dev_TX_tune_abl_ref (openai/gpt-oss-120b, openrouter:groq, effort low, repeat 0); 405,079 entities over 753,024 records.

## Cardiologists

| Graph | B-cubed precision | B-cubed recall | B-cubed F1 | Purity | Over-merged entities | Split providers | Registry attachment |
|---|---|---|---|---|---|---|---|
| graph_splink | 99.94% | 96.14% | 98.00% | 99.96% | 7 | 257 of 3,054 | 98.02% |
| graph | 99.91% | 96.84% | 98.35% | 99.95% | 9 | 231 of 3,054 | 98.92% |

## All providers

| Graph | B-cubed precision | B-cubed recall | B-cubed F1 | Purity | Over-merged entities | Split providers | Registry attachment |
|---|---|---|---|---|---|---|---|
| graph_splink | 99.63% | 98.48% | 99.05% | 99.71% | 1,904 | 5,027 of 398,324 | 97.95% |
| graph | 99.65% | 98.51% | 99.08% | 99.72% | 1,818 | 4,962 of 398,324 | 98.03% |
