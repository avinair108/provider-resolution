# Phase 5: identity graph vs gold (dev TX)

Entity-level metrics over labeled records. Aggregates only.

- `graph_splink`: built 2026-11-15T23:11:07+00:00 from `20261115-170134_matching_heldout_CA_baseline`, Splink alone; 867,975 entities over 1,255,481 records.
- `graph`: built 2026-11-15T23:11:15+00:00 from `20261115-170134_matching_heldout_CA_baseline`, LLM decisions from 20261115-173702_adjudicate_heldout_CA_all_final (openai/gpt-oss-120b, openrouter:groq, effort medium, repeat 0); 867,741 entities over 1,255,481 records.

## Cardiologists

| Graph | B-cubed precision | B-cubed recall | B-cubed F1 | Purity | Over-merged entities | Split providers | Registry attachment |
|---|---|---|---|---|---|---|---|
| graph_splink | 99.94% | 96.46% | 98.17% | 99.96% | 6 | 253 of 4,077 | 97.73% |
| graph | 99.86% | 97.90% | 98.87% | 99.91% | 11 | 174 of 4,077 | 99.02% |

## All providers

| Graph | B-cubed precision | B-cubed recall | B-cubed F1 | Purity | Over-merged entities | Split providers | Registry attachment |
|---|---|---|---|---|---|---|---|
| graph_splink | 99.72% | 98.92% | 99.32% | 99.78% | 2,534 | 6,324 of 860,063 | 97.59% |
| graph | 99.74% | 98.96% | 99.35% | 99.79% | 2,431 | 6,151 of 860,063 | 97.71% |
