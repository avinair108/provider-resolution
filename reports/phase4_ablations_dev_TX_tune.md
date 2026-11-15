# Phase 4 ablations (dev TX, tuning split, gpt-oss-120b on Groq)

Subset `tune`. Splink alone makes 352 errors on cardiologists' records statewide; every variant below changes only the band records it sends to the model. Ranges over repeats. Cost is live spend for that run (cached answers are free). Aggregates only.

| Variant | Band records | Errors (cardiologists, statewide) | Accuracy | Fixed | Broken | Fallbacks | ECE | Cost |
|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b, groq, effort low, band [0.01, 0.99) | 503 | 320–324 | 98.18%–98.20% | 42–45 | 13–14 | 3–6 | 0.030–0.038 | $0.32 |
| gpt-oss-120b, groq, effort low, band [0.01, 0.99), Splink scores shown | 503 | 317–318 | 98.22% | 41–42 | 6–7 | 5–6 | 0.032–0.045 | $0.31 |
| gpt-oss-120b, groq, effort low, band [0.01, 0.99), order: splink | 503 | 325–329 | 98.15%–98.18% | 41–42 | 15–18 | 2–7 | 0.016–0.036 | $0.20 |
| gpt-oss-120b, groq, effort low, band [0.01, 0.99), order: reversed | 503 | 322–325 | 98.18%–98.19% | 41–44 | 13–16 | 2–7 | 0.028–0.048 | $0.20 |
| gpt-oss-120b, groq, effort medium, band [0.01, 0.99) | 503 | 311–314 | 98.24%–98.25% | 43–45 | 4–7 | 2–4 | 0.052–0.068 | $0.55 |
| gpt-oss-120b, groq, effort low, band [0.001, 0.999) | 903 | 316–323 | 98.19%–98.23% | 47–51 | 15–18 | 4–10 | 0.038–0.044 | $0.24 |
| gpt-oss-120b, groq, effort low, band [0.05, 0.9) | 229 | 323–328 | 98.16%–98.19% | 33–36 | 7–9 | 1–6 | 0.024–0.030 | $0.00 |
| gpt-oss-120b, groq, effort low, band [0.01, 0.99), no repair | 503 | 318–323 | 98.19%–98.22% | 43–44 | 10–14 | 7–12 | 0.030–0.043 | $0.00 |

Runs: `20261114-234242_adjudicate_dev_TX_tune_abl_ref`, `20261114-234544_adjudicate_dev_TX_tune_abl_scores`, `20261114-234544_adjudicate_dev_TX_tune_abl_order_splink`, `20261114-234852_adjudicate_dev_TX_tune_abl_order_reversed`, `20261114-234852_adjudicate_dev_TX_tune_abl_effort_medium`, `20261114-235327_adjudicate_dev_TX_tune_abl_band_wide`, `20261114-235327_adjudicate_dev_TX_tune_abl_band_narrow`, `20261114-235327_adjudicate_dev_TX_tune_abl_no_repair`
