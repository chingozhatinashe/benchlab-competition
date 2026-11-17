# Near-miss combination (step 1 of the next steps)

**The question (2026-11-17).** Four changes to the submitted linear model were each at least 0.3% better on at least 4 of 6 folds, but short of the 1% bar on their own. Do they pass together? This is step 1 of [next_steps.md](next_steps.md).

**Result: (a) passes on the folds but not on the eval year, so it is not submitted (user decision, 2026-11-17).**
- **On the six folds,** all three candidates pass the rule. It picks the smallest, (a) cap 5% + density 1 h: 1.08% better, 5 of 6 folds.
- **On the eval year,** (a) is 0.45% worse than the submitted model, so the submitted model stays.
- **In staging,** (a) runs as `upload_candidate/near-miss-a`, to collect live scores next to the submitted model.
- **(b) and (c)** go back to [next_steps.md](next_steps.md) as builds for after the freeze.
- **Study:** `python -m training.near_miss` (`--pilot --folds 2024`; `--summary`). Results: `src/experiments/near_miss.csv`. Notebook: `src/experiments/near_miss.ipynb`.

## Setup

**Reference:** the submitted config before the study: density window 24 h, regressor cap 0.5% (`training.linear_model.CAP_QUANTILE`; `near_miss.reference_config` pins it), refitted per fold with the data fixes.

**The near-misses:**

| Near-miss | Change | Earlier result alone |
|---|---|---|
| cap 5% | `Config.cap_quantile` 0.05 instead of 0.005 | −0.83%, 5 of 6 ([slow_wind.md](slow_wind.md)) |
| density 1h | `density_mean_1h` instead of `density_mean_24h` | −0.31%, 4 of 6 ([official_reselection.md](official_reselection.md)) |
| wsa | `wsa_speed_lead_0h`, the offline WSA calibration of each fold (`training.wsa_model`'s cache); a run without a ring gets the fit runs' median | −0.75%, 4 of 6 ([wsa.md](wsa.md)) |
| bayes | `structural_bayes_parr_lead_0h` and `structural_bayes_mean_lead_0h` from each fold's latent Bayes stream fit (cached, no refit) | −0.41%, 4 of 6 ([bayes_features.md](bayes_features.md)) |

The protons (−0.30%, 2 of 6) fail the near-miss rule and were left out.

**Fixed before running** ([next_steps.md](next_steps.md)):
- **Three nested candidates,** cheapest at runtime first: (a) cap 5% + density 1h; (b) (a) + wsa; (c) (b) + bayes.
- **The rule:** more than 1% better than the reference (geometric mean over the six folds), winning at least 4 of 6, and slow-wind runs not more than 1% worse. The smallest passing set wins.
- **A selection-bias check:** for each fold, what the rule picks on the other five.

**Run:** the pilot on the 2024 fold took 3 min (0.9 GB peak); the six folds took 13 min. Almost all of the time goes on the Bayes features; each linear fit takes about a second.

## Six folds

| Variant | vs reference | Folds won | Slow wind | CME | Stream arrival | Fast stream | Intermediate | Passes |
|---|---|---|---|---|---|---|---|---|
| **(a) cap + density** | **−1.08%** | 5 | +0.8% | +0.2% | −1.9% | −1.7% | −1.2% | **yes, picked** |
| (b) (a) + wsa | −1.42% | 5 | −1.7% | −0.1% | −1.4% | −3.8% | −1.5% | yes |
| (c) (b) + bayes | −1.92% | 5 | −1.5% | −0.7% | −1.7% | −6.9% | −1.0% | yes |
| cap 5% | −0.83% | 5 | +0.9% | −0.1% | −1.4% | −2.9% | −0.7% | (check) |
| density 1h | −0.35% | 5 | −0.3% | −0.1% | −0.4% | +0.8% | −0.3% | (check) |
| wsa | −0.75% | 4 | −3.0% | −0.6% | +0.0% | −1.9% | −0.5% | (check) |
| bayes | −0.41% | 4 | −0.3% | −0.7% | −0.4% | −2.5% | +0.1% | (check) |

By fold (S_final relative to the reference):

| | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|
| (a) | −0.7% | −0.9% | −2.5% | −1.4% | +0.3% | −1.2% |
| (b) | −1.2% | −0.6% | −2.1% | −3.1% | +1.0% | −2.5% |
| (c) | −3.8% | −3.4% | −2.1% | −0.6% | +0.4% | −2.0% |

- **The setup reproduces the single results.** Cap, WSA and Bayes alone land where their own studies had them.
- **(a) is borderline.** Leaving one fold out, the rule picks (a) on 4 of the 6 subsets. Without 2021 or 2022, (a) fails and (b) is picked.
- **Each step beyond (a) wins only 3 of 6 folds:** (b) over (a) in 2019, 2022 and 2024; (c) over (b) in 2019, 2020 and 2023. Adding the Bayes features gains about 3% in 2019 and 2020, and loses 2.5% in 2022 and 0.5% in 2024.
- **(b) and (c) improve slow-wind runs;** (a) makes them slightly worse.

## Eval year (descriptive, once)

(a) against the submitted model, 1,448 runs, 2025-10-20 → 2026-10-16, official metric, through the live imputation:

| | all | CME | stream arrival | fast stream | intermediate | slow wind |
|---|---|---|---|---|---|---|
| (a) | 118.51 | 142.94 | 100.95 | 80.86 | 68.87 | 63.43 |
| submitted model | 117.98 | 140.94 | 104.41 | 82.98 | 69.10 | 61.34 |
| change | **+0.45%** | +1.4% | **−3.3%** | **−2.6%** | −0.3% | **+3.4%** |

- **(a) does what the folds showed:** it is better at stream arrivals and in fast streams.
- **But its weaknesses are larger here.** It loses in slow wind (+3.4%, against +0.8% on the folds) and in CME runs, which make up more of the eval year.

**Decision (user, 2026-11-17):** by the project's rule the eval year doesn't choose. But this would change the submitted model two days before the freeze, for a borderline fold gain against a measured eval-year loss. The submitted model stays.
- **(a) in staging:** `upload_candidate/near-miss-a` (commit `2043c1a`), a commit where (a) is the default. Staging scores it live next to the submitted model.
- **This branch** restores the submitted config; only the study, its results and the docs go to `main`.
