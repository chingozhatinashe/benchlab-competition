# New regressors: recurrence, solar cycle, protons (none kept)

**What it is.** Three kinds of regressor the linear model had never been offered, from [model_review.md](model_review.md) item 4:
- **recurrence:** Bt and temperature one solar rotation earlier (26, 27 or 28 days);
- **solar cycle:** the 81-day F10.7 level and its trend;
- **protons:** the GOES ≥ 10 MeV proton flux, a marker of an Earth-connected CME shock.

Each was added to the submitted linear model and scored on the official metric.
- study: `python -m training.new_features` (`--sample 500` for the pilot);
- proton history: `python -m training.pull_protons` (`--check` compares it with SWPC's live feed);
- analysis: `src/experiments/new_features.ipynb`; results: `src/experiments/new_features.csv`.

**Result:** none passes the 1% / 4-of-6 rule. The best single candidate, the 6-hour proton mean, is 0.30% better and wins 2 of 6 folds. The best of each group together is 0.43% better (4 of 6), with the combination picked on the same folds. **Nothing is built; the submitted config stays.**

## Candidates

All are `feature_engineering.windows` specs, outside the linear selection pool (`windows.candidates`).

| Group | Candidates | Definition |
|---|---|---|
| Recurrence (`recurrence_candidates`) | `bt_lag{26,27,28}_72h`, `log_temperature_lag{26,27,28}_72h` | the mean over the 72 h starting 26, 27 or 28 days before t0: the hours one rotation before the forecast window |
| Solar cycle (`cycle_candidates`) | `f107_mean_1944h`, `f107_trend_1944h`, `f107_trend_648h` | the 81-day F10.7 mean; the 81- or 27-day mean minus the same mean 27 days earlier, per day |
| Protons (`proton_candidates`) | `log_p10_mean_6h`, `log_p10_max_24h`, `log_p10_max_72h` | log10 of the ≥ 10 MeV flux, floored at 1 pfu; hours that ended by t0 − 10 min |

The speed terms already use the speed 27 and 13.5 days earlier, so speed recurrence isn't a candidate.

## Proton history

SWPC's live feed (`services.swpc.noaa.gov/json/goes/primary/integral-protons-7-day.json`) keeps 7 days, so the history is assembled from two archives:

| Period | Source |
|---|---|
| 2016-09 → 2020-03-04 | OMNI hourly `PR-FLX_10` (GOES-13/15); OMNI's proton fluxes stop there |
| 2020-03 → 2020-10 | missing (solar minimum; no event); filled with the fold's median |
| 2020-11 → 2025-04-06 | NCEI GOES-R SGPS L2 5-minute averages, GOES-16 |
| 2025-04-07 → | the same, GOES-18 (GOES-16 on days GOES-18 lacks) |

- **No ≥ 10 MeV channel in SGPS:** the flux is the sum over the 13 differential channels of flux × the part of the channel above 10 MeV, averaged over the two sensors. The > 500 MeV integral channel is left out; with it, quiet-time values are about twice SWPC's.
- **Agreement with live:** over the last week (157 hours, quiet) the pulled flux is 1.04 × SWPC's (5–95%: 0.95–1.17). No event fell in the overlap, so event-time agreement is untested.
- **The quiet-time level varies:** yearly medians go from 0.17 pfu (2016) to 0.44 (2021) and back to 0.20 (2025). Most of this is cosmic rays, which peak at solar minimum, but sources and file versions differ too. The 1 pfu floor makes quiet times read 0, so only events count.
- **Events:** hours ≥ 10 pfu by year: 2017 191, 2018–2020 0, 2021 34, 2022 18, 2023 296, 2024 659, 2025 156, 2026 70.
- **Cost:** 2,188 daily files, about 25 min with 4 downloads at a time (IPv4 only: NCEI's IPv6 address stalls from this machine). Each is reduced to 24 hourly values; the netCDF files are not kept.

## Method

- **Base:** the submitted config (`training/linear_model_config.json`: `bt_mean_1h`, `ch_cm_lead_72h`, `xray_peak_max_72h`, `density_mean_24h`; official loss weights at blend 0.25; ridge 0.1), refitted per fold with the data fixes.
- **Variants:** each candidate added on its own; then the best of each group (by the geometric mean over folds) together.
- **Missing values** of a new candidate get the median of the fold's fit rows, so every variant fits on the same runs, as the live model would impute.
- **Score:** official S_final against the official truth, relative to persistence, as a geometric mean over the six yearly folds. The eval year is not used. The submitted config reproduces its selection score exactly (0.7926).
- **Decision rule, set before running:** keep a candidate only if it is more than 1% better than the submitted config and wins at least 4 of 6 folds.
- **Run:** a 500-run pilot (9 s; estimate 4 min), then all 13,148 runs (about 3.5 min in all).

## Results

S_final of the submitted config: 88.49 km/s (0.7926 of persistence).

| Variant | vs submitted | Folds won |
|---|---|---|
| best of each group together (`log_temperature_lag27_72h`, `f107_trend_648h`, `log_p10_mean_6h`) | −0.43% | 4 |
| `log_p10_mean_6h` | −0.30% | 2 |
| `log_p10_max_24h` | −0.10% | 3 |
| `log_temperature_lag27_72h` | −0.06% | 3 |
| `f107_trend_648h` | +0.08% | 2 |
| `log_temperature_lag28_72h` | +0.08% | 3 |
| `log_p10_max_72h` | +0.11% | 3 |
| `f107_mean_1944h` | +0.13% | 3 |
| `bt_lag28_72h` | +0.36% | 2 |
| `f107_trend_1944h` | +0.47% | 1 |
| `log_temperature_lag26_72h` | +0.47% | 2 |
| `bt_lag27_72h` | +0.98% | 1 |
| `bt_lag26_72h` | +1.22% | 1 |

**By fold** (% vs submitted, below 0 is better):

| Variant | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|
| `log_p10_mean_6h` | +0.23 | +0.64 | −2.40 | −0.73 | +0.43 | +0.07 |
| best of each group | +0.26 | −0.03 | −1.93 | −1.45 | +0.72 | −0.13 |
| `log_temperature_lag27_72h` | −0.04 | −0.89 | +0.75 | −0.59 | +0.16 | +0.21 |

## Findings

- **Recurrence adds nothing over the speed terms.** Bt one rotation earlier is worse at every lag (up to +1.2%): its link to speed is already carried by the speed 27 days earlier, and the extra column only adds variance. Temperature at 27 days ties.
- **The solar cycle adds nothing.** The level and the trend each change slowly, so within a fold they act as a near-constant offset, and the relative-to-persistence model already removes the level. The 81-day trend is worst in the 2023 and 2024 folds (+1.2–1.3%), the cycle's maximum, where its relation to speed differs from the earlier years.
- **Protons help in one fold.** The 6-hour mean gains 2.4% in 2021 (the first rise of cycle 25, with CME-driven high-speed periods) and 0.7% in 2022, but loses in 2019, 2020 and 2023. Proton events are rare (0 hours ≥ 10 pfu in 2018–2020, 34 in 2021), so the column is 0 in most runs and its coefficient rests on a few dozen events per fold.
- **The combination's 4 of 6 is not evidence:** its three members were chosen on the same folds, and it is still under half the 1% bar.
- **Live cost, if a later study wanted them:** protons need the SWPC feed above (a declared domain, 7 days kept). The 81-day F10.7 features need about 108 days of F10.7, more than SWPC's live feed keeps (about 40), so they would need a second live source.
