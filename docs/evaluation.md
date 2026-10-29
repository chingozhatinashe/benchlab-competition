# Local evaluation: data, split and rules

> **Official metric (2026-10-25).** The organisers released the competition metric: per-lead RMSE and percentile TIMSE combined, against SWPC's hourly ACE file. It is implemented in `training.competition_score`, with the shipped models re-scored in [official_score.md](official_score.md). On it, the hybrid (S 94.0 km/s) no longer beats the linear model (93.7), and all models carried a +55 km/s bias. Re-tuning the loss for the official metric ([official_retune.md](official_retune.md)) brought the linear model to 88.5 km/s (0.79 of persistence), and **since 2026-10-26 the re-tuned linear model is the submitted model**. The TIMSE scores below are the old metric, kept for comparison.

Every model in development is scored the same way: with **TIMSE** (see [timse.md](timse.md)), on a held-out year of the same real-time L1 speed series the live leaderboard uses.

**The submitted model** (since 2026-10-24) is the XGBoost hybrid on the lead-aligned linear model ([xgboost_model.md](xgboost_model.md), [linear_model.md](linear_model.md)). It scores 0.4190 on the six yearly folds (per-run mean TIMSE / persistence), against 0.4256 for the linear model and 0.4843 for the final baseline, which is now the last model fallback. New models are compared against the hybrid.

## Data

- **Source.** SWPC's real-time ACE SWEPAM 1-minute lists, archived daily on the NASA SOHO mirror: `https://soho.nascom.nasa.gov/sdb/ace/daily/YYYYMMDD_ace_swepam_1m.txt`. These are the values SWPC published in real time, uncorrected, which is what Benchlab's provisional live scoring uses. The briefing says final scoring uses a "definitive target-data process", probably the science-quality data, which can differ a little.
- **Pull.** `python -m training.pull_ace_rtsw` writes `src/data/ace_rtsw/speed_hourly.csv` and `manifest.json`. Raw daily files are cached in `src/data/ace_rtsw/raw/`. Everything under `src/data/` is gitignored and dev-only.
- **Valid minutes.** A minute counts only if its status flag `S` is 0 and its speed isn't the fill value -9999.9.
- **Hourly values.** Each hour is labelled by its start: `HH:00` is the mean of the valid minutes from `HH:00` to `HH:59`. An hour with **no** valid minutes is left as NaN (status `missing`). **Nothing is imputed.** `n_valid` (0–60) records how many minutes went into each hour, so a stricter coverage rule can be applied later without re-pulling.

## Split (pinned in `src/training/datasets.py`)

| Set | Window (UTC, end exclusive) | Hours | Use |
|---|---|---|---|
| train | 2016-10-20 → 2025-10-20 | 78,888 | fitting, tuning, feature selection |
| eval | 2025-10-20 → 2026-10-20 | 8,760 | **scoring only** |

The dates are fixed, so re-pulling the data never moves the eval year.

**The training history was extended back from 2021-10-20 to 2016-10-20** (branch `history_extension`). The eval year and the 2021–2026 values are unchanged.
- **Coverage:** hours with a speed are 67% in late 2016, 76% in 2017, 84% in 2018 and over 90% from 2019. The older real-time ACE data is gappier.
- **Sanity check:** where it exists, it matches OMNI hourly speed (correlation 0.994, mean difference 2 km/s, in 2016-11).
- **Models trained so far** were fitted on 2021-10 → 2025-10 unless their docs say otherwise.

## Validation folds (`src/training/folds.py`)

Model choices are made on walk-forward folds inside the training years. Each fold fits on the hours before its start and scores the fold's 6-hourly runs. The objective is per-run mean TIMSE relative to persistence, as a geometric mean over folds.
- `OLD`: the two folds used by every selection so far, 2023-10 → 2024-10 and 2024-10 → 2025-10.
- `YEARLY`: six yearly folds, 2019-10 → 2025-10, possible with the longer history.

**Does the longer history help?** `python -m training.compare_history` refits the shipped configs of the final baseline, the linear model and the XGBoost hybrid on hours since 2016 or since 2021. Results are in `src/experiments/history_comparison.csv`; lower is better.

| Folds | History | Final baseline | Linear | Hybrid |
|---|---|---|---|---|
| 2022-10 → 2025-10 (common) | since 2021 | 0.5101 | 0.4747 | 0.4674 |
| 2022-10 → 2025-10 (common) | since 2016 | 0.5030 | 0.4533 | **0.4421** |
| 2019-10 → 2025-10 (all six) | since 2016 | 0.4843 | 0.4523 | **0.4342** |

- **Overall:** the longer history helps every model, the regression models most (−4.5% linear, −5.4% hybrid).
- **The year closest to the eval year gains most:** −12% for the linear model and the hybrid on fold 2024-10.
- **At solar maximum the old years don't help:** on fold 2023-10, the quiet 2016–2020 years leave the linear model level and make the baseline 4.5% worse.
- **The hybrid beats the linear model and the baseline in all six folds.**

## Studies that were not built

Scored on the yearly folds and documented, but not built:
- [Official-metric re-selection](official_reselection.md): the linear model's whole selection re-run on the official metric. It picks the shipped model again, apart from a density window worth −0.31% (4 of 6 folds); the shipped config stays.
- [New regressors](new_features.md): Bt and temperature one rotation earlier, the 81-day F10.7 level and trend, and GOES ≥ 10 MeV protons, each added to the submitted linear model (official metric, 2026-10-29). Best single −0.30% (protons, 2 of 6 folds), best of each group together −0.43% (4 of 6); none kept.
- [Regime models](regime_models.md): one model per regime, routed by a classifier. The oracle would gain 22.6%, but the best realistic method gains only 1.0%.
- [Gaussian-process correction](gaussian_process.md): a GP correction to the linear model. It gains 1.6% over the linear model but stays 1.4% behind the hybrid. Re-scored on the official metric on the re-tuned linear model (2026-10-29): best −0.38% (3 of 6 folds); the re-tune absorbed its gain.
- [HMM regimes](regime_hmm.md): hidden-Markov regime features for the hybrid. They add +0.1% at best.
- [Regime switching](regime_switching.md): a standalone model with one expert per HMM regime, weighted by the filtered state. It gains 2.1% only for experts without the regressors; with them, every multi-regime version is worse than one regression.
- [Quantile mapping](quantile_mapping.md): a post-step mapping each lead's forecast onto the observed speed distribution. It is worse at every blend strength (+0.9% at the mildest), because it removes the upward offset that TIMSE weighting makes optimal. Re-scored on the official metric (2026-10-29) it ties at the mildest strength (+0.02%, 2 of 6 folds) and is worse beyond: it narrows the bias but raises the tail term.
- [Recency weights](recency_weights.md): sample weights halving every 1–8 years into the past. Worse at every half-life (+0.3% to +5.4%): the models gain more from the older years than they lose to drift.
- [Coronal-hole latitude grid](coronal_grid.md): latitude bands, B0 and darkness of the central-meridian strip. The best addition gains 0.3%; the low-latitude band helps at solar maximum but loses up to 21% at minimum.
- [Hybrid forward-add](hybrid_forward_add.md): a second window of each regressor family added to the hybrid. Best −0.27% (3 of 6 folds); none kept, so the hybrid's pruning didn't drop signal.
- [Coronagraph features](coronagraph.md): LASCO C2/C3 brightening per position-angle sector, windowed and fed straight into the hybrid and the linear model. Best −0.36% (3 of 6 folds): CME runs gain about 3%, the other runs lose about 1%.
- [CME detector](cme_detector.md): a two-stage monotone detector (CME in the image, then CME at L1 in the next 0–72 h) whose outputs feed the hybrid. Best −0.21% (3 of 6 folds): the image stage works (AUC about 0.80), but the arrival stage is near chance (AUC 0.49–0.64).
- [Magnetogram features](magnetogram.md): SDO/HMI quick-look magnetograms on a 10° grid, central-strip field summaries fed straight into the hybrid and the linear model. Best −0.36% (4 of 6 folds): the 8-bit browse images show active regions but not the weak one-sided field of coronal holes.

## Rules

1. **Hours at or after `t0` are never used for a forecast at `t0`.** `predict(history, t0)` gets only the hours fully observed by `t0` (labels ≤ `t0 − 1h`), as in the live run.
2. **A model may refit on that history.** Coefficients don't have to be frozen: a model can refit inside `predict`, for example on the last 30 days before `t0` (`training.baselines.Rolling`). This uses only data that would be available live.
3. **Design choices are made on the training years only.** Hyper-parameters (window length, lags, regularisation), features and model selection must not be chosen by looking at eval scores. The eval year is for scoring. `model.fit()` gets the training years, from `training.datasets.load_train()` or the harness.
4. **Every model is scored with `training.evaluate.evaluate(model)`.** The harness calls `fit` with the training years, then forecasts every eval `t0`.

## Harness

```python
from training.baselines import Persistence
from training.evaluate import evaluate

evaluate(Persistence())
# {"timse": ..., "run_timse_mean": ..., "run_timse_median": ..., ..., "rmse": ..., "n_forecasts": ...}
```

- **Model interface:** `fit(train: pd.Series)` and `predict(history: pd.Series, t0) -> 72 floats`. A rolling model refits inside `predict`.
- **Forecast times:** every 6 hours (00/06/12/18Z) from 2025-10-20 00Z, up to the last `t0` whose 72-hour window ends inside the eval year. That gives 1,448 forecasts.
- **Target alignment:** lead *k* (1–72) is compared with the hour labelled `t0 + k h`. Benchlab hasn't said exactly how it bins hours against `t0+1h … t0+72h`, so this is an assumption.
- **Scoring:** two views, both lower-is-better. Hours with a missing observation are excluded, following the TIMSE rule.
  - **Pooled** (`timse`): all forecasts are combined into one TIMSE, with thresholds from the whole year.
  - **Per run** (`run_timse_mean`, `_median`, `_std`, `_p10`, `_p90`, `_max`): each 72-hour run is scored on its own hours, then summarised. A run needs at least two observed hours.
  - Benchlab hasn't said how it aggregates runs, so look at both.
- **Raw forecasts:** `run_forecasts(model)` returns `(t0s, forecasts, actuals)` for diagnostics, using the same protocol.
- **Reference scores:** the speed-only baselines in `src/experiments/baselines.ipynb`, both frozen and refitted at every `t0` on the last 30 days. Rolling 27-day recurrence has the best pooled TIMSE and per-run mean of those; frozen 27-day recurrence has the best per-run median.
- **The reference baseline** that ships, and that every new model is compared against, is `models.final_baseline` ([final_baseline.md](final_baseline.md)): per-run mean TIMSE 2.59 × 10⁶ and pooled TIMSE 5.58 × 10⁷ on the eval year. Its hyper-parameters were tuned with walk-forward folds inside the training years (`python -m training.final_baseline tune`).
