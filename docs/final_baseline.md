# Reference baseline (`models.final_baseline`)

> **2026-10-26:** `tail_power` is 0.5 (was 1), chosen on the yearly folds with the official metric ([official_retune.md](official_retune.md)). It is the submitted linear model's fallback.

The model that ships in `src/` and the reference every later model is compared
against. It uses the hourly real-time ACE speed only.

## Model

For each lead k = 1…72 h at `t0`:

    R_k = a_k + b_k·(last − μ) + Σ_j c_kj·(r_j(k) − μ)      recurrence regression
    F_k = w_k·last + (1 − w_k)·R_k                         persistence blend

- `last` is the last observed hourly speed before `t0`, and μ the training mean (439 km/s).
- `r_j(k)` is the mean speed within ±3 h of `t0 + k − lag_j`.

How the five design requirements map onto the code (`Config` in `src/models/final_baseline.py`):

| Requirement | Implementation | Setting |
|---|---|---|
| (i) blend persistence and recurrence | w_k ∈ [0, 1] per lead, fitted out of sample: the regression is fitted without the last training year, run walk-forward over it, and w_k minimises the training loss there | `blend_persistence` |
| (ii) sub-rotation lags | one recurrence term per lag | `lags_days` |
| (iii) tail-weighted loss | each hour's squared error weighted by its share of its run's TIMSE (`loss="run_timse"`, exact: Σ weight·error² = per-run TIMSE) or by pooled TIMSE thresholds (`loss="timse"`), raised to `tail_power` (0 = least squares) | `loss`, `tail_power` |
| (iv) blend frozen and rolling coefficients | coef = λ·rolling + (1 − λ)·frozen. Frozen is fitted once on the training years; rolling is refitted at every `t0` on the window before it | `rolling_weight` (λ) |
| (v) tunable window | rolling refit window | `window_days` |

The rolling refit uses the same loss as the frozen fit. Training runs that end after
`t0` are cut at `t0`, and their weights are recomputed from the hours observed by then,
so the refit never sees the future. `predict(history, t0)` and the vectorised
walk-forward used for tuning share one code path. A test checks they agree.

## Tuning (training years only)

`python -m training.final_baseline tune` searches 1,080 configurations on two
walk-forward folds inside the training years:
- **fold 1:** fit 2021-10 → 2023-10, validate 2023-10 → 2024-10;
- **fold 2:** fit 2021-10 → 2024-10, validate 2024-10 → 2025-10.

Each fold forecasts every 6-hourly `t0` of its validation year. The objective is
per-run mean TIMSE relative to persistence, as a geometric mean over the two folds.
The grid:
- lags {27}, {27, 13.5}, {27, 13.5, 9} d;
- tail power 0–1 under both losses;
- window 7–90 d;
- rolling weight 0–1;
- blend on/off.

The full results are in `src/experiments/final_baseline_tuning.csv`.

**Chosen:** lags 27 + 13.5 d, `loss="run_timse"`, `tail_power=1`, window 60 d,
rolling weight 0.25, persistence blend on (`src/training/final_baseline_config.json`).

Changing one setting at a time from the chosen config (training folds, relative to persistence):

| Change | per-run mean | per-run median | pooled TIMSE | RMSE |
|---|---|---|---|---|
| **chosen** | **0.518** | 1.190 | 0.809 | 1.043 |
| no persistence blend | 0.508 | 1.304 | 0.806 | 1.080 |
| lags 27 d only | 0.520 | 1.195 | 0.817 | 1.043 |
| lags 27, 13.5, 9 d | 0.534 | 1.149 | 0.829 | 1.039 |
| pooled-threshold loss | 0.872 | 2.860 | 0.630 | 1.526 |
| tail power 0 (least squares) | 0.780 | 0.572 | 1.192 | 0.799 |
| tail power 0.5 | 0.618 | 0.731 | 0.987 | 0.861 |
| rolling weight 0 (frozen only) | 0.538 | 1.227 | 0.824 | 1.057 |
| rolling weight 1 (rolling only) | 0.611 | 1.098 | 0.810 | 1.032 |
| window 14 d | 0.539 | 1.160 | 0.788 | 1.031 |
| window 30 d | 0.520 | 1.181 | 0.769 | 1.034 |

What the search shows:
- **Training on the metric matters most.** Per-run TIMSE weights take the per-run mean from 0.78× persistence (least squares) to 0.52×.
- **The loss must match the aggregation.** A loss built on pooled thresholds gives the best pooled TIMSE (0.63×) but a poor per-run mean (0.87×). It does this by forecasting high everywhere: RMSE is 1.5× persistence.
- **The persistence blend costs about 2% on the per-run mean.** It improves the median run and RMSE, and it is kept because (i) asks for it. Its weights are large only at short leads: w = 0.67 at 1 h, 0.31 at 6 h, and 0 from 12 h on.
- **The 13.5 d lag helps slightly; adding 9 d hurts.**
- **A little rolling helps.** λ = 0.25 with a 30–90 d window beats both frozen-only and rolling-only fits.
- **The tail-weighted model gives up typical runs to win bad ones.** Its median run is worse than persistence, but its p90 is a third of persistence's. The fitted intercept rises with lead (445 km/s at 1 h, 550 km/s at 72 h): under TIMSE, over-predicting slow wind is cheap and under-predicting fast wind is expensive.

## Eval year (scored once, after the choice was fixed)

`python -m training.final_baseline evaluate`, 1,448 runs from 2025-10-20 to 2026-10-20:

| Model | pooled TIMSE | per-run mean | per-run median | per-run p90 | RMSE |
|---|---|---|---|---|---|
| persistence | 8.13 × 10⁷ | 5.57 × 10⁶ | 7.76 × 10⁵ | 1.38 × 10⁷ | 110.0 |
| recurrence_27d (frozen) | 8.40 × 10⁷ | 4.29 × 10⁶ | **4.12 × 10⁵** | 8.52 × 10⁶ | **84.4** |
| recurrence_27d (rolling) | 7.09 × 10⁷ | 3.75 × 10⁶ | 5.70 × 10⁵ | 6.90 × 10⁶ | 89.7 |
| **final_baseline** | **5.58 × 10⁷** | **2.59 × 10⁶** | 8.45 × 10⁵ | **3.21 × 10⁶** | 109.7 |

- **Best on every TIMSE view except the median.**
  - Per-run mean: 0.47× persistence and 0.69× the previous best (rolling recurrence).
  - Pooled TIMSE: 0.69× persistence.
  - p90: under half the previous best's.
- **The median run and RMSE are at persistence's level,** as on the training folds. This is the price of the tail-weighted loss.

## Running it live

`inference.run_forecast(t0)`:
1. **Fetch:** `history_hours` (about 3 months) of 1-minute ACE SWEPAM speed from the SOHO archive of SWPC's daily lists, the same source and parser as the training data, plus the newest minutes from the SWPC real-time feed. Only minutes before `t0` are used.
2. **Average** to hourly with `feature_engineering.hourly`, the same code as `training.pull_ace_rtsw`. Over 480 hours the live series matched the training CSV exactly.
3. **Predict** with the asset `src/assets/final_baseline.json`. If the model fails, the forecast falls back to persistence; with no data at all, to the training mean.

Local container run (`test_local.sh`, t0 2026-10-21 12Z):
- 83 MiB image;
- 90 daily files in 19 s;
- 23 s from launch to upload.

## Reproduce

    python -m training.final_baseline tune --fresh --max-seconds 480   # repeat until complete
    python -m training.final_baseline train
    python -m training.final_baseline evaluate
