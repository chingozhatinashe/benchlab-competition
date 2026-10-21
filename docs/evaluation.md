# Local evaluation: data, split and rules

Every model in development is scored the same way: with **TIMSE** (see [timse.md](timse.md)), on a held-out year of the same real-time L1 speed series the live leaderboard uses.

## Data

- **Source.** SWPC's real-time ACE SWEPAM 1-minute lists, archived daily on the NASA SOHO mirror: `https://soho.nascom.nasa.gov/sdb/ace/daily/YYYYMMDD_ace_swepam_1m.txt`. These are the values SWPC published in real time, uncorrected, which is what Benchlab's provisional live scoring uses. The briefing says final scoring uses a "definitive target-data process", probably the science-quality data, which can differ a little.
- **Pull.** `python -m training.pull_ace_rtsw` writes `src/data/ace_rtsw/speed_hourly.csv` and `manifest.json`. Raw daily files are cached in `src/data/ace_rtsw/raw/`. Everything under `src/data/` is gitignored and dev-only.
- **Valid minutes.** A minute counts only if its status flag `S` is 0 and its speed isn't the fill value -9999.9.
- **Hourly values.** Each hour is labelled by its start: `HH:00` is the mean of the valid minutes from `HH:00` to `HH:59`. An hour with **no** valid minutes is left as NaN (status `missing`). **Nothing is imputed.** `n_valid` (0–60) records how many minutes went into each hour, so a stricter coverage rule can be applied later without re-pulling.

## Split (pinned in `src/training/datasets.py`)

| Set | Window (UTC, end exclusive) | Hours | Use |
|---|---|---|---|
| train | 2021-10-20 → 2025-10-20 | 35,064 | fitting, tuning, feature selection |
| eval | 2025-10-20 → 2026-10-20 | 8,760 | **scoring only** |

The dates are fixed, so re-pulling the data never moves the eval year.

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
