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

1. **No model is trained on the eval year.** That covers fitting weights, early stopping, hyper-parameter search and choosing features by looking at eval data. Load data only through `training.datasets.load_train()`, or let the harness do it.
2. **Every model is scored with `training.evaluate.evaluate(model)`.** The harness calls `model.fit()` itself with the training years only, then forecasts every eval `t0`.
3. **Eval hours *before* `t0` may be used as inputs.** In the live run, a model sees recent observations, so `predict(history, t0)` gets every hour fully observed by `t0` (labels ≤ `t0 − 1h`), including earlier eval hours. It never sees hours at or after `t0`.

## Harness

```python
from training.baselines import Persistence
from training.evaluate import evaluate

evaluate(Persistence())
# {"timse": ..., "n_forecasts": ..., "n_scored": ..., "n_missing_actual": ...}
```

- **Model interface:** `fit(train: pd.Series)` and `predict(history: pd.Series, t0) -> 72 floats`.
- **Forecast times:** every 6 hours (00/06/12/18Z) from 2025-10-20 00Z, up to the last `t0` whose 72-hour window ends inside the eval year. That gives 1,448 forecasts.
- **Target alignment:** lead *k* (1–72) is compared with the hour labelled `t0 + k h`. Benchlab hasn't said exactly how it bins hours against `t0+1h … t0+72h`, so this is an assumption.
- **Scoring:** all forecasts are pooled into one TIMSE. Hours with a missing observation are excluded, following the TIMSE rule.
- **Reference score:** `Persistence` (the last observed hour, repeated) is the score a model must beat.
