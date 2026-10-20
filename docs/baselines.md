# Speed-only baselines

Four reference models, all in `src/training/baselines.py`. They use **only** the hourly real-time ACE speed series, and each is scored by `training.evaluate` on the held-out eval year (see [evaluation.md](evaluation.md)). Reproduce the scores with:

```bash
.venv/Scripts/python -m training.run_baselines
```

## Models

The fitted models learn separate coefficients for each lead hour k = 1…72. Training samples use every hour of the training years (2021-10-20 → 2025-10-20) as a t0. A sample is dropped when its target is missing. Hyper-parameters are fixed and were not tuned on the eval year.

| Model | Forecast for t0 + k h |
|---|---|
| `persistence` | the last observed speed, for every k |
| `climatology_blend` | μ + wₖ·(last − μ). μ is the training mean; wₖ ∈ [0, 1] is fitted by least squares |
| `recurrence_27d` | aₖ + bₖ·last + cₖ·r. r is the mean speed within ±3 h of t0 + k − 27 days (one solar rotation); fitted by least squares |
| `autoregressive_48h` | Ridge regression (α = 1) on the last 48 hourly speeds |

**Missing inputs.** If no speed has been observed, `last` becomes μ. A recurrence window with no data uses μ. A missing lag hour is replaced with `last`. These substitutions only affect the model inputs; the stored data stays un-imputed. Every input at t0 uses only hours fully observed before t0.

## Eval-year results (2025-10-20 → 2026-10-20)

1,448 forecasts (every 6 h); 102,912 hourly points scored and 1,344 skipped because the actual was missing. Lower is better.

| Model | TIMSE (km/s)³ | RMSE km/s | RMSE lead 1 h / 24 h / 72 h |
|---|---|---|---|
| **persistence** | **8.13 × 10⁷** | 110.0 | 22 / 95 / 143 |
| recurrence_27d | 8.40 × 10⁷ | **84.4** | 22 / 82 / **98** |
| climatology_blend | 9.86 × 10⁷ | 91.9 | 22 / 86 / 111 |
| autoregressive_48h | 10.43 × 10⁷ | 91.4 | 21 / 85 / 111 |

Error by observed speed (eval year; median 446, 90th percentile 622, 99th percentile 759 km/s):

| Model | bias / RMSE, < 450 km/s | 450–600 | ≥ 600 |
|---|---|---|---|
| persistence | +37 / 81 | −9 / 108 | **−121** / 187 |
| recurrence_27d | +39 / 60 | −26 / 62 | −137 / **173** |
| climatology_blend | +40 / 60 | −38 / 72 | −158 / 192 |
| autoregressive_48h | +37 / 57 | −39 / 72 | −156 / 193 |

## What this shows

- **Persistence is the TIMSE bar to beat (8.13 × 10⁷).** It has the worst RMSE but the best TIMSE.
- **TIMSE rewards getting fast wind right, and RMSE doesn't.** The fitted models minimise squared error, so they pull forecasts toward the mean. That helps RMSE but makes them under-predict high-speed streams more. Above 600 km/s, persistence is the least biased model, and TIMSE weights the upper thresholds most.
- **Recurrence carries real information at long leads.** Its 72-hour RMSE is 98 km/s against 143 for persistence, and it has the best RMSE above 600 km/s. A model that keeps this information without shrinking toward the mean is the obvious next step. Two options:
  - fit to a TIMSE-like weighted loss rather than plain least squares;
  - use persistence for short leads and recurrence for long ones.
- **Every model is too high in slow wind and too low in fast wind.** None of them predicts the onset of fast streams. That needs inputs other than past speed, such as coronal holes or solar-wind source-region models.
