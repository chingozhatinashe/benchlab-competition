# Quantile mapping as a post-step (not built)

**What it is.** A post-step for the linear model and the hybrid. For each lead, it maps the
forecast onto the observed speed distribution, to undo the shrinkage towards the mean that
least-squares fitting causes (item 2d of [model_review.md](model_review.md), after Collin et al.
2025).
- fold study and the map (`QuantileMap`): `src/training/quantile_mapping.py`;
- analysis: `src/experiments/quantile_mapping.ipynb`.

**Result:** not built. It makes both models worse at every blend strength, including on the fast-stream runs it was meant to help.

## Method

- **The map:** for lead k, a forecast f at quantile p of the training forecasts becomes the observed training speeds' quantile p, F_y⁻¹(F_f(f)). It has 100 knots (0.5%–99.5%); beyond them the end offsets are added, so it never clips.
- **Blend:** f' = f + α (map(f) − f), with α ∈ {0, 0.25, 0.5, 0.75, 1}. α = 0 is the shipped model.
- **Fitting:** per fold, the linear model (shipped config, data fixes and the regressor cap) and the hybrid's trees on top are refitted on the data before the fold. The map is learned from their in-sample forecasts on the 6-hourly training runs.
- **Folds:** the six yearly walk-forward folds (2019-10 → 2025-10). Score: per-run mean TIMSE / persistence, as a geometric mean over folds. The eval year is not used.
- **Run:** `python -m training.quantile_mapping` (about 1 min for the six folds).

## Results

| α | Linear | vs α = 0 | Hybrid | vs α = 0 | Folds won (linear / hybrid) | Fast runs* vs α = 0 (linear / hybrid) |
|---|---|---|---|---|---|---|
| **0 (shipped)** | **0.4375** | | **0.4253** | | | |
| 0.25 | 0.4416 | +0.9% | 0.4303 | +1.2% | 1 / 1 | +12% / +11% |
| 0.5 | 0.4767 | +9.0% | 0.4595 | +8.0% | 1 / 0 | +27% / +24% |
| 0.75 | 0.5439 | +24% | 0.5122 | +20% | 0 / 0 | +43% / +38% |
| 1 | 0.6406 | +46% | 0.5868 | +38% | 0 / 0 | +62% / +54% |

\* Runs whose observed speed peaks above 600 km/s (97–527 per fold).

## Why it fails

- **The offset is deliberate.** TIMSE weights each run's fastest hours, so the forecast that minimises it sits above the plain average. Both models are fitted with those weights, and the linear forecast is 41–66 km/s above the observed mean in every fold. The map is learned from the unweighted distribution, so it removes the offset, and that costs TIMSE.
- **A wider forecast amplifies mistimed peaks.** The map widens the forecast towards the observed spread (from 0.68 to 0.95 of it for the linear model). Where a stream arrives at the wrong time, the larger excursion adds error, and TIMSE weights those hours most.
- **The 2019 fold is the exception** (−13% at α = 0.25). It is the solar-minimum year, after training years with the declining phase's fast streams, so pulling the forecast down helps there. That is a cycle-phase shift, which item 2e tests directly.
- **Why the paper differs:** Collin et al. fit and score by RMSE, without CMEs, and judge the peaks separately. Under TIMSE, the weighting already moves the forecast towards the peaks.

## Re-run on the official metric (2026-10-29)

The study above used the old per-run TIMSE, which favours an upward offset that the map removes. The official metric counts RMSE and a percentile TIMSE per lead, so the map was re-scored on it (`python -m training.quantile_mapping`, `--metric official`, now the default; analysis `src/experiments/quantile_mapping_official.ipynb`).
- **Model:** the submitted linear model only (official loss weights, blend 0.25), refitted per fold; the hybrid is no longer submitted.
- **Score:** the official S_final against the official truth, on the six yearly folds. Decision rule, set before running: more than 1% better than alpha = 0, winning at least 4 of 6 folds.

| alpha | S_final (km/s) | vs shipped | Folds won | RMSE | TRMSE | Bias (km/s) | Bias at 72 h | Spread |
|---|---|---|---|---|---|---|---|---|
| **0 (shipped)** | **88.49** | | | 68.28 | **116.20** | +21.7 | +29.7 | 0.80 |
| 0.25 | 88.52 | +0.02% | 2 | **68.08** | 116.69 | +18.6 | +25.0 | 0.83 |
| 0.5 | 88.89 | +0.45% | 2 | 68.28 | 117.36 | +15.4 | +20.3 | 0.87 |
| 0.75 | 89.59 | +1.31% | 0 | 68.88 | 118.19 | +12.2 | +15.6 | 0.91 |
| 1 | 90.61 | +2.58% | 0 | 69.88 | 119.19 | +9.1 | +11.0 | 0.95 |

**Result: still not built.** The map does what it is designed to do: it widens the forecasts towards the observed spread and roughly halves the mean bias at full strength. But the official metric doesn't reward that.
- **The tail term gets worse at every strength.** TRMSE (the percentile-TIMSE part) rises from 116.2 to 119.2 km/s. A wider forecast that is wrong about *when* a stream arrives is penalised twice, at the extremes the percentile weights emphasise.
- **RMSE improves only at alpha 0.25** (−0.3%), and that is offset by the tail term, so the score ties (+0.02%, 2 of 6 folds).
- **The bias that is left is not a distribution mismatch.** The map learns the training years' distribution, so it can't remove a fold-specific bias (+42.5 km/s on the fast 2024-10 fold stays at +35.5 km/s).
