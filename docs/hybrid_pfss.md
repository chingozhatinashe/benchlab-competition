# Hybrid (linear + XGBoost) with PFSS and WSA features (not built)

**The question.** Can the hybrid's gradient-boosted trees use the magnetic-map inputs that the
linear model could not? In the linear model:
- the expansion factor and the coronal-hole boundary distance added nothing on their own
  ([pfss.md](pfss.md));
- the WSA/HUX speed gained only 0.75% ([wsa.md](wsa.md)).

Bailey et al. (2021) found exactly these inputs decisive in gradient boosting, and the hybrid
fits on every training hour (78,888 forecast times), not only the 6-hourly runs.

- **Study:** `XGB_NTHREAD=2 python -m training.hybrid_pfss` (`--pilot --folds 2024` for the pilot).
- **Analysis:** `src/experiments/hybrid_pfss.ipynb`; results in `src/experiments/hybrid_pfss.csv`.

**Result: nothing passes the 1% / 4-of-6 rule, so nothing is built.**
- **Best variant:** hybrid + WSA, 0.10% better than the submitted linear model, on 2 of 6 folds.
- **The PFSS ring strips hurt the trees:** 1.4% worse than the hybrid without them, 1.5% worse than the linear model.
- **The hybrid on the new anchor** still ties with the linear model (+0.12%), as in the retune ([official_retune.md](official_retune.md)).

The submitted linear model stays.

## Method

- **Folds:** the six yearly folds (`training.folds.YEARLY`, 2019-10 → 2025-10). Each model is fitted on the data before its fold starts. The eval year is not used.
- **Anchor:** the submitted linear config (`bt_mean_1h`, `ch_cm_lead_72h`, `xray_peak_max_72h`, `density_mean_24h`, `pfss_d_hcs_lead_72h`; official weights, blend 0.25), fitted on the 6-hourly runs. Its PFSS gaps take the fit runs' median, as in training. It reproduces its score exactly (0.7794 of persistence).
- **Trees:** the hybrid's selected regressors (`training/xgboost_model_linear_config.json`, 100 trees), official weights at blend 0.1 (the retune's best for the hybrid), fitted on every hour before the fold.
- **New tree columns,** left missing where there is no usable map (the trees route missing values):
  - **pfss:** the ring signals `d_hcs`, `log_fs`, `theta_b`, `foot_lat`, `br_ss` at lags −24, 0, 24, 48, 72, 96 and 120 h, plus the three map scalars (open area, open flux, current-sheet maximum latitude). 38 columns.
  - **wsa:** the WSA/HUX speed at lags −72, −48, −24, 0, 24 and 48 h, with that fold's offline calibration from `src/experiments/wsa_model.csv` (fitted on the fold's training years only). HUX is solved once per map; the speed is read at the longitude that faced Earth at t0 − lag.
  - A column at lag L reads the newest map posted by t0 at the Carrington longitude facing Earth at t0 − L. Negative lags are longitudes that will face Earth after t0; the map itself is from before t0.
- **Score:** official S_final against the official truth, relative to persistence, as a geometric mean over the six folds.
- **Decision rule, set before running:** keep a hybrid variant only if it is more than 1% better than the submitted linear model and wins at least 4 of 6 folds.
- **Run:** a one-fold pilot (2024, 3 min, peak memory 1.0 GB), then all six folds (under 2 min per fold).
- **Tests:** `tests/test_wsa.py` checks that a ring constant in longitude gives one speed at every lag, and that the WSA table uses only maps posted by t0.

## Results

| Variant | / persistence | vs linear | Folds won | vs hybrid |
|---|---|---|---|---|
| submitted linear model | 0.7794 | — | — | −0.12% |
| hybrid + wsa | 0.7786 | −0.10% | 2 | −0.22% |
| hybrid | 0.7803 | +0.12% | 2 | — |
| hybrid + pfss + wsa | 0.7872 | +1.01% | 2 | +0.89% |
| hybrid + pfss | 0.7911 | +1.51% | 1 | +1.39% |

**By fold,** % change against the linear model (negative = better):

| Variant | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|
| hybrid | +0.3 | +1.0 | +1.9 | −0.1 | −3.1 | +0.8 |
| hybrid + pfss | +1.3 | +1.2 | +4.8 | +1.9 | +0.3 | −0.5 |
| hybrid + wsa | +0.6 | +0.3 | +2.2 | −1.2 | −2.9 | +0.5 |
| hybrid + pfss + wsa | +1.5 | +1.1 | +4.7 | +0.3 | −0.3 | −1.0 |

- **The pilot fold was the exception.** On 2024, hybrid + pfss + wsa was 1.0% better than the linear model and 1.8% better than the hybrid; on the other five folds the ring strips made the trees worse, by up to 3.6% (2023 fold).
- **The hybrid's gain is one fold.** It wins 2023 by 3.1% and loses or ties the other five. WSA moves that pattern by a few tenths.

## Why it doesn't help

- **The trees overfit the ring strips.** The 38 PFSS columns are strongly correlated with each other (one map, neighbouring longitudes) and with the anchor's `pfss_d_hcs_lead_72h`. With 100 shallow trees on hourly rows that repeat the same map for 12 h, the trees find splits that fit the training years' particular streams and do not carry to the next year. 2021, the first year of the new cycle, suffers most (+4.8%).
- **The anchor already takes the useful part.** Since 2026-11-07 the linear anchor includes the current-sheet distance, the PFSS signal that did pass. What remains in the other ring signals is what the linear study already found to be mostly noise at this horizon.
- **WSA is a smooth summary,** so it does no harm, but its timing errors (docs/wsa.md) limit what the trees can take from it.

## Not tried

- A regularised tree set: fewer PFSS columns (e.g. `log_fs` and `theta_b` at lag 0 only), stronger `min_child_weight`, or forward selection of ring columns (as in `training.xgboost_model select`).
- Trees fitted on the 6-hourly runs only, where each map appears once or twice.
- Other source-surface heights or map ensembles (as in docs/wsa.md).
