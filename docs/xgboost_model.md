# XGBoost model (linear + trees hybrid)

**What it is.** Gradient-boosted trees that correct the linear model's forecast
(`docs/linear_model.md`):
- runtime model: `src/models/xgboost_model.py`;
- selection, taming and training: `src/training/xgboost_model.py`;
- eval analysis: `src/experiments/xgboost_model.ipynb`.

**The shipped hybrid** was re-selected, tamed and refitted on 9 years with six yearly folds
([Nine years of history](#nine-years-of-history)). On the eval year, against its 9-year linear
anchor, it:
- wins 62% of runs;
- has a 5% lower per-run mean (2.22e6) and a 3% lower median (6.45e5), with RMSE 100.8 km/s;
- is better on pooled TIMSE (−2%).

Against the final baseline:
- per-run mean −14%;
- median −24%;
- 66% of runs won.

The sections before "Nine years of history" describe the 4-year hybrid.

**How it runs live:** the submission stays the final baseline. The hybrid runs only when
`SWP_MODEL=xgboost` is set; any failure falls back to the linear model, then the final baseline.

## Pipelines

| | Steps |
|---|---|
| **Training** | input features → feature selection → linear anchor → trees on the residual. Every hour of the training years is a forecast time, and runs with missing regressors are kept (the trees learn where NaN goes). |
| **Live** | input features → imputation → selected features → linear anchor + trees. NaN follows each split's learned default branch. The exceptions are the three regressors never missing in training (M-flare count, 120 h peak flux, 7-day F10.7): their default branch was never trained, so they take training medians (`impute_values`). The linear anchor imputes its own. |

For a run at t0 and lead k:

    forecast_k = linear_k + Σ_trees f(lead k, last speed, recurrence means at t0+k−27 d and
                 t0+k−13.5 d, linear_k − last, regressors)

## Design

- **One pooled model over all 72 leads,** with the lead as a feature. Leads share structure, so there is no per-lead noise.
- **Hourly training runs.** Fitting uses all 35,064 training hours as forecast times, not only the 6-hourly schedule. With only 6-hourly runs, lead k sees target hours at one time of day modulo 6 (the six-lead pattern of the linear model). Each run contributes a random 12 of its 72 leads to keep memory in check. Scoring stays on the 6-hourly runs.
- **Coronal-hole images for hourly runs:** a forecast time between the 6-hourly pulled images takes the latest image at or before its cut-off, at most 5 h older. That data is still available at t0. Live is unchanged: it fetches the exact cut-off.
- **Loss:** squared error with per-run TIMSE weights, the competition loss, as for the linear model. Training speeds are despiked.
- **Library:** `xgboost-cpu` 3.2.0 (CPU-only build), in `requirements-dev.txt` only.
  - **Inference is numpy:** the trees are stored as flat node arrays in `src/assets/xgboost_model.json` (about 0.1 MB, including the embedded linear model) and evaluated by `TreeEnsemble`.
  - **Parity:** it matches `xgboost.Booster.predict` to about 1e-5 km/s, with float32 comparisons, `x < threshold` going left, and NaN taking the default branch. The image doesn't contain xgboost.

## Why a hybrid

Plain XGBoost, which corrects persistence, was tried first with the same folds and search:

| Model | Fold score | Fold 1 | Fold 2 |
|---|---|---|---|
| XGBoost alone (120 regressors, depth 4, min child weight 100, 100 trees) | 0.4783 | 0.5735 | 0.3988 |
| Linear model | 0.4614 | ≈0.546 | ≈0.396 |
| Hybrid as selected (min child weight 100, 25 trees) | 0.4532 | 0.5338 | 0.3847 |
| **Hybrid, tamed (shipped: min child weight 20,000, 100 trees)** | **0.4526** | 0.5392 | **0.3798** |
| Final baseline | 0.518 | | |

- **XGBoost alone lost 5% to the linear model on fold 1.** That fold scores Oct 2023 – Oct 2024, around solar maximum, after only two years of training data from the rising phase. Trees can't forecast beyond the levels they saw in training; a linear model can extend a trend.
- **Overfitting:** every config peaked at 50–100 trees, and depth 6 was clearly worse.
- **The hybrid** keeps the linear model's ability to extrapolate and lets the trees add non-linear corrections. It wins on both folds.

## Selection (training folds only)

The folds and objective are the linear model's: walk-forward over 2023-10 → 2024-10 and
2024-10 → 2025-10, scored as the geometric mean of per-run mean TIMSE relative to persistence.
In each fold, the linear anchor is refitted on the fold's runs, as in its own selection.

- **Pilot first:** 10% of the hourly runs, 3 min. The full selection took 7 min, with a peak of 2.2 GB.
- **Stage 1, candidate regressor sets** (default depth 4, min child weight 5):
  - A: the linear model's 4 regressors, 0.4636;
  - B: one window per family, 24 regressors, 0.4589;
  - **C: B plus coronal-hole strips at 0/24/48/72 h and the eastern and full-disk strips, 31 regressors, 0.4549;**
  - D: all 120, 0.4578.
- **Stage 2, depth × min child weight on C:** depth 2 with min child weight 100 or 500 scored 0.4533. Shallow trees and few of them work best: the corrections are small.
- **Tree counts** from 5 to 800 were scored. The first full run's best sat at its smallest count (25), so the range was extended down to 5; 25 remained best.
- **Stage 3, pruning:** keeping the 25 regressors that carry 95% of the gain scored 0.4532 and was kept.
- **Regressors** (25 for the trees, plus the linear model's own 4):
  - **solar wind:** Bt (6 h max and mean), Bz (24 h min), density (24 h mean; 27-day and 13.5-day lags), pressure (27-day lag), temperature and temperature ratio (48 h), polarity (48 h mean, 27-day lag), ICME share (48 h), speed (48 h mean), EPAM (1 h);
  - **coronal holes:** central-meridian strip at 0/12/24/48/72 h, eastern strip at 24 h, full disk at 0 h;
  - **solar activity:** flares (120 h peak and M-class count), F10.7 (7 days).

## Taming the flare-count rule (`python -m training.xgboost_model tame`)

**The problem.** The eval analysis of the selected hybrid found one dominant rule:
- 15 of its 25 trees split first on the 5-day M-flare count at ≥ 48–53;
- that condition held in only 0.66% of training hours, in May and August 2024;
- in the eval year it fired only on 4–7 February 2026. It added about 111 km/s to runs that stayed slow and gave back half of the hybrid's gain over the linear model.

**Why the folds can't catch it.** Fold 1's training data never exceeds 33 flares, and fold 2's
validation year peaks at 46. So variants were judged on two things, both from training data only:
- the fold score;
- **leaf support:** the number of distinct training days whose runs reach each leaf.

**The rule,** fixed before scoring: keep the best fold score among the variants with no leaf
under 30 days (about a solar rotation).

| Variant | Fold score | Weakest leaf (days) | Largest thin-leaf correction (km/s) |
|---|---|---|---|
| selected (min child weight 100) | 0.4532 | 3 | 150 |
| min child weight 1,000 / 3,000 / 5,000 | 0.4495 / 0.4502 / 0.4518 | 3 / 9 / 19 | 196 / 110 / 62 |
| min child weight 10,000 / **20,000** / 40,000 | 0.4550 / **0.4526** / 0.4558 | 44 / **72** / 193 | 0 |
| min child weight 10,000, depth 3 | 0.4538 | 49 | 0 |
| no flare count (min child weight 100 / 1,000 / 3,000) | 0.4545 / 0.4509 / 0.4505 | 2 / 5 / 9 | 72 / 77 / 49 |
| no flare inputs (min child weight 3,000) | 0.4518 | 14 | 53 |

- **Dropping the flare inputs doesn't remove thin leaves:** other rare conditions take their place.
- **A large minimum leaf weight does.**
- **Shipped:** depth 2, min child weight 20,000, learning rate 0.05, 100 trees, subsample and colsample 0.8.
  - Its 341 leaves are each reached on at least 72 training days (median 398).
  - The flare count is now split only at 13–19.
  - The root splits are led by 7-day F10.7, EPAM 310 keV and the full-disk coronal-hole area.

## Eval year (2025-10-20 → 2026-10-20, 1,448 runs)

| | Hybrid (tamed) | Hybrid as selected | Linear model | Final baseline | Persistence |
|---|---|---|---|---|---|
| Pooled TIMSE | **4.36e7** | 4.48e7 | 4.74e7 | 5.58e7 | 8.13e7 |
| Per-run mean | **2.22e6** | 2.33e6 | 2.37e6 | 2.59e6 | 5.57e6 |
| Per-run median | 7.04e5 | 6.94e5 | **6.76e5** | 8.45e5 | 7.76e5 |
| Per-run p90 | 3.62e6 | 3.73e6 | 3.76e6 | **3.21e6** | 1.38e7 |
| RMSE (km/s) | 104.6 | 104.6 | **102.2** | 109.7 | 110.0 |

By regime (median per-run TIMSE ratio):

| Regime | Runs | Hybrid / baseline | Hybrid beats baseline | Hybrid / linear | Hybrid beats linear |
|---|---|---|---|---|---|
| CME | 295 | 0.86 | 69% | 0.99 | 53% |
| possible CME | 81 | 0.72 | 78% | 0.92 | 65% |
| stream arrival | 304 | 1.01 | 49% | **0.97** | **59%** |
| fast stream | 105 | 0.90 | 56% | 0.98 | 57% |
| slow wind | 186 | 0.65 | 77% | 0.97 | 57% |
| intermediate | 477 | 0.82 | 64% | 1.04 | 43% |

**Caveat:** the eval year was scored for both the selected and the tamed hybrid. The taming was
chosen on the training folds and leaf support, but it was prompted by the eval analysis, so treat
the tamed gain as slightly optimistic.

## Analysis (`src/experiments/xgboost_model.ipynb`)

- **Where the gain is:** it sits in the upper half of the run distribution (p90 to p99, 4–7% below the linear model). Typical runs are within ±4%, and the median is 4% worse.
- **Which runs gain:** CME runs and stream arrivals carry the gain (−85% and −25% of the change against the linear model). Intermediate wind is the one regime that loses on typical runs (43% won).
- **How the gain is spread:** more broadly than before taming. The model wins more runs than it loses in 7 of 13 months. February 2026 now costs 1%, against 50% untamed.
- **What the trees learned:** run-level offsets, never lead-dependent (no split on the lead). On average +13 km/s in CME runs, +8 to +9 in arrivals and fast streams, about 0 in slow wind.
- **Most useful inputs** (ablation):
  - EPAM 310 keV, Bt and the 13.5-day density lag;
  - the 27-day polarity lag and the coronal-hole strips trade a better median for a worse pooled TIMSE.
- **Open problems:**
  - **Runs without a coronal-hole image** get the largest corrections (+18.5 km/s at 48–72 h). The trees likely learned "no image" from the active Nov 2024 – Jan 2025 archive gap. Those runs tie the linear model.
  - **The large-coronal-hole over-forecast:** it wins only 46% of runs in the largest quarter.
  - **Stream timing beyond 2 days.**

## Live

- **`SWP_MODEL=xgboost`** makes `model.py` use `inference.run_xgboost_forecast`. The speed history and all 27 input regressors are fetched in parallel (`inference.regressor_features`).
- **In the image** (4 vCPU / 16 GiB, t0 2026-10-22 06Z): the tamed hybrid observed 27 of 27 regressors and finished in 15.1 s with a valid forecast. No xgboost is installed in the image.
- **A missing feed:** a regressor never missing in training takes its training median, so a flare-feed outage can't trigger an untrained branch.

## Reproduce

```bash
python -m training.xgboost_model select --base linear --sample-frac 0.1 --seed 0   # pilot
python -m training.xgboost_model select --base linear --max-seconds 480            # full; resumable
python -m training.xgboost_model tame                                              # leaf-support variants
python -m training.xgboost_model train --base linear                               # -> src/assets/xgboost_model.json
python -m training.xgboost_model evaluate                                          # eval year
```

- **Config:** the tamed settings are in `src/training/xgboost_hybrid_config.json`, with the taming record under `selection.taming`.
- **XGBoost alone:** without `--base linear` the same commands run XGBoost alone, the comparison above. Its results are in `src/experiments/xgboost_selection.csv` and `src/training/xgboost_model_config.json`.

## Nine years of history

**Setup:**
- **History:** the hybrid was rebuilt on the training history extended to 2016-10 → 2025-10.
- **Folds:** six yearly walk-forward folds, 2019-10 → 2025-10 (`training.folds.SELECTION`).
- **Anchor:** the re-selected 9-year linear model.

**Selection** (pilot on 10% of the hourly runs, then the full run on 78,888 hourly runs):

| Setting | Fold score |
|---|---|
| Set C (30 regressors), depth 4, min child weight 100, 100 trees | 0.4247 |
| Set C, default settings | 0.4253 |
| Set B (24) / set A (the linear model's 4) | 0.4332 / 0.4390 |
| 9-year linear model alone | 0.4387 |

- **Set D left out:** set D (all 120 regressors) was not run; it lost in the 4-year selection and in this pilot, and it would have needed 3–4 GB.
- **Memory:** the run fits on this 7.7 GB laptop with `XGB_NTHREAD=2` and a linear anchor computed in chunks of 4,000 runs, peaking at 640 MB.

**Taming** (`tame`, resumable, six folds):
- **The problem:** the selected setting had 644 leaves reached on fewer than 30 training days, one of them on a single day, moving some runs by up to 297 km/s.
- **The choice, by the same rule:** min child weight 10,000, depth 4, 100 trees. Fold score 0.4258, giving up 0.3%; every leaf is reached on at least 43 days.
- **Against the 9-year linear model** on the folds: −2.9%.

**Eval year** (`src/experiments/xgboost_model.ipynb`):

| | 9-year hybrid | 4-year hybrid | 9-year linear | Final baseline |
|---|---|---|---|---|
| Pooled TIMSE | 4.73e7 | **4.36e7** | 4.82e7 | 5.58e7 |
| Per-run mean | 2.22e6 | **2.22e6** | 2.34e6 | 2.59e6 |
| Per-run median | **6.45e5** | 7.04e5 | 6.67e5 | 8.45e5 |
| Runs won against its linear anchor | **62%** | 52% | | |

- **Typical runs improve:** the trees now help them (p10 to the median 3–11% below the linear model), and they win against it in every regime except stream arrivals (level). Big stream arrivals improve most (pooled −21%).
- **The corrections now depend on the lead,** which is split on 42 times.
- **No leaf rests on one episode.**
- **Remaining weaknesses:**
  - downward corrections on the eve of the big CMEs (18 Jan 2026, 10 Nov 2025);
  - the unchanged 72 h over-forecast of quiet wind;
  - pooled TIMSE, where the few largest CME runs favoured the 4-year hybrid.

```bash
XGB_NTHREAD=2 python -m training.xgboost_model select --base linear --sets A B C --max-seconds 240
XGB_NTHREAD=2 python -m training.xgboost_model tame --max-seconds 240
XGB_NTHREAD=2 python -m training.xgboost_model train --base linear
```
