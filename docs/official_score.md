# The official score, and the shipped models re-scored with it

**What it is.** On 2026-10-25 the organisers released the competition metric, `docs/timse_implementation.pdf`. It is kept locally and not committed, like `docs/Briefing.pdf`. The metric differs from the TIMSE the models were chosen on ([timse.md](timse.md)).
- implementation: `src/training/competition_score.py`, with the spec's invariants in `tests/test_competition_score.py`;
- official truth: `src/training/pull_ace_1h.py`;
- re-scoring study: `src/training/official_rescore.py`; analysis: `src/experiments/official_rescore.ipynb`.

**Result.**
- **Hybrid against linear:** on the official metric the hybrid is 0.14% *worse* than the linear model (3 of 6 folds); on the old metric it was 1.5% better.
- **Both against the final baseline:** both still beat it by about 6%, and persistence by about 16%.
- **Bias:** all three models carry a large upward bias, +55 km/s on average, rising to +95 km/s at 72 h. The old metric rewarded it; the official one charges it through RMSE, where every model is worse than persistence.

## The metric

For each lead τ = 1…72, over every run's valid (t0, τ) pair:
- **RMSE_τ.**
- **Percentile TIMSE_τ:**
  - thresholds: the 101 empirical quantiles q_k of that lead's observations (k = 0…100);
  - M_k: the MSE over the pairs with v ≥ q_k;
  - TIMSE_τ = Σ_k (q_{k+1} − q_k) M_k / (v_max − v_min). The weights are velocity widths, and the result is in km²/s².
- **S_τ = √(RMSE_τ · √TIMSE_τ)**, a geometric mean of RMSE and tail RMSE, in km/s.
- **S_final = the mean of S_τ over the 72 leads.** The leads are equally weighted, never pooled, and not scored per run.

**What changes from the old metric** ([timse.md](timse.md)):
- thresholds were all the sorted observations, now 100 percentiles;
- the old metric was unnormalised, the new one is divided by the speed range;
- the old one pooled runs (or scored them one by one), the new one is per lead;
- the old one had no RMSE term; now RMSE counts as much as the tail.

**Missing values:**
- a pair with missing truth contributes nothing;
- a missing or failed forecast is **replaced by persistence**, never dropped.

**Diagnostics** (spec §9): mean RMSE, mean TRMSE, bias, the S_τ/RMSE_τ/TRMSE_τ curves, and S_49:72 (long leads).

**Conventions the spec says must be fixed** (constants in `competition_score.py`):
- **Quantiles:** numpy's "linear" interpolation, computed with integer positions. `np.quantile` works in floating point and can put a quantile a hair above an observation that it should equal (70.00000000000001), and the spec's v ≥ q_k then drops that observation.
- **Time:** UTC; hours are labelled by their start, as in the official file; a duplicate timestamp keeps the last row.
- **Persistence:** the latest valid hour *available* at t0, i.e. labelled ≤ t0 − 1 h. The spec's remark that it "reduces to v(t0)" would also allow the hour labelled t0, which is published after t0. Both are scored below; the literal reading is 1.5% better.
- **Climatology:** v_clim isn't released yet. The mean of the truth before each fold start stands in for it.

## Ground truth

**Source:** the official file is SWPC's hourly `YYYYMM_ace_swepam_1h.txt` (Bulk Speed) on `sohoftp.nascom.nasa.gov/sdb/goes/ace/monthly/`, updated hourly. `python -m training.pull_ace_1h` writes `src/data/ace_1h/speed_hourly.csv`, which takes 40 s.

**Against the training truth** (the 1-minute daily files averaged by hour), 2016-10 → 2026-10:
- **Where both have a value they agree:** median |difference| 0.25 km/s, 94% within 5 km/s, and the same hour labels (the mean |difference| is 1.2 km/s at lag 0 and 7.7–8.0 km/s at ±1 h).
- **But the official file has far fewer valid hours: 70% against 91%.**

  | Status | Share of hours | Meaning |
  |---|---|---|
  | 0 | 70.5% | valid |
  | 1 | 9.9% | "bad data record". The speed is still given and matches ours (median difference 1.2 km/s), but runs faster (452 km/s against 423 km/s on status 0). |
  | 9 | 17.1% | "no data". Mostly isolated single hours (median block 1 h); our 1-minute data has a value for half of them. |
  | 3, 4, 6 | 2.5% | other bad-data codes |

  The spec counts every status other than 0 as missing. The official evaluation therefore drops 30% of hours, slightly more of them fast wind.
- **Sensitivity:** the study also scores against "any status" (every non-fill speed in the official file) and against the training truth. The conclusions don't change.

## Method

- **Models:** the final baseline, the linear model and the hybrid, fitted on 2016-10 → fold start exactly as in `training.compare_history`. Their inputs are unchanged; only the scoring is new.
- **Spec baselines:**
  - persistence;
  - persistence (t0), the literal reading;
  - 27-day recurrence, with persistence where that hour is missing;
  - climatology (provisional).
- **Folds:** the six yearly walk-forward folds (2019-10 → 2025-10), 1,460 runs each. The eval year is not used.
- **Summary:** the mean S_final, and S_final / persistence's as a geometric mean over folds. The keep rule is the usual one: more than 1% better, winning at least 4 of 6 folds.
- **Run:** `python -m training.official_rescore`, about 2 min (pilot: one fold, 30 s, 722 MB).

## Results (official truth)

| Forecast | S_final (km/s) | / persistence | RMSE | TRMSE | Bias | S_49:72 |
|---|---|---|---|---|---|---|
| **linear** | **93.7** | **0.842** | 85.8 | 104.3 | +56.1 | 115.6 |
| hybrid (shipped) | 94.0 | 0.843 | 87.1 | 102.9 | +57.3 | 114.5 |
| final baseline | 99.8 | 0.899 | 85.7 | 118.7 | +52.5 | 121.9 |
| persistence (t0) | 109.0 | 0.985 | 78.5 | 152.9 | +1.8 | 143.3 |
| persistence | 110.6 | 1 | 79.7 | 154.9 | +1.9 | 143.9 |
| 27-day recurrence | 129.0 | 1.163 | 95.6 | 176.7 | −0.8 | 138.3 |
| climatology | 138.6 | 1.249 | 90.7 | 214.4 | −6.1 | 138.6 |

**Pairs, official against old metric:**

| Pair | Official change | Folds won | Old-metric change |
|---|---|---|---|
| hybrid vs linear | +0.14% | 3 | −1.54% |
| hybrid vs final baseline | −6.2% | 5 | −13.5% |
| linear vs final baseline | −6.4% | 6 | −12.1% |

- **The hybrid's edge is gone.**
  - **By year:** the hybrid wins 2019-10, 2020-10 and 2023-10 (by 2.9%, 1.2% and 0.6%), and loses 2021-10, 2022-10 and 2024-10 (by 2.7%, 0.8% and 2.2%).
  - **At short leads it is clearly worse:** S at 1 h is 27.7 km/s against 23.3 for the linear model, with RMSE 25.2 against 19.3. The trees, pooled over leads, add error where the speed persists.
  - **At long leads it is slightly better:** S_49:72 is 114.5 against 115.6.
- **The bias is the largest problem.** Every model is biased upward, by about +3 km/s at 1 h, +46–50 km/s at 24 h and +90–95 km/s at 72 h. This is the "upward offset that TIMSE weighting makes optimal" ([quantile_mapping.md](quantile_mapping.md)): the old metric's training weights favour over-forecasting, since missing a fast stream costs more than over-forecasting slow wind.
  - **Under the official metric it splits:** it helps the tail term (TRMSE 103–119 against persistence's 155), but RMSE counts as much. There every model (85.7–87.1) is worse than persistence (79.7).
- **The other baselines:** recurrence and climatology are far behind persistence on this metric. Recurrence's tail term is good, but its RMSE isn't.
- **The truth doesn't change the picture.** With "any status" or the training truth, the order and the ratios are within 0.5% of the official ones (linear 0.840/0.841, hybrid 0.840/0.842, final baseline 0.897/0.900).

## Decision

- **Done 2026-10-26** ([official_retune.md](official_retune.md)): re-tuned for the official metric, the linear model (blend 0.25) is 5.9% better and ships; the final baseline uses tail_power 0.5.
- **Nothing changes yet.** The shipped hybrid stays until the re-tuning study has run.
- **The models were tuned for the wrong metric.** Re-tuning them for the official one, mainly reducing the bias, is the next study, run on these folds with `competition_score`:
  1. a per-lead bias correction learned on earlier years;
  2. refitting the linear model and the hybrid with training weights that match the official metric's mix of RMSE and tail error.
- **Hybrid against linear:** the re-tuning study also settles which ships in November. On the current tuning the hybrid no longer clears the 1% / 4-of-6 bar over the linear model.
- **Future studies:** all are scored with `training.competition_score`; the old per-run ratio is kept only for comparison.
