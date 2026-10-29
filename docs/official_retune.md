# Re-tuning for the official metric (built: the linear model ships)

**What it is.** The follow-up to [official_score.md](official_score.md). The models were tuned on the briefing's per-run TIMSE, whose weights favour over-forecasting, so they carried a +55 km/s bias that the official metric charges through RMSE. This study re-tunes the loss weights for the official metric.
- **Code:**
  - new weights: `models.final_baseline.official_weights` and `lead_timse_weights`, available as `weights="official"` and `blend` in the linear model's and the hybrid's `Config`;
  - study: `src/training/official_retune.py`;
  - analysis: `src/experiments/official_retune.ipynb`.

**Result: built.**
- **Linear model:** official weights at blend 0.25 make it 5.9% better on the official score, winning 6 of 6 folds.
- **Final baseline:** `tail_power` 0.5 makes it 3.7% better, also 6 of 6.
- **What ships:** from 2026-10-26 the **re-tuned linear model is the submitted model**. It replaces the hybrid, which re-tuned scores the same (88.5 km/s), and is simpler.

## The weights

The official score per lead is S_τ = √(RMSE_τ · √TIMSE_τ). Both parts are weighted squared errors over the same pairs:
- **RMSE:** uniform weights.
- **Percentile TIMSE:** each observation gets w_i = Σ over q_k ≤ v_i of (q_{k+1} − q_k) / (N_k · span), over all runs at that lead. `lead_timse_weights` reproduces the official TIMSE exactly; the test is in `tests/test_competition_score.py`.

`official_weights(y, blend)` mixes the two parts: (1 − blend) + blend × t, where t is the TIMSE weights scaled to mean 1 per lead. The blend that minimises S isn't known in advance, so it is chosen on the folds.

The old weights (`run_timse_weights`) score each run on its own thresholds. Within every run the fastest hours get the most weight, which pulls every forecast up.

## Method

- **Folds:** the six yearly walk-forward folds (2019-10 → 2025-10), fitted on 2016-10 → fold start as in `training.compare_history`. Features and every other setting stay as shipped.
- **Score:** the official S_final against the official truth (`training.competition_score`, `training.pull_ace_1h`). The summary is the geometric mean over folds of S_final / the shipped model's.
- **Variants:**
  - `shipped`;
  - `shipped − bias`: subtract the per-lead mean in-sample error on the fit runs; nothing refitted;
  - `official b=0 … 1`: both models refitted with official weights, the hybrid's trees on the linear model of the same blend;
  - final baseline `tp=1 … 0`: its loss weights raised to `tail_power`.
- **Keep rule, set before running:** more than 1% better than shipped, winning at least 4 of 6 folds.
- **Run:** `python -m training.official_retune`, about 7 min (pilot: one fold, 88 s, 1 GB). Blend 0.1 was added after the pilot, since the best value lay between 0 and 0.25.

## Results (six folds, official truth; S in km/s)

| Model | Variant | S_final | vs shipped | Folds won | / persistence | RMSE | TRMSE | Bias (72 h) |
|---|---|---|---|---|---|---|---|---|
| linear | shipped | 93.7 | | | 0.842 | 85.8 | 104.3 | +56 (+95) |
| linear | shipped − bias | 91.5 | −2.4% | 4 | 0.822 | 64.4 | 131.6 | +7 (+6) |
| linear | b = 0 | 91.1 | −2.8% | 4 | 0.818 | 62.6 | 134.3 | +1 (+1) |
| linear | b = 0.1 | 89.0 | −5.1% | 5 | 0.799 | 64.3 | 124.9 | +10 (+13) |
| **linear** | **b = 0.25** | **88.5** | **−5.9%** | **6** | **0.793** | 68.3 | 116.2 | +22 (+30) |
| linear | b = 0.5 | 90.2 | −4.2% | 5 | 0.807 | 77.3 | 106.9 | +41 (+55) |
| linear | b = 1 | 103.4 | +10.5% | 0 | 0.930 | 111.1 | 97.7 | +89 (+117) |
| hybrid | shipped | 94.0 | | | 0.843 | 87.1 | 102.9 | +57 (+92) |
| hybrid | b = 0 | 89.6 | −4.6% | 5 | 0.805 | 63.2 | 128.8 | +5 (+5) |
| hybrid | **b = 0.1** | **88.5** | **−5.9%** | **6** | 0.794 | 65.5 | 121.1 | +13 (+16) |
| hybrid | b = 0.25 | 89.2 | −5.3% | 6 | 0.799 | 69.4 | 116.0 | +21 (+27) |
| final baseline | tp = 1 (shipped) | 99.8 | | | 0.899 | 85.7 | 118.7 | +52 |
| **final baseline** | **tp = 0.5** | **96.2** | **−3.7%** | **6** | 0.866 | 70.9 | 132.7 | +24 |
| final baseline | tp = 0.25 | 96.3 | −3.6% | 6 | 0.866 | 67.2 | 139.8 | +12 |
| final baseline | tp = 0 | 97.9 | −2.0% | 3 | 0.881 | 65.8 | 147.5 | +1 |

- **The bias was the main loss, but not all of it.**
  - Removing the bias as a post-step recovers 2.4%.
  - Refitting with official weights recovers 5.9%, because the fit balances the two parts of the score instead of only shifting the forecast.
- **Some upward lean stays optimal.** The best blends keep +10 to +30 km/s of bias at long leads: the TIMSE half of the score still rewards erring towards fast wind. Blend 1 (the TIMSE term alone) is the worst, since it optimises half the score and gives up the other half.
- **Hybrid against linear: a tie.**
  - Each at its best blend scores 88.5 km/s.
  - At the same blend the hybrid is 0.6% better at 0.1 (4 of 6 folds) and 0.8% worse at 0.25 (3 of 6).
  - The trees don't clear the 1% / 4-of-6 bar over the linear model, so the simpler model ships. It was already the hybrid's anchor and first fallback.

## The eval year (used once, after choosing)

`python -m training.official_retune --eval-year` scores the assets on 2025-10-20 → 2026-10-16 (1,448 runs; 69% of pairs have official truth). Results are in `src/experiments/official_eval_year.csv`.

| Forecast | S_final | / persistence | RMSE | TRMSE | Bias (72 h) |
|---|---|---|---|---|---|
| hybrid (previously shipped, old weights) | 122.7 | 0.780 | 97.8 | 156.9 | +59 (+106) |
| **linear (new, b = 0.25)** | **123.7** | **0.787** | 87.6 | 177.4 | +34 (+50) |
| linear (previous) | 125.8 | 0.800 | 98.2 | 164.9 | +59 (+107) |
| final baseline (new, tp = 0.5) | 131.6 | 0.837 | 87.6 | 199.5 | +28 (+49) |
| final baseline (previous) | 136.8 | 0.870 | 107.6 | 177.1 | +66 (+107) |
| persistence | 157.3 | 1 | 105.1 | 238.4 | +1 |
| 27-day recurrence | 165.1 | 1.050 | 115.6 | 236.6 | +5 |

- **Re-tuning holds on the eval year:**
  - the linear model is 1.7% better than its previous version;
  - the final baseline is 3.8% better;
  - both have about half the bias.
- **The previously shipped hybrid scores 0.8% better than the new linear model this year.** The eval year was the fastest and most variable year: mean 466 km/s, sd 110, 9% of hours above 600 km/s, against 378–462 km/s and 1–7% in the fold years. The old hybrid's larger upward bias helped the tail term there (TRMSE 156.9 against 177.4).
  - This is one year, scored after the choice, and choices are never made on it ([evaluation.md](evaluation.md)).
  - On the six folds the old hybrid is 5.9% *worse* than the new linear model.
  - A re-tuned hybrid wasn't trained on all years, so it isn't in this table.

## Decision

- **Built:**
  - the linear model with `weights="official"`, `blend=0.25` (`src/training/linear_model_config.json`, `src/assets/linear_model.json`) is the submitted model: `inference.run_selected_forecast` runs it by default;
  - the final baseline uses `tail_power=0.5` (`src/training/final_baseline_config.json`, `src/assets/final_baseline.json`).
- **Fallback chain:** linear model → final baseline → persistence → training mean.
- **The hybrid** is kept, with its old weights and its own embedded linear anchor, behind `SWP_MODEL=hybrid`. Re-tuned, it ties with the linear model, so a candidate for later is a hybrid at blend 0.1 on the new linear anchor, compared in the staging environment.
- **Not changed:** the features and every other hyper-parameter were selected on the old metric. *Re-selected on the official metric 2026-10-29:* the same variant and regressor families come out, and the only difference (a 1 h density window, −0.31%, 4 of 6 folds) is below the bar, so nothing changed ([official_reselection.md](official_reselection.md)).
