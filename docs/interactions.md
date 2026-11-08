# Interaction and nonlinear terms in the linear model (not built)

**The question.** Do couplings between the inputs, or curvature in them, improve the submitted linear model? This is item (iv) of the follow-up plan to [model_diagnostics.md](model_diagnostics.md).
- **Why the linear model:** the XGBoost hybrid's trees, which can find couplings themselves, didn't beat the linear model ([hybrid_pfss.md](hybrid_pfss.md)). So the couplings are added to the linear model as a few explicit terms.
- **The terms** are `Config.terms` in `models.linear_model`, built from the last speed and the standardised regressors, so nothing extra is needed live ([slow_wind.md](slow_wind.md)).
- **Study:** `python -m training.interactions` (`--pilot --folds 2024`; about 3 min).
- **Analysis:** `src/experiments/interactions.ipynb`; results in `src/experiments/interactions.csv` and `interactions_path.csv`.

**Result: no term passes, so nothing is built.** The forward selection stops at step 1.
- **Best:** the current-sheet distance squared, 0.16% better on 3 of 6 folds. That is below the step rule (at least 0.3% on at least 4 folds).
- **The rest:** every other candidate changes the score by −0.02% to +0.40%.

## Method

- **Operands** (z is the standardised regressor at each lead; last = (last speed − training mean) / 100):
  - the coronal-hole strip (`ch_cm_lead_72h`);
  - the current-sheet distance (`pfss_d_hcs_lead_72h`);
  - field strength (`bt_mean_1h`);
  - X-ray flares (`xray_peak_max_72h`);
  - density (`density_mean_24h`);
  - the last speed.
- **Candidates:** all 15 pairwise products, all 6 squares, and one-sided versions max(0, z) of the coronal-hole and current-sheet inputs. 23 in all.
  - **How they were chosen:** the one-fold pilot used 13 candidates picked by hand, from the diagnostics' findings and physics priors (for example, fast wind needs an open coronal hole far from the current sheet).
  - **After the pilot,** all pairs and squares were tested instead, so the result doesn't depend on that choice (user decision, 2026-11-08).
- **Forward selection:**
  - each step adds the candidate with the best six-fold score;
  - a step is kept only if it gains at least 0.3% over the current set and wins at least 4 of 6 folds;
  - at most 3 terms.
  - Terms that use the optional coronal-hole strip are left out of its submodel.
- **Reference:** the submitted config refitted on the six yearly folds, 0.7794 of persistence.
- **Score:** official S_final relative to persistence, as a geometric mean over the folds.
- **Decision rule for the final set, set before running:** more than 1% better on at least 4 of 6 folds, with slow runs not worse. The eval year is not used.

## Results (step 1, % change in S_final against the reference; negative = better)

| Term | Six folds | Folds won | Note |
|---|---|---|---|
| current-sheet distance² | **−0.16** | 3 | −0.4, +0.1, −0.7, +0.4, +0.3, −0.7 by fold; slow runs −1.3% |
| coronal hole × density | −0.02 | 5 | at most ±0.2% in any fold |
| density × last, Bt × last, Bt × X-ray, X-ray × last | +0.05 to +0.06 | 1–4 | |
| coronal hole × X-ray, current-sheet distance one-sided, current-sheet distance × X-ray, density², X-ray × density, coronal hole × Bt | +0.11 to +0.14 | 0–5 | |
| last², Bt², current-sheet distance × Bt, X-ray², Bt × density | +0.16 to +0.24 | 1–2 | |
| coronal hole², coronal hole one-sided | +0.25, +0.26 | 1–2 | |
| coronal hole × current-sheet distance | +0.28 | 3 | −0.8 (2022 fold), −0.6 (2023), **+3.2 (2024)** |
| current-sheet distance × density | +0.29 | 0 | |
| current-sheet distance × last | +0.38 | 2 | |
| coronal hole × last | +0.40 | 1 | −0.4 on the 2024 fold (the pilot), +1.7 on the 2022 fold |

**The pilot misled again.** On the 2024 fold, the current-sheet distance squared was −0.70% and coronal hole × last speed −0.39%. Over six folds they are −0.16% and +0.40%.

## Why it doesn't help

- **The main relations are close to linear at this horizon.** The fit's per-lead coefficients already let each input's effect change with lead time. The squares and one-sided versions add at most a few tenths of a percent.
- **Products of correlated inputs are unstable.** The coronal-hole strip and the current-sheet distance both describe the same open-field pattern. Their product helps the 2022–2023 folds, then costs 3.2% on the 2024 fold. A likely reason (not checked) is that the solar-maximum maps of that fold move the product outside its training range.
- **The speed-dependent terms (× last speed) fight the persistence term.** The model is relative to persistence, and its last-speed coefficient already does the decay. A product with the last speed mostly re-weights that, and its sign changes from fold to fold.
- **This matches the trees.** The hybrid, which can find any coupling, ties with the linear model. The selection here finds no coupling worth 0.3% either.

## What goes forward

Nothing. The combination step keeps only the 5% regressor cap from [slow_wind.md](slow_wind.md).
