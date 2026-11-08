# PFSS stream-arrival switch (aborted, not built)

**The idea.** Use a stream-arrival indicator at t0 to switch between two models:
- a **slow-wind model:** persistence, or the submitted config fitted on slow-wind runs only;
- a **stream model:** the submitted pfss-dhcs linear model.

**Why try it:** runs that stay slow are 1.235 of persistence with the submitted model ([model_diagnostics.md](model_diagnostics.md)). The PFSS map at t0 also covers the longitudes that will face Earth after t0, so it might tell slow wind that stays slow from slow wind about to be hit by a stream, which state terms couldn't ([slow_wind.md](slow_wind.md)).
- **Study:** `python -m training.pfss_switch` (`--pilot --folds 2024`).
- **Analysis:** `src/experiments/pfss_switch.ipynb`; results in `src/experiments/pfss_switch.csv` and `pfss_switch_classifier.csv`.

**Result: aborted after the one-fold pilot (2024), not built (user decision, 2026-11-08).**
- **No realistic switch beats the submitted model.**
- **PFSS doesn't improve the indicator** over in-situ inputs.
- **Even a perfect switch** gains only 3.3–3.9% on that fold.

## Method

- **Indicator:** P(the run stays slow) at t0.
  - **Label:** maximum observed speed below 450 km/s over the 72 target hours (`training.solar_wind_events`).
  - **Model:** a logistic regression (standardised, L2 with C = 1, not tuned), per fold on the fit runs whose target hours end before the fold.
  - **Feature sets:**
    - **in-situ:** last speed, 24-h speed trend, Bt, density, coronal-hole strip;
    - **in-situ + PFSS:** the above plus five ring signals at lags −72…+72 h and the open area (36 columns).
- **Slow-wind models:**
  - official persistence;
  - the submitted config fitted on the slow-wind fit runs only (added after the first pilot, at the user's request).
- **Switches:**
  - hard: the slow model where P > τ, with τ from 0.5 to 0.9, or τ chosen on the year before the fold;
  - soft: f + P^γ (slow − f), with γ = 1 or 2;
  - shrunk: f + α P (slow − f), with α chosen on the year before.
- **References:**
  - the submitted model, refitted per fold;
  - the oracle: the slow model on the true slow runs.
- **Decision rule, set before running:** more than 1% better than the submitted model on at least 4 of 6 folds.

## Results (2024 fold)

**Indicator:**

| | In-situ | In-situ + PFSS |
|---|---|---|
| AUC | **0.862** | 0.831 |
| Log-loss skill vs climatology | 0.387 | 0.355 |
| Slow runs given P > 0.5 | 29% | 21% |
| Stream arrivals given P > 0.5 | 5.7% | 0.9% |

**Switches** (% change in S_final against the submitted model; negative = better):

| | Persistence as the slow model | pfss-dhcs fitted on slow runs |
|---|---|---|
| Oracle (true slow runs) | −3.3% | −3.9% |
| Oracle, slow runs only | −53% | −64% |
| Slow model on every run | +28% | +33% (stream arrivals +64%, CMEs +46%) |
| Best fixed hard switch (P > 0.6, PFSS) | −0.01% | −0.03% |
| Hard, τ chosen on the year before | 0.00% (never switches) | 0.00% (never switches) |
| Soft and shrunk | +0.04% to +2.2% | +0.04% to +0.9% |

## Why it fails

- **The ceiling is low.** Slow runs have small errors in absolute terms, so even fixing them perfectly moves the official score by under 4%. Sending intermediate runs to persistence as well makes the oracle worse (+0.5% over six folds, measured from the diagnostics cache).
- **A wrong switch costs more than a right one gains.** A model for slow wind (persistence, or one fitted on runs where nothing arrived) under-forecasts every arrival or CME it is handed. Fitted on the year before, the threshold backs off to never switching.
- **PFSS doesn't sharpen the indicator.**
  - The last speed carries most of the skill.
  - The 36 PFSS columns lower the AUC slightly.
  - On their own, the forward longitudes separate slow runs from arrivals only moderately: AUC 0.66–0.69 for the current-sheet distance among runs that start slow.
  - The submitted model already uses that signal through `pfss_d_hcs_lead_72h`.
- **The earlier routing study found the same** before PFSS ([regime_models.md](regime_models.md)): the oracle's gain is mostly the label leaking the future, and the realistic routing gained 1.0% on 3 of 6 folds.
