# Slow-wind state bias and high-speed decay (not built)

**The question.** Can state-dependent terms fix the two biases the performance study found ([model_diagnostics.md](model_diagnostics.md))?
- **Slow-wind over-forecast:** runs that stay slow are 1.235 of persistence, and slow hours are 41 km/s too high.
- **Slow decay:** high forecasts relax too slowly after fast wind (−64 km/s at a last speed of 700–750 km/s).

This covers items (i a), (i c) and (ii) of the follow-up plan.
- **Model support:** optional `Config.terms` in `models.linear_model`. These are hinge, gate, interaction and square columns built from the last speed and the standardised regressors, so they need nothing extra live. The shipped asset has none and is unchanged.
- **Study:** `python -m training.slow_wind` (`--pilot --folds 2024`; about 2 min; `--variants` re-runs a subset).
- **Analysis:** `src/experiments/slow_wind.ipynb`; results in `src/experiments/slow_wind.csv`.

**Result: nothing passes the 1% / 4-of-6 rule.**
- **Best:** a tighter regressor cap (5% instead of 1%), 0.83% better on 5 of 6 folds. It goes forward to the combination step as a near-miss.
- **Slow wind:** no state term reduces the over-forecast.
- **Loss-weight blend:** it trades slow-wind accuracy against active-wind accuracy one for one.

## Method

- **Reference:** the submitted config (with `pfss_d_hcs_lead_72h`), refitted per fold with the data fixes. It reproduces 0.7794.
- **Variants,** each added to the reference:
  - **(i a) state bias:** `hinge:last<450` = max(0, 450 − last speed); both hinges (plus `hinge:last>550`); a slow indicator 1[last < 450].
  - **(i c) quiet wind:**
    - the current-sheet or X-ray term gets its own slope when last < 450 (`gate:<x>:last<450`), or both do;
    - X-ray dropped;
    - the regressor cap at 5% (`cap_quantile` 0.05, against 0.01).
  - **(ii) decay:** `hinge:last>550`; the speed trend over the last 6 or 24 h (`speed_slope_6h/24h`, new window regressors); the 24-h trend × last speed.
  - **Blend:** the loss weights' share of percentile-TIMSE weights at 0.1, 0.15, 0.2 and 0.35 (shipped: 0.25). The 0.1 and 0.2 points were added after the one-fold pilot.
- **Score:** official S_final relative to persistence, as a geometric mean over the six yearly folds, and per run regime.
- **Decision rule, set before running:**
  - **Keep:** more than 1% better on at least 4 of 6 folds, with slow runs not worse.
  - **Near-miss:** at least 0.3% better on at least 4 folds goes forward to the combination step.
  - **The eval year is not used.**

## Results (% change in S_final against the reference; negative = better)

| Variant | All | Folds won | CME | Stream arrival | Fast | Intermediate | Slow | Slow residual (km/s) |
|---|---|---|---|---|---|---|---|---|
| **cap 5%** | **−0.83** | **5** | −0.1 | −1.4 | −2.9 | −0.7 | +0.9 | −45 |
| speed trend 24 h × last | −0.12 | 4 | −0.2 | −0.1 | −1.0 | +1.5 | +1.4 | −47 |
| blend 0.2 | −0.08 | 1 | +0.3 | +1.5 | +2.0 | −1.9 | −3.8 | −43 |
| speed trend 24 h | −0.07 | 5 | −0.3 | −0.1 | −1.0 | +1.5 | +1.4 | −47 |
| blend 0.15 | 0.00 | 1 | +0.9 | +3.2 | +4.4 | −3.8 | −7.8 | −40 |
| reference | — | — | — | — | — | — | — | −47 |
| slow indicator | +0.02 | 3 | +0.2 | −0.2 | +0.2 | −0.6 | +1.3 | −48 |
| hinge last < 450 | +0.03 | 4 | +0.3 | −0.4 | −0.8 | −0.2 | +0.1 | −48 |
| X-ray gated in slow wind | +0.04 | 2 | +0.1 | +0.1 | +0.1 | 0.0 | 0.0 | −47 |
| speed trend 6 h | +0.04 | 3 | +0.1 | +0.1 | +0.3 | +0.1 | −0.1 | −46 |
| both hinges | +0.20 | 1 | +0.2 | −0.5 | +0.7 | +1.1 | 0.0 | −47 |
| X-ray dropped | +0.24 | 3 | +1.5 | +2.4 | +2.7 | −1.5 | −5.1 | −40 |
| hinge last > 550 | +0.26 | 1 | +0.1 | +0.1 | +0.7 | +0.9 | +0.3 | −46 |
| blend 0.1 | +0.31 | 1 | +1.7 | +5.2 | +7.7 | −5.9 | −11.9 | −36 |
| current sheet gated in slow wind | +0.41 | 4 | +0.2 | +0.1 | +0.7 | −1.2 | +3.7 | −47 |
| both gated | +0.41 | 3 | +0.3 | +0.2 | +0.7 | −1.1 | +3.2 | −47 |
| blend 0.35 | +0.54 | 1 | −0.1 | −2.4 | −3.0 | +3.5 | +7.7 | −54 |

**By fold, cap 5%:** −0.7% (2019 fold), −0.2%, −2.7%, −1.2%, +0.7%, −0.9%.

## Why the slow-wind over-forecast stays

- **At t0, slow wind that will stay slow looks like slow wind about to be hit by a stream.** The run regime is known only afterwards. A term that depends on the state at t0 (last speed below 450, the slow indicator) sees both kinds of run.
  - The fitted model already has one intercept per lead. A hinge can only move the average over both kinds, and that average is already fitted.
  - The slow runs' residual doesn't move (−47 to −48 km/s).
- **The upward lean is what the official metric pays for.**
  - **Lowering the blend** cuts the lean. Slow runs improve by up to 12% (blend 0.1) and intermediate runs by up to 6%. But CME, stream-arrival and fast runs lose 2–8%, so the six-fold score is flat from 0.15 to 0.25.
  - **Only the 2024 fold gains** from a lower blend (−2 to −2.7%). It is the fold with the largest over-forecast. The others prefer 0.25.
  - So the over-forecast of slow wind is largely the hedge for the arrivals the model can't foresee, and the retune's blend 0.25 already balances it.
- **The current-sheet and X-ray terms carry the hedge.**
  - Dropping X-ray helps slow (−5%) and intermediate runs but costs every active regime (+1.5 to +2.7%).
  - Giving the current-sheet term its own slope in slow wind hurts the slow runs themselves (+3.7%): fitted on slow runs, the slope learns the arrivals that follow, which helps the 2019–2020 folds and costs 3.2% on the 2022 fold.
- **Decay terms don't help.** The fast hinge and the 6-h trend change nothing. The 24-h trend helps fast runs (−1.0%) but costs slow and intermediate runs (+1.4%).
- **The tighter cap works through outliers.** Clipping each regressor to its 5–95% training range stops extreme values (flares, field spikes, current-sheet extremes) from being extrapolated linearly. It helps stream-arrival and fast runs most.

## What goes forward

**Combination step:** cap 5% (near-miss, −0.83% on 5 of 6 folds). Its slow runs get slightly worse (+0.9%), so on its own it would fail the slow-runs condition of the keep rule.

**Not tested here:** a slow-wind fix that needs information about whether a stream is coming, such as the coronal-hole strip's trend or an interaction of the strip with the current sheet. That is phase 4 ([model_review.md](model_review.md)).
