# Fallbacks for a missing coronal-hole strip (not built)

**The question.** When the coronal-hole input is missing, is there a better fallback than the current one? The performance study ([model_diagnostics.md](model_diagnostics.md)) found 20% of eval-year runs and 4% of fold runs without the input (`ch_cm_lead_72h`, the strip from the AIA image about 72 h before each forecast hour). Those runs go to the submodel fitted without it (`Config.optional`), and they are worse: RMSE 102 against 81 km/s on the eval year.
- **Study:** `python -m training.ch_fallback` (`--pilot --folds 2024`; about 1 min).
- **Analysis:** `src/experiments/ch_fallback.ipynb`; results in `src/experiments/ch_fallback.csv`.

**Result: the submodel stays.** No fallback improves the affected runs consistently, and every one makes the six-fold score slightly worse (+0.04% to +0.35%).

## Where the gaps come from

- **The SDO browse archive has holes.** The day directories at `sdo.gsfc.nasa.gov/assets/img/browse` exist but hold no AIA 193 images for:
  - **all of** 2024-12, 2025-10 and 2025-11;
  - **part of** 2019-08, 2022-06, 2025-01, 2025-09, 2025-12 and 2026-03 (checked 2026-11-08).

  This is the live source too, so the history can't be refilled from it, and a re-pull would change nothing.
- **Live, the strip has been present** in all 52 staging cycles since 2026-10-26 (staging records, `sources.sdo` and `missing_regressors`). The risk is a future outage like the one in Oct–Nov 2025, which would put every run on the submodel for weeks.
- **The connector already reaches back.** It takes the newest image up to 24 h before each cut-off (`connectors.sdo.MAX_AGE`), and the windows fall back 6–12 h more (`windows.CH_FALLBACK_HOURS`). Short gaps are covered; only multi-day holes reach the submodel.

## Method

- **Variants.** Each fills the strip's missing lagged values, in fitting and forecasting, before the model sees them. Anything still missing goes to the submodel as before.
  - **reference:** as submitted, with images up to 12 h older.
  - **older 48 h:** images up to 48 h older than the lag.
  - **rotation:** the strip one solar rotation (27 days) earlier at the same lag. Coronal holes often recur.
  - **pfss proxy:** the strip estimated from the PFSS ring at the same Earth-facing longitude. This is a ridge regression on six ring signals and the open area, fitted on the fit runs.
  - **combined:** older + rotation, and older + rotation + proxy, applied in that order.
- **Folds:** the six yearly folds, with the submitted config refitted on the years before each. The reference reproduces 0.7794.
- **Score:** official S_final relative to persistence, over all runs and over the *affected* runs (those the reference sends to the submodel).
- **Decision rule, set before running:**
  - **Why the usual rule can't apply:** the affected runs fall in a few folds only, so 4 of 6 can't be met.
  - **Rule used:** a variant goes forward if it improves the affected runs by more than 2% in every fold that has them (at least 3), without making the six-fold score worse.
  - **The eval year is not used.**

## Results

| Variant | / persistence | vs reference | Affected runs vs reference (2021 fold, 19 runs / 2024 fold, 312 runs) | Runs still on the submodel |
|---|---|---|---|---|
| reference | 0.7794 | — | — | 331 |
| older 48 h | 0.7797 | +0.04% | −11.1% / −1.3% | 313 |
| rotation | 0.7797 | +0.05% | −27.1% / +5.2% | 96 |
| pfss proxy | 0.7804 | +0.13% | −9.5% / +1.3% | 0 |
| older + rotation | 0.7814 | +0.26% | −23.0% / +4.5% | 96 |
| older + rotation + proxy | 0.7821 | +0.35% | −23.0% / +5.7% | 0 |

- **Affected runs are almost all in the 2024 fold** (312 of 331). They are the Dec 2024 and Jan 2025 holes, plus the Sep–Oct 2025 start of the long one. The 2021 fold's 19 runs (a June 2022 hole) gain a lot, but on too few runs to count.
- **On the 2024 fold, filling the strip with a stand-in is worse than the submodel.**
  - **The strip from a rotation earlier** loses 5%: coronal holes evolve over a month, especially near solar maximum.
  - **The PFSS proxy explains only 15–16% of the strip's variance,** so it is mostly the mean with noise.
  - **Older images** help a little (−1.3%), but most of the gap is longer than 48 h.
- **The overall score gets slightly worse with every variant.** Filling also changes the fit, by moving a few hundred runs from the submodel's fit into the full model's.

## Implications

- **Keep the submodel** as the fallback for a missing strip. It is the best of the options tested.
- **A long SDO outage would cost** roughly the eval-year gap: RMSE about 100 against 80 km/s on affected runs. Nothing tested here recovers that.
- **What would help is another live image source** for the same strip: GOES SUVI 195 Å from `services.swpc.noaa.gov` (already a declared domain), or Helioviewer's AIA. That needs the strip recalibrated to the other instrument, and SUVI has no history before 2017–2018 in this format. It is not done in this cycle.
