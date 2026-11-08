# Performance study of the submitted linear model ("pfss-dhcs")

**What it is.** A diagnostic study of the submitted model: the linear model with the PFSS current-sheet distance (`pfss_d_hcs_lead_72h`, since 2026-11-07; staging tag `upload_candidate/pfss-dhcs`). It covers:
- performance by wind regime;
- the residuals;
- the importance of each input.

Nothing is selected or changed from it.
- **Study:** `python -m training.model_diagnostics` (`--pilot --folds 2024` for the pilot, about 2 min for all).
- **Analysis:** `src/experiments/model_diagnostics.ipynb`. Tables in `src/experiments/model_diagnostics_{regimes,residuals,importance}.csv`.

## Setup

- **Forecasts:**
  - **Folds (the main analysis):** out-of-fold on the six yearly folds (2019-10 → 2025-10, 8,768 runs). The submitted config is refitted on the years before each fold. It reproduces its score exactly: 0.7794 of persistence, as a geometric mean over the folds.
  - **Eval year (a descriptive check):** the shipped asset through the live imputation (2025-10 → 2026-10, 1,448 runs). It scores **0.750** of persistence, against 0.787 for the linear model before the PFSS feature and 0.837 for the final baseline on the same year and truth (`src/experiments/official_eval_year.csv`). This is the eval year's only use; nothing is chosen from it.
- **Truth:** SWPC's hourly ACE file. **Residual = observed − forecast**, so negative means over-forecast. The official `bias` has the opposite sign.
- **Regimes** (`training.solar_wind_events`, thresholds fixed before looking at errors):
  - **Per run,** from the 72 target hours:
    - CME, including possible CMEs (cold ejecta in situ, their sheaths, shocks);
    - otherwise stream arrival;
    - fast stream (mean ≥ 550 km/s);
    - slow wind (maximum < 450 km/s);
    - intermediate.
  - **Per forecast hour:** a CME hour; otherwise slow < 450, intermediate 450–550 or fast ≥ 550 km/s observed.
- **Caveat:** the run regimes are defined from what happened, so they describe where the skill comes from, not how the model would do if the regime were known ahead.
  - "Slow wind" runs are, by construction, those where nothing arrived, which favours persistence.
  - "Stream arrival" runs are those where persistence must fail.
- **Importance, three views:**
  - **Refit without:** refit without each input, per fold.
  - **Shuffle:** shuffle each input across the fold's runs, 5 repeats, with no refit.
  - **Contribution:** coefficient × value of each design column, in km/s.
  - Changes are in official S_final against the full model; > 0 means the input helps.

## (i) Performance by regime

| Run regime | Runs (folds) | Linear / persistence | Final baseline / persistence | Beats persistence (run RMSE) | Eval year: runs, linear / persistence |
|---|---|---|---|---|---|
| all | 8,768 | 0.787 (pooled) | 0.879 | 55% | 1,448, 0.750 |
| CME | 1,903 (22%) | 0.809 | 0.894 | 57% | 376, 0.739 |
| stream arrival | 1,476 (17%) | **0.645** | 0.733 | 78% | 304, 0.692 |
| fast stream | 266 (3%) | 0.795 | 0.958 | 63% | 105, 0.780 |
| intermediate | 2,460 (28%) | 0.878 | **0.823** | 61% | 477, 0.761 |
| slow wind | 2,663 (30%) | **1.235** | 1.209 | 34% | 186, 1.223 |

- **The skill comes from stream arrivals, CMEs and fast wind.** On stream arrivals the model is 35% better than persistence.
- **Slow wind is the weak spot.** The model is 24% worse than persistence on runs that stay slow, the largest group (30% of runs). It is also over-forecast there, by 37 km/s on average.
  - **Getting worse over time:** 0.94 (2019 fold), 1.08, 1.20, 1.75, 1.23, 2.12 (2024 fold).
  - **Intermediate runs:** in the 2024 fold they also go above persistence (1.10).
- **The final baseline beats the linear model on intermediate and slow runs** (0.823 against 0.878, and 1.209 against 1.235). It loses everywhere else.

**By forecast hour** (folds pooled):

| Hourly regime | Share of hours | Share of squared error | Mean residual | RMSE |
|---|---|---|---|---|
| slow | 69% | 60% | **−41 km/s** | 68 |
| intermediate | 17% | 17% | −3 | 72 |
| fast | 8% | 14% | **+41** | 97 |
| CME | 6% | 10% | +12 | 92 |

- **Slow hours carry 60% of the squared error,** mostly as a steady over-forecast.
- **Fast and CME hours** have larger errors per hour, under-forecast.
- **The eval year** has more fast hours (18%), and their share of the error rises to 27%.

## (ii) Residuals

- **The bias grows with lead time.** The mean residual is −1.6 km/s at 1 h, −22 at 24 h, −32 at 48 h and −34 at 72 h. The RMSE is 19, 66, 84 and 89 km/s.
  - The median is more negative still (−37 km/s at 72 h), so this is a typical over-forecast, not a few outliers.
  - A possible cause (not tested here) is the loss weights: the 25% share of percentile-TIMSE weights stresses the upper speeds.
- **Distribution:**
  - right-skewed: under-forecasts of arrivals and CMEs form the long positive tail (skewness 0.4–1.0);
  - heavy-tailed at short leads (excess kurtosis 15 at 1 h, 1.4 at 72 h).
- **Calibration is good at low speeds and poor at high ones** (residual binned by the forecast):
  - forecasts of 300–350 km/s are unbiased (+3);
  - forecasts of 450–500 are too high by 35;
  - forecasts of 650–700 are too high by 90;
  - forecasts of 800–850 are too high by 173.
- **High speed is held too long.** Binned by the last observed speed, the residual goes from −6 km/s (250–300) to −64 (700–750).
  - This shows in the worst runs. In March–June 2025, after fast streams and CMEs (last speed 650–800), the model forecast up to 900 km/s while the wind decayed. Those runs were over-forecast by 200–260 km/s.
- **The largest under-forecasts are big CMEs.** The May 2024 storm runs (observed 963 km/s, forecast at most about 500) are under-forecast by 300 km/s. The model has no CME input that reacts in time.
- **Timing:**
  - **No systematic lead or lag.** For stream arrivals, the shift (±24 h) that minimises each run's RMSE has a median of −1 h (CME runs −4 h).
  - **The method doesn't separate timing from amplitude.** The best shift lowers the RMSE by about 20% in *every* regime, slow wind included, where 36% of runs end at the ±24 h bound. So with one free shift per run, timing errors can't be told apart from noise.
- **Errors persist from run to run:**
  - the run-mean residual correlates 0.90 with the next run, 0.54 at 24 h and 0.16 at 48 h;
  - about 0 at 72 h, and 0.11 one solar rotation (27 days) later;
  - within a run, lead 1 against lead 25 correlates only 0.22.

  The model's errors come in episodes lasting a day or two.
- **Drift:** the run-mean residual by fold is −10 km/s (2019), −20, −20, −27, −23, then **−51 (2024)**. The RMSE rises from 43 to 87 km/s.
  - The over-forecast grew through the solar maximum. On the eval year it is −29 km/s.
- **Input quality:**
  - **Submodel runs** (coronal-hole strip missing) are worse: RMSE 97 against 73 on the folds, 102 against 81 on the eval year.
  - **On the eval year, 20% of runs use the submodel**, against 4% on the folds: the strip is missing far more often in the eval year's AIA history (the cause was not investigated). Those runs cost part of the eval-year score.
  - **Runs with the PFSS value imputed** (8% on the folds) are not worse: RMSE 59.

## (iii) Feature importance

| Input | Refit without (folds; folds it helps) | Shuffled (folds; folds it helps) | Shuffled, eval year | Mean \|contribution\| |
|---|---|---|---|---|
| coronal-hole strip `ch_cm_lead_72h` | **+5.4%** (5) | **+9.7%** (6) | +7.6% | 21 km/s |
| current-sheet distance `pfss_d_hcs_lead_72h` | +1.7% (5) | +4.2% (6) | +6.1% | 15 km/s |
| recurrence 27 d | +2.1% (4) | — | — | 12 km/s |
| field strength `bt_mean_1h` | +1.1% (5) | +3.0% (6) | +2.2% | 8 km/s |
| density `density_mean_24h` | +0.45% (4) | +0.8% (5) | +1.6% | 5 km/s |
| X-ray flares `xray_peak_max_72h` | +0.24% (3) | +0.3% (5) | +1.2% | 10 km/s |
| recurrence 13.5 d | **−0.15%** (1) | — | — | 1 km/s |
| last speed (persistence term) | — (kept: the model is relative to it) | — | — | 32 km/s |

- **The coronal-hole strip is the most important input** by every measure.
  - Next come the current-sheet distance, the 27-day recurrence and the field strength.
  - The current-sheet distance is worth more on the eval year (+6.1% shuffled).
- **Weak inputs:**
  - **Density and X-ray flares** add under 0.5% each when refitted without.
  - **The X-ray term** gives a 10 km/s mean contribution but only +0.24% in score, so most of what it moves is noise.
  - **The 13.5-day recurrence term** is redundant: dropping it improves the score slightly (−0.15%, 5 of 6 folds), below the 1% rule.
- **By regime (refit without, % change):**
  - The current-sheet distance helps CME, stream-arrival and fast runs (+1.7 to +2.7%) but **hurts slow and intermediate runs** (−4.9%, −4.2%).
  - The X-ray term behaves the same way (slow runs −5.1%).
  - In quiet wind both push the forecast up, which feeds the slow-wind over-forecast.
  - The coronal-hole strip helps in every regime (CME +8.8%, slow +7.5%).
- **Coefficients by lead** (shipped asset, per standard deviation):
  - **Field strength** is 19 km/s at 24 h and 2 km/s at 72 h: it matters for the first day.
  - **Coronal-hole strip, current-sheet distance and X-ray** grow with lead, to 36, 24 and 24 km/s at 72 h.
  - **Last speed:** its coefficient falls to −0.67 by 72 h. The forecast relaxes two-thirds of the way from persistence to the training mean.

## What this suggests (hypotheses, not tested here)

- **The slow-wind over-forecast is the largest remaining error,** and it grew with the solar maximum.
  - A lead-dependent bias term, or less weight on the upper percentiles (a lower `blend`), would target it.
  - A plain quantile-mapping post-step was tried and made things worse ([quantile_mapping.md](quantile_mapping.md)). The retune chose blend 0.25 on the folds ([official_retune.md](official_retune.md)).
- **High forecasts relax too slowly after fast wind and CMEs.** A decay term conditioned on the last speed, or an interaction between the last speed and lead, could help.
- **The current-sheet distance and X-ray terms hurt quiet-wind runs.** A sign-constrained or regime-gated version would test whether their gain in active runs can be kept without the loss in quiet ones. The hybrid's trees did not manage this ([hybrid_pfss.md](hybrid_pfss.md)).
- **Coronal-hole coverage matters.** 20% of eval-year runs fell back to the submodel. Live, the strip comes from the AIA image about 72 h earlier; checking its live availability (and a better fallback than the submodel) is cheap.
- **The 13.5-day recurrence term could be dropped** for simplicity. Its effect is below the rule either way.
