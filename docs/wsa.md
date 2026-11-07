# WSA/HUX speed with auto-calibration (not built)

**What it is.** The Wang-Sheeley-Arge (WSA) empirical speed relation, applied to the same PFSS
rings as the current-sheet feature ([pfss.md](pfss.md)), carried to Earth by the HUX upwind
scheme, and calibrated automatically:
- **offline**, per fold, on the training years;
- **rolling**, in each run, on the run's own recent days.

It was scored on its own and as a regressor added to the submitted linear model, which includes
`pfss_d_hcs_lead_72h` since 2026-11-07.
- **Model:** `src/models/wsa.py` (numpy only).
- **Study:** `python -m training.wsa_model` (`--sample 500` for the pilot).
- **Analysis:** `src/experiments/wsa_model.ipynb`; results in `src/experiments/wsa_model.csv`.

**Result: nothing passes the 1% / 4-of-6 rule, so nothing is built.**
- **Best variant:** the offline-calibrated WSA speed for the hour being forecast (`wsa_speed_lead_0h`), 0.75% better on 4 of 6 folds.
- **WSA alone:** 7.5% worse than persistence.
- **Rolling re-fit:** it makes the feature worse.

The submitted config stays.

## Model (`models.wsa`)

1. **Source speed,** per ring longitude (Arge et al. 2003; McGregor et al. 2011):

       v = v0 + dv / (1 + fs)^α · (1 − 0.8·exp(−(θb / w)^δ))³,  clipped to 200–950 km/s

   fs is the expansion factor and θb the footpoint's distance to the coronal-hole boundary, from the PFSS ring (`feature_engineering.pfss`). Gaps in the ring are interpolated around the circle.
2. **Propagation** from 21.5 R☉ to 1 AU by HUX (Riley & Lionello 2011), the kinematic scheme HUXt is built on, in the frame that corotates with the Sun at the sidereal rate:
   - v(r + dr, φ) = v + dr·Ω / v · ∂v/∂φ, in 1 R☉ steps;
   - fast wind steepens into the slow wind ahead of it instead of overtaking it.
   - HUXt itself needs Python 3.12, so these 10 lines are written in numpy (`wsa.hux`).
3. **Reading at Earth:** the steady 1 AU solution is read at the Carrington longitude facing Earth at each forecast hour (`pfss.earth_longitude`), plus a calibrated `shift`.

`tests/test_wsa.py` checks:
- uniform wind stays unchanged;
- a narrow stream drifts by its travel time;
- fast wind steepens into the slow wind ahead;
- the relation's limits;
- the interpolation around the ring;
- the rolling fit.

## Calibration

- **Offline:** per fold, the six parameters (v0, dv, α, w, δ, shift) by Nelder-Mead. The objective is the official S_final of the WSA forecast alone on 1,500 of the fold's fit runs. It starts from the literature values (285, 625, 2/9, 2°, 1.25, 0) and stays within bounds. Runs without a ring count as persistence.
- **Rolling:** in each run, the same map's hindcast over the last 14 or 27 days (the longitudes that faced Earth then) is compared with the ACE hours that ended by t0 − 1 h. An offset and a scale are fitted by least squares, with a ridge pull toward 0 and 1 (`wsa.rolling_fit`). No extra PFSS solve is needed.

**Calibrated values per fold:**

| Fold | v0 | dv | α | w (deg) | δ | shift (deg) | S_final default → calibrated (fit runs) |
|---|---|---|---|---|---|---|---|
| 2019 | 337 | 900 (cap) | 0.33 | 0.35 | 0.47 | 1.7 | 186 → 111 |
| 2020 | 330 | 793 | 0.33 | 0.53 | 0.99 | 7.5 | 175 → 106 |
| 2021 | 342 | 855 | 0.36 | 0.47 | 0.83 | 1.8 | 172 → 102 |
| 2022 | 353 | 722 | 0.37 | 0.46 | 1.35 | 1.7 | 180 → 107 |
| 2023 | 364 | 717 | 0.34 | 0.54 | 0.88 | 2.6 | 180 → 107 |
| 2024 | 375 | 632 | 0.33 | 0.76 | 3.81 | 1.8 | 183 → 109 |

- **Error:** calibration cuts S_final on the fit runs by about 40%. The literature values, tuned for other magnetograms and solvers, run far too fast here.
- **Exponent:** α is stable (0.33–0.37), near the published 2/9–1/3.
- **Boundary width:** w is 0.35–0.76°, well below the published 2–4°. On these maps the boundary distance barely matters, which matches the PFSS study, where `theta_b` added nothing on its own.
- **Slow-wind floor:** v0 rises steadily as more solar-maximum years enter the fit.

## Method

- **Base:** the submitted config (`bt_mean_1h`, `ch_cm_lead_72h`, `xray_peak_max_72h`, `density_mean_24h`, `pfss_d_hcs_lead_72h`), refitted per fold with the data fixes. It reproduces its score exactly (0.7794 of persistence).
- **Variants:** `wsa_speed_lead_<D>h` added, at D = 0, 12 or 24 h, with offline parameters or with each rolling window. Lead k reads the WSA speed at the longitude facing Earth at t0 + k − D. The map was observed before t0, so lags may be negative (`linear_model.SIGNED_LEAD_PREFIXES`).
- **Missing values:** a missing value (no usable map) and the base's PFSS columns get the median of the fold's fit rows.
- **Score:** official S_final against the official truth, relative to persistence, as a geometric mean over the six yearly folds. The eval year is not used.
- **Decision rule, set before running:** keep only if more than 1% better than the submitted config and winning at least 4 of 6 folds.
- **Run:** a 500-run pilot (7 min; estimate 2 h), then all 13,148 runs (62 min). The pilot already showed no variant near the rule; the full run was made for completeness.

## Results

| Variant | / persistence | vs submitted | Folds won |
|---|---|---|---|
| submitted config | 0.7794 | — | — |
| + WSA, lead 0 h, offline | 0.7736 | −0.75% | 4 |
| + WSA, lead 12 h, offline | 0.7754 | −0.51% | 4 |
| + WSA, lead 24 h, offline | 0.7774 | −0.26% | 4 |
| + WSA, lead 0 h, rolling 14 d | 0.7785 | −0.12% | 4 |
| + WSA, lead 12 h, rolling 14 d | 0.7790 | −0.05% | 4 |
| + WSA, lead 24 h, rolling 14 d | 0.7796 | +0.03% | 3 |
| + WSA, rolling 27 d (any lead) | 0.7802–0.7803 | +0.10 to +0.12% | 3 |
| WSA alone, offline | 1.0751 | | |
| WSA alone, rolling 27 d | 1.1003 | | |
| WSA alone, rolling 14 d | 1.1444 | | |

**By fold:** the best variant changes the score by −0.2% (2019 fold), +0.3% (2020), −0.5% (2021), −2.7% (2022), +0.5% (2023) and −1.9% (2024). The gain comes from two folds.

**WSA alone by fold** (/ persistence): 1.12, 0.99, 1.10, 1.12, 1.15, 0.99. Calibrated WSA never beats persistence by more than 1% in any fold.

## Why it doesn't help

- **The linear model already has the map's information.** The current-sheet distance and the coronal-hole strip carry the slow/fast pattern. In the PFSS study, the expansion factor added only 0.4–0.6% on its own; passing it through the WSA formula and HUX adds about the same (0.75%).
- **WSA's timing errors cost more than its structure gains.** A stream placed a few degrees off (about 7 h per 4°) is penalised twice by the official metric, through RMSE and through the percentile TIMSE. Persistence makes no such errors in the first hours.
- **The rolling re-fit chases noise.** Over 14–27 days, the offset and scale fitted to the hindcast mostly track the last stream's mismatch, which doesn't carry over to the next 72 h.
- **One map, a steady solution.** The forecast uses the newest map only, as a steady corotating solution, so evolving or transient structure (CMEs) is outside it. Ensembles of maps, time-dependent HUX, or a neural correction (as in the WSA literature) could help, but not within this cycle.

## Not tried

- WSA in the XGBoost hybrid.
- Calibrating against the linear model's objective instead of WSA alone.
- Other source-surface heights (rss = 2.5 only; the raw maps are cached in `src/data/gong/raw/` to allow this).
- Ensembles of maps (e.g. the last 24 h of GONG maps).
