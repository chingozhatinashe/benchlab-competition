# PFSS features: current-sheet distance (built 2026-11-07)

**What it is.** Coronal magnetic-field features from a potential-field source-surface (PFSS)
model of GONG magnetograms. This is the largest physics gap in [model_review.md](model_review.md)
§3.1: Bailey et al. 2021 rank the flux-tube expansion factor and the distance to the coronal-hole
boundary as the most decisive WSA inputs. Each feature was added to the submitted linear model
and scored on the official metric.
- **Ring history:** `python -m training.pull_gong inventory`, then `python -m training.pull_gong rings`.
- **Study:** `python -m training.pfss_features` (`--sample 500` for the pilot).
- **Analysis:** `src/experiments/pfss_features.ipynb`; results in `src/experiments/pfss_features.csv`.

**Result:**
- **The current-sheet distance passes.** `pfss_d_hcs_lead_72h` is 1.66% better than the submitted config and wins 5 of 6 folds (0.7794 of persistence, against 0.7926).
- **Nothing else passes.** The expansion factor helps a little (−0.3 to −0.6%). The coronal-hole boundary distance, footpoint latitude and polarity, and the map scalars add nothing or hurt.
- **It ships alone.** The linear model was retrained with it and switched on live.

## Data: GONG synoptic maps

- **Maps:** zero-point-corrected hourly synoptic maps (`mrzqs…fits.gz`, about 240 KB), from `gong2.nso.edu/oQR/zqs/`. The NSO archive keeps each file's original upload time, so availability can be checked per file over the whole history.
- **Inventory** (`training.pull_gong inventory`): the day listings, 2016-10 → now. Listing times are MST (UTC−7 all year).
- **Posting delay:** the median is 37–71 min a year.
- **Backfill:** 383 days in 2018–2022 were bulk-backfilled weeks to months later. A live run would never have had those maps, so they are not used.
- **Availability rule** (shared by the history and live runs, `connectors.gong.usable`): the newest map uploaded by t0 − 10 min and observed within 72 h.

| Year | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t0s with a usable map | 100% | 99% | 92% | 66% | 81% | 88% | 88% | 97% | 100% | 100% | 95% |

**Gaps:** a run without a map gets the training median. 2020-08 and 2022-01 have no usable map at all. Ring coverage matches the inventory month by month, so no solve failed.

## The PFSS ring (`feature_engineering.pfss.reduce`)

Each map is solved and reduced with exactly the same code in the history pull and in the live runs.

- **Solver:** sunkit-magex 1.1.0 (GPL-3, the successor of pfsspy, archived 2025-10), nr = 35 radial cells, source surface at 2.5 R☉. Field lines are traced with streamtracer.
- **Open/closed mask:** photospheric seeds on a 2° grid. A field line that reaches the source surface is open.
- **The ring:** 360 seeds on the source surface, one per degree of Carrington longitude, at the sub-Earth latitude B0. These are the field lines that connect to the ecliptic. Per longitude:

| Signal | Meaning |
|---|---|
| `log_fs` | log10 expansion factor, (B_phot / B_ss)·(1 / rss)² |
| `theta_b` | angular distance of the footpoint to the nearest closed-field footpoint (coronal-hole boundary distance) |
| `foot_lat` | absolute footpoint latitude |
| `foot_pol` | footpoint polarity |
| `br_ss` | Br on the source surface |
| `d_hcs` | angular distance on the source surface to the heliospheric current sheet (Br_ss = 0) |

- **Map scalars:** the open area fraction, the open flux and the maximum current-sheet latitude.
- **Cost:** about 4 s per map, 270 MB resident and about 650 MB peak per process.
- **History:** one map per 12 h (each t0 uses the map chosen at t0 floored to 12 h, which was posted by then), 121 months, about 8 h with 2 processes. The rings are stored in `src/data/features/pfss/YYYY-MM.npz`.

## Windows: the longitude facing Earth

- **Reading the ring:** `pfss_<var>_at_<L>h` is the ring of the newest map at t0, read at the Carrington longitude that faced Earth L hours before t0. Earth's longitude is sunpy's L0, cached per month and interpolated (`pfss.earth_longitude`).
- **Lead alignment:** `pfss_<var>_lead_<D>h` works as for the coronal-hole strip. Lead k reads the longitude that faced Earth about D hours before t0 + k (`Config.input_features` expands it into 13 lag columns).
- **Negative lags:** one map covers all longitudes, so ring signals may also take negative lags, i.e. longitudes that will face Earth after t0 (`linear_model.SIGNED_LEAD_PREFIXES`). The shipped 72 h delay needs lags 0–72 h only.
- **No future data:** `tests/test_pfss.py` checks that scrambling every map posted after t0 leaves the features unchanged.

## Method

- **Base:** the submitted config (`bt_mean_1h`, `ch_cm_lead_72h`, `xray_peak_max_72h`, `density_mean_24h`; official loss weights at blend 0.25; ridge 0.1), refitted per fold with the data fixes.
- **Variants:**
  - each ring signal at a 72 h delay, and each scalar at t0;
  - the best two ring signals at 24, 48, 96 and 120 h;
  - the best two together, and with the best scalar.
- **Missing values** get the median of the fold's fit rows, so every variant fits on the same runs, as the live model imputes.
- **Score:** official S_final against the official truth, relative to persistence, as a geometric mean over the six yearly folds. The eval year is not used. The submitted config reproduces its score exactly (0.7926).
- **Decision rule, set before running:** keep only if more than 1% better than the submitted config and winning at least 4 of 6 folds.
- **Run:** a 500-run pilot, then all 13,148 runs (2.5 min). A first pilot took 29 min because Earth's longitude was recomputed for every column; it is now cached per month.

## Results

S_final of the submitted config: 88.49 km/s (0.7926 of persistence).

| Variant | / persistence | vs submitted | Folds won |
|---|---|---|---|
| `d_hcs` 72 h + `log_fs` 48 h | 0.7778 | −1.86% | 6 |
| … + `hcs_max_lat` | 0.7787 | −1.75% | 5 |
| **`d_hcs` 72 h (shipped)** | **0.7794** | **−1.66%** | **5** |
| `d_hcs` 48 h | 0.7805 | −1.52% | 5 |
| `d_hcs` 24 h | 0.7845 | −1.01% | 4 |
| `d_hcs` 96 h | 0.7856 | −0.88% | 4 |
| `log_fs` 48 h | 0.7882 | −0.55% | 3 |
| `log_fs` 72 h | 0.7890 | −0.44% | 3 |
| `foot_lat` 72 h | 0.7918 | −0.10% | 4 |
| `hcs_max_lat` | 0.7919 | −0.08% | 2 |
| `br_ss` 72 h | 0.7922 | −0.05% | 2 |
| `theta_b` 72 h | 0.7926 | 0.00% | 3 |
| `foot_pol` 72 h | 0.7930 | +0.06% | 3 |
| `d_hcs` 120 h | 0.7938 | +0.15% | 4 |
| open flux | 0.7976 | +0.64% | 3 |
| open area | 0.8009 | +1.05% | 2 |

**By fold:** `d_hcs` 72 h changes the score by −1.6% (2019 fold), −1.6% (2020), −0.3% (2021), −3.2% (2022), −3.9% (2023) and +0.7% (2024).

**Why `d_hcs` alone:**
- The pair with the expansion factor is 0.2% better, well under the 1% a feature has to earn.
- The pair's 48 h delay was picked on the same folds.
- The WSA study (next) tests the expansion factor again inside the speed formula.

**What it measures:**

| Current-sheet distance at the longitude facing Earth at t0 | under 5° | 5–10° | 10–15° | 15–20° | 20–30° | 30–45° | over 45° |
|---|---|---|---|---|---|---|---|
| median speed 72 h later | 378 km/s | 385 km/s | 391 km/s | 407 km/s | 420 km/s | 460 km/s | 480 km/s |

Wind from near the current sheet is slow, and wind from far from it is fast. This is the slow-wind belt around the sector boundaries, which the speed history and the coronal-hole strip don't fully capture.

**What didn't work:**
- **The expansion factor and boundary distance:** they don't help on their own in a linear model. They are the WSA inputs, and they may only help combined through the WSA formula.
- **The open area:** it hurts. It is a whole-Sun number that doesn't say which longitude faces Earth.

## Live

- **Connector** (`connectors.gong.GONGConnector.latest`):
  - It first reads SWPC's copy of the maps (`services.swpc.noaa.gov/products/gong/zqs/`, byte-identical, posted about 4 min after NSO, a UTC listing of the last 72 files).
  - Then it falls back to NSO's day listings across three host names (`gong2.nso.edu`, `gong.nso.edu`, `nispdata.nso.edu`, all one server, which timed out from GitHub's runners).
  - It tries up to 3 maps, the newest first.
- **Fetch:** `inference.fetch_inputs` downloads the map and solves the ring in parallel with the other sources (300 s budget), only when the selected regressors use PFSS.
- **Failures:** if no map is found or the solve fails, the column takes its training median and the forecast goes on.
- **Run** (2026-11-07 12Z, locally): the map was 1.8 h old, and the fetch and solve took 11.8 s. All 29 regressor columns were observed, and the whole forecast took 21 s.
- **Image:** the PFSS stack (sunkit-magex, sunpy, astropy, scipy, scikit-image, streamtracer) takes the image to 649 MiB, an estimated 6 s pull. The solve peaks at about 650 MB, well within the declared 16 GiB.
- **Shadow staging** (`upload_candidate/pfss-shadow-3`, 2026-11-07): the PFSS input was fetched and solved on every run with the forecast unchanged. On GitHub's runners GONG was found through SWPC.
- **Live and history differ slightly:** SWPC posts about 4 min after NSO, so a live run can occasionally use a map about 1 h older than the history rule would. This difference was accepted.

## Licence

sunkit-magex is GPL-3. Using it at runtime was accepted on 2026-11-07. It is installed unmodified from PyPI (`src/requirements.txt`).
