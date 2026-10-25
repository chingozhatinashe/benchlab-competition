# Regressors beyond speed

The reference baseline (`docs/final_baseline.md`) forecasts from past speed alone. It
over-forecasts quiet slow wind by about 100 km/s and gives no warning of stream or
CME arrivals (`src/experiments/final_baseline_analysis.ipynb`). This document
defines the extra signals a later model can use:
- what each signal is;
- where it comes from, live and in history;
- the rule that keeps it free of future data;
- how gaps are filled.

The code is in `src/feature_engineering/`, `src/connectors/` and
`src/inference/fetch_inputs.py`. The history pull is `src/training/pull_features.py`.

Nothing in the live forecast uses these regressors yet. They are checked in the
real container with a dry run (`python -m inference.check_features <t0>`).

## The regressors

There are 40 regressors, each one value per forecast time t0 (`feature_engineering/registry.py`).

| Group | Regressors | Why they should help |
|---|---|---|
| **Plasma** (12) | density (last hour, 6 h, 24 h, 6 h slope), log temperature (6 h), temperature ratio T/T_exp(V) (6 h), dynamic pressure (last hour, 6 h, 24 h max), density and pressure over the 72 h one rotation (27 d) earlier, density half a rotation (13.5 d) earlier | A density and pressure pile-up (a stream interface) comes before a fast stream. One rotation earlier, the same pile-up marks a recurrent stream about to return. An abnormally cold plasma marks CME ejecta. |
| **Magnetic field** (8) | Bt (last hour, 6 h, 24 h), Bz (last hour, 6 h minimum), sector polarity −Bx/Bt (6 h, and one rotation earlier), field-direction variability (6 h) | A strong field marks compressions and CMEs. A polarity change (sector boundary) sits next to the slow-wind belt. A smooth, low-variability field marks a flux rope. |
| **CME and shock state** (6) | share of the last 6 h and 24 h with CME ejecta signatures, hours since the last interplanetary shock, EPAM 47–68 keV proton flux and its 6 h trend, EPAM 310–580 keV flux | It tells the model when it is inside or behind a CME. A rising low-energy proton flux precedes a CME shock by hours to a day. |
| **Coronal holes** (8) | coronal-hole area share of the central-meridian strip (±10° longitude, \|latitude\| ≤ 60°) 0, 1, 2, 3, 4 and 5 days before t0; share 20–40° east of the meridian; share of the disk within 60° | Fast wind comes from coronal holes. Wind from a hole crossing the central meridian reaches Earth about 3–4 days later, and its peak speed grows with the hole's area (Vršnak et al. 2007). |
| **Eruptive activity** (4) | log10 peak X-ray flux of flares in the last 24 h and 72 h, number of M/X flares in the last 72 h, hours since the last M/X flare | Large flares often launch CMEs, which arrive 1–4 days later. |
| **Solar activity level** (2) | F10.7 of the previous day, 27-day mean | The general activity level: how much slow, CME-rich wind to expect. |

## Sources

Every live source is declared in `src/submission.json`. Probed on 2026-10-21.

| Source | Live (runtime) | History (training) | Same series? |
|---|---|---|---|
| ACE SWEPAM, MAG, EPAM | SWPC real-time daily lists on the SOHO mirror (`soho.nascom.nasa.gov/sdb/ace/daily/`); today's file runs a few minutes behind. The latest minutes come from SWPC RTSW (`services.swpc.noaa.gov/json/rtsw/`, ACE rows) | The same daily files, cached in `src/data/ace_{rtsw,mag,epam}/raw/` | yes, identical files |
| SDO/AIA 193 Å | browse JPEGs (`sdo.gsfc.nasa.gov/assets/img/browse/`), posted about 15 min after capture, every ~10 min | the same images, one per 6-hourly t0, cached in `src/data/aia193/`. The archive has gaps: all of December 2024, part of January 2025, about 2025-09-12 to 2025-12-16 and part of March 2026. Those coronal-hole regressors are imputed (6% of training forecast times, and the first two months of the eval year) | yes, identical files |
| GOES X-ray flares | SWPC `json/goes/primary/xray-flares-7-day.json` | NCEI GOES-R XRS science flare summaries (GOES-16 to 2025-04, GOES-18 from 2022-06), peaks merged within 5 min | yes: peak fluxes agree to 0.3% (median ratio 0.997 over 66 flares in both) |
| F10.7 | SWPC `json/f107_cm_flux.json` (about 40 days; noon reading) | OMNI hourly `F10_INDEX` (CDAWeb) | yes: mean difference 0.4 sfu over the overlap |

**Rejected:**
- **WSA-Enlil** (`json/enlil_time_series.json`): available live, but there's no archive, so it can't be trained on.
- **NASA DONKI CME catalogue:** unreachable (kauai.ccmc times out, api.nasa.gov returns 503).
- **SOHO/LASCO CDAW catalogue:** published months late. It is kept as an analysis label only (`training.pull_lasco_cme`).
- **SOHO/LASCO C2/C3 images** (`connectors.lasco`, `feature_engineering.coronagraph`): the same browse JPEGs live and in history, posted about 15 min after capture. Studied as direct regressors and not built ([coronagraph.md](coronagraph.md)); live runs don't fetch them while no shipped regressor uses them.
- **HEK SPoCA coronal holes:** archived 13 h to 13 days after the observation, and old archive dates don't record when a detection was first published. AIA images are fresher and the same in history and live.
- **Sunspot number:** SWPC publishes the SESC count, while OMNI history uses SILSO, a different scale.

## No future data

Each source has its own availability rule. Live and training apply the same rule in the same code.

| Source | Rule |
|---|---|
| ACE minutes | Only minutes measured before t0 − 10 min. Hourly bins are shifted by the same 10 min (`insitu.hourly_frame`): the row labelled h holds minutes from h − 10 min to h + 50 min, so every row before t0 was published by t0. |
| Shocks | A shock at t needs data up to t + 20 min, so it counts only once t + 20 min ≤ t0 − 10 min. |
| AIA | The newest usable image taken at or before t0 − 1 h − d days, and no more than 24 h before that. |
| Flares | Flares whose peak was at or before t0 − 10 min. |
| F10.7 | Days before the UTC day of t0 only. The day's 20 UT reading is final by midnight, and OMNI spreads one value over the whole day, which would otherwise leak the afternoon reading into the morning. |

`tests/test_features.py::test_features_do_not_change_when_the_future_does` builds every
regressor twice, once with real inputs and once with every value not yet
available at t0 replaced by nonsense, and checks that the two rows are identical.

**Open issue (speed input of the baseline):** the live speed history currently uses
every minute stamped before t0. The last few of those minutes are published just
after t0. The regressors use the 10-minute margin above. Applying the same margin
to the speed input means retraining the baseline; that is a follow-up.

## Gaps and imputation

1. **Bad values become missing.**
   - In-situ: the source's status flags, fill values, and the physical ranges in `insitu.VALID`.
   - AIA: frames that are blank or truncated fail `coronal_holes.usable`, and the next older image is tried, up to 12 images (about 2 h). Both kinds of bad frame occur, for example on 2026-10-21 from 16:20 to 17:00 UT.
2. **Short gaps are carried forward from the past only.** Interpolation would use later values.
   - Last-hour in-situ values: up to 3 h.
   - AIA images: up to 24 h.
   - F10.7: up to 3 days.
3. **Window statistics need half their hours**, otherwise they are missing.
4. **Whatever is still missing gets the training-years median** (`src/assets/feature_climatology.json`), plus a `<name>_missing` flag of 1 (`feature_engineering/impute.py`).
   - Counts are values: no M flares in 72 h from a working feed is 0, while an unreachable feed is missing.
   - If the asset can't be read, typical values from the registry are used instead.

A failed or slow source never stops a run. Each source is one job with a time budget (300 s in all), and its regressors become missing.

## Live cost

`python -m inference.check_features 20261021T180000Z` on a home connection:
- all 8 sources answered;
- 30 days each of SWEPAM, MAG and EPAM files;
- 6 AIA images (one blank frame skipped);
- 40 of 40 regressors from real data;
- 11 s, 12.5 MB downloaded.

The 15-minute limit and the 20 GB network limit are far away.

## First screening (training years only)

`src/experiments/feature_screening.ipynb` relates each regressor to the reference baseline's
out-of-sample errors. Those errors come from the two tuning folds, 2023-10 to 2025-10, 2,908 runs.

- **Field strength (Bt)** is the strongest single signal: Spearman ρ ≈ 0.35 with the residual at 1–48 h.
  - It holds just as well in quiet wind. With the weakest field, quiet-wind runs are over-forecast by about 110 km/s at 25–48 h; with Bt ≈ 10 nT, by about 55 km/s.
- **Coronal-hole strip areas** matter at the leads their travel time predicts:
  - an image from 3 days before t0 peaks at 30 h lead;
  - 2 days before at 48 h;
  - 1 day before at 60 h;
  - the day of t0 at 66 h, with ρ up to 0.26.

  The signal is strongest at 49–72 h, where the baseline is weakest.
- **Density, pressure and the cold-plasma (CME) share** come next (ρ ≈ 0.10–0.20).
- **Little signal at this level:** flares, F10.7, the EPAM trend and the 27-day lagged in-situ regressors. These are the first candidates to drop. Flares may still matter for individual CMEs.
- **Near-duplicates:** the Bt windows, the density/pressure windows, and the three flare measures.

These are associations, not gains. The next step is to add the leading regressors to the
per-lead regression and test it on the training folds.

**Live risk:** the SDO browse archive can stop for weeks (about 2025-09-12 to 2025-12-16). Coronal-hole
regressors then fall back to the median plus a flag. A second image source would remove this
weak point if a model comes to depend on them.

## Rebuilding

```bash
.venv/Scripts/python -m training.pull_features all   # resumable; the AIA step takes the longest
```

It writes `src/data/features/table.csv.gz`, with one row of raw regressors per 6-hourly t0 from
2016-10-20 to 2026-10-20 (14,608 t0s; every regressor present for at least 85% of them) and NaN
where unknown. Flares before GOES-16 (Feb 2017) come from NCEI's reprocessed GOES-15 science flare
summaries; their fluxes read about 10% below GOES-16's for the same flares. It also refits the imputation medians on the
training years and writes them to `src/assets/feature_climatology.json`.
`src/experiments/feature_screening.ipynb` looks at the table on the training years only.
