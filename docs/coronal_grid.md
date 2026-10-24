# Coronal-hole latitude grid, B0 and darkness (not built)

**What it is.** Finer coronal-hole features: the central-meridian strip split by latitude, latitudes corrected for Earth's B0 tilt, and a darkness-weighted hole area (item 3 of [model_review.md](model_review.md), after Collin et al. 2025 and Obridko et al. 2009).
- metrics: `feature_engineering.coronal_holes.grid_metrics` and `b0_angle` (not used by the models);
- study: `src/training/coronal_grid.py` (`images` recomputes the metrics from the cached AIA images into `src/data/features/coronal_grid.csv`; `folds` scores them);
- analysis: `src/experiments/coronal_grid.ipynb`.

**Result:** not built. The best addition gains 0.3% (3 of 6 folds). The low-latitude strip helps a lot at solar maximum but loses heavily at minimum.

## Method

- **Metrics,** within |longitude| ≤ 10°, with latitudes corrected for B0 (the heliographic latitude of the disk centre, ±7.25° over the year):

| Name | Region |
|---|---|
| `ch_cm_b0` | \|lat\| ≤ 60°: the shipped strip, B0-corrected |
| `ch_cm_low` | \|lat\| ≤ 30° |
| `ch_cm_earth` | \|lat − B0\| ≤ 15° |
| `ch_cm_toward`, `ch_cm_away` | the hemisphere tilted towards or away from Earth, 10° < \|lat\| ≤ 60° |
| `ch_cm_depth` | \|lat\| ≤ 60°, each hole pixel weighted by how far below the threshold it is |

- **Images:** the 13,950 usable cached AIA 193 Å browse images, with the same segmentation as the shipped strip. They take 0.08 s each, 12 min with 4 processes.
- **Model:** every metric is lead-aligned at D = 72 h, like the winner of [lead_aligned_ch.md](lead_aligned_ch.md). Its linear model ("at 72", 0.4257) is the reference.
- **Variants:** replace the strip with a metric, add a metric to it, add both hemispheres, or add B0 at t0.
- **Folds:** the six yearly walk-forward folds. Score: per-run mean TIMSE / persistence, as a geometric mean over folds. The eval year is not used.
- **Run:** `python -m training.coronal_grid images`, then `folds` (about 15 min).

## Results

| Variant (lead-aligned, linear) | Score | vs reference | Folds won |
|---|---|---|---|
| **reference: strip \|lat\| ≤ 60°** | **0.4257** | | |
| + Earth-facing band | 0.4246 | −0.3% | 3 |
| + away hemisphere | 0.4246 | −0.3% | 3 |
| + low-latitude strip | 0.4251 | −0.1% | 4 |
| + darkness / + toward / + both hemispheres | 0.426 | +0.05% to +0.1% | 2 |
| + B0 at t0 | 0.4275 | +0.4% | 3 |
| B0-corrected strip | 0.4281 | +0.6% | 4 |
| darkness-weighted strip | 0.4280 | +0.5% | 2 |
| low-latitude strip | 0.4360 | +2.4% | 4 |
| towards-Earth hemisphere | 0.4422 | +3.9% | 0 |
| Earth-facing band | 0.4434 | +4.2% | 2 |
| away hemisphere | 0.4647 | +9.2% | 0 |

**Low-latitude strip replacing the full strip, by fold:**

| Fold | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|
| change | +21% | +11% | −1.6% | −5.7% | −2.0% | −5.9% |

## Interpretation

- **Latitude matters, but differently over the cycle.** The low band helps from the rising phase to maximum, and the full strip is needed around minimum. The share of hole area at low latitude is 0.7–1.1 of the strip in every year, so it isn't the area that changes; it is which parts of the holes feed the wind (polar-hole extensions near minimum).
- **A cycle-dependent band weight** is a possible follow-up. With two minimum folds, choosing it would be fitting the folds.
- **B0 and darkness add nothing** at this resolution.
