# Magnetogram (SDO/HMI) features, path (i) (not built)

**What it is.** Step 3 of the item-4 plan ([model_review.md](model_review.md), item 4). The photospheric magnetic field is summarised into generic numbers and fed straight into the speed models, with no model of the corona ("path (i)").
- feature code: `src/feature_engineering/magnetogram.py`, windows in `feature_engineering.windows` (`mg_*` signals);
- history: `src/training/pull_hmi_images.py`, through `connectors.sdo.SDOConnector(channel="HMIB")`;
- study: `src/training/magnetogram_features.py`; analysis: `src/experiments/magnetogram_features.ipynb`.

**Result: not built.**
- The best set gains 0.36% on the hybrid (4 of 6 folds); the bar was 1% and 4 of 6.
- The linear-model terms change the score by at most 0.04%.
- The browse magnetograms show active regions, not the weak one-sided field of coronal holes, so they add little to the AIA coronal-hole feature the models already use. Following the plan, the PFSS/WSA build (step 5) is not started.

## Data

**Sources checked (2026-10-25):**

| Source | Result |
|---|---|
| GONG synoptic maps (`gong2.nso.edu`, `gong.nso.edu`, `nispdata.nso.edu`) | Unusable: the names resolve, but connections time out from the dev machine. |
| JSOC (`jsoc.stanford.edu`) | Unusable. Metadata answers over plain HTTP, but file downloads redirect to the unreachable `jsoc1`. |
| JSOC `hmi.Mrdailysynframe_720s` (synoptic frames) | Unusable: created 7–29 days late. |
| JSOC `hmi.m_720s_nrt` (near-real-time series) | Unusable: about 45 min latency, but only the last 90 days are kept, so there is no history. |
| **SDO HMI quick-look browse magnetograms (used)** | `sdo.gsfc.nasa.gov/assets/img/browse/YYYY/MM/DD/YYYYMMDD_HHMMSS_512_HMIB.jpg`: every 15 min since 2016, posted within about 30 min, and the same server as the AIA images (already a declared domain). History and live read the same files. |

**Encoding:** the line-of-sight field in grey. About 127 is no field, white is positive, black is negative, and strong field saturates. The gauss scale can't be recovered, so every number is in grey levels.

**Resolution:** 512 px. The 10° grid built from 512 and from 1024 images agrees (r 0.98–0.995, May 2024), and 512 takes half the time.

**History:** 2016-10 → 2026-10, 121 months. One image per 6-hourly cut-off t0 − 1 h; if that one is missing or unusable, the newest older one, up to 24 h older.
- 14,636 cut-offs, 95.2% with a usable image.
- Gaps are the same as AIA's: 2024 91% covered, 2025 71%, 2026 89%.
- Each image is reduced to its grid and discarded: `src/data/features/hmi_grid/YYYY-MM.npz`, about 1.3 h to pull.

## Features

1. **Disk:** found in the image itself, from the extent of the non-black pixels within a centred circle. This leaves out the caption, and black sunspots don't change the extent. A frame whose radius differs from the expected one by more than 3%, or whose disk median is more than 12 grey levels from 127, is unusable.
2. **Grid:** heliographic latitude and longitude (B0 included) on 10° × 10° cells over the inner 95% of the disk. Per cell:
   - the signed and unsigned mean field;
   - the share of strong positive and strong negative pixels (more than 40 grey levels from zero);
   - mean μ;
   - the pixel count.
3. **Zero level:** the JPEG's zero level drifts by about half a grey level. Each image's zero is taken as the median signed value of its disk cells, which are mostly quiet Sun.
4. **Signals per image** (area-weighted; the central strip is |longitude| ≤ 10°, |latitude| ≤ 60°; Br = line-of-sight / μ):
   - `mg_cm_signed`: the strip's net Br (polarity);
   - `mg_cm_unipolar`: its size (a coronal hole is one-sided open field);
   - `mg_cm_unsigned`: the strip's total field;
   - `mg_cm_strong`: its share of strong pixels;
   - `mg_cm_imbalance`: |positive − negative| / (positive + negative) of the strong pixels;
   - `mg_disk_unsigned`: total field over |longitude| ≤ 60°;
   - `mg_polar`: the 60–80° north cap minus the south cap.
5. **Windows:** read like the AIA strips. `mg_cm_unipolar_at_24h` is the image 24 h before the latest by t0 − 1 h, with the same 6 h and 12 h fallbacks. The `_lead_72h` terms are lead-aligned like `ch_cm_lead_72h`. The scramble test (`tests/test_magnetogram.py`) checks that data after the cut-off changes nothing.

**Checks before the study:**
- **Polar field:** the polar signal follows the solar cycle. North minus south is +0.06 to +0.07 in 2016–2021, after cycle 24's reversal, and falls to about −0.03 from 2023, after cycle 25's.
- **Coronal holes aren't visible.** Over the 6,658 runs with both signals, with an 81-day running mean removed (the solar-cycle trend):
  - the strip's unipolarity is uncorrelated with the AIA coronal-hole fraction `ch_cm` (r = 0.02) and with the mean speed 73–96 h later (r = 0.03);
  - only the strong-field imbalance tracks later speed (r = 0.16), against 0.32 for `ch_cm`.

  At 8 bits over a range set for active regions, a coronal hole's net field of a few gauss is lost in the noise.

## Method

- **Variants:**
  - four sets added to the hybrid's 30 regressors (trees refitted with the shipped settings on the shipped linear model): unipolar at lags 0/24/48/72 h; "cm" (unipolar, unsigned and strong at 24/72 h); cm + polarity; cm + imbalance, disk and polar;
  - four single terms added to the linear model, lead-aligned at 72 h.
- **Linear terms aren't optional.** The linear model has one fallback submodel, fitted without all its optional features, so a missing magnetogram would also drop `ch_cm_lead_72h`. Missing values are filled with the training median instead.
- **Folds:** the six yearly walk-forward folds (2019-10 → 2025-10).
  - Score: per-run mean TIMSE / persistence, geometric mean over folds.
  - Also scored on CME runs and on the rest (`training.coronagraph_features.route_ratios`).
  - The eval year is not used.
- **Keep rule, set before running:** more than 1% better on the six-fold score, winning at least 4 of 6 folds.
- **Run:** `python -m training.magnetogram_features`, about 8 min (pilot: one fold, 98 s, 682 MB).

## Results

| Variant | Score | vs shipped | Folds won | CME runs | Other runs |
|---|---|---|---|---|---|
| hybrid + unipolar (4) | 0.4175 | −0.36% | 4 | +0.1% | −0.5% |
| hybrid + cm + polarity (8) | 0.4179 | −0.25% | 3 | +0.1% | −0.1% |
| **hybrid shipped** | **0.4190** | | | | |
| hybrid + cm (6) | 0.4199 | +0.21% | 2 | +0.8% | +0.4% |
| hybrid + cm + disk + polar (10) | 0.4217 | +0.66% | 2 | +1.7% | +0.4% |
| linear + `mg_cm_strong_lead_72h` | 0.4254 | −0.04% | 4 | −0.4% | +1.1% |
| linear + `mg_cm_imbalance_lead_72h` | 0.4254 | −0.04% | 4 | +2.3% | −0.6% |
| linear + `mg_cm_unsigned_lead_72h` | 0.4254 | −0.03% | 4 | −0.4% | +1.0% |
| linear shipped | 0.4256 | | | | |
| linear + `mg_cm_unipolar_lead_72h` | 0.4268 | +0.30% | 2 | +1.4% | +0.2% |

- **The best gain doesn't hold.** The unipolar set gains 0.6–2.1% in 2019-10 to 2021-10 (solar minimum, rising phase), loses 0.8–1.0% in 2022-10 and 2024-10, and is flat in 2023-10.
- **More features make the hybrid worse.** The ten-feature set is the worst. The trees spend splits on active-region numbers, which are mostly a solar-cycle clock that the existing regressors (F10.7, the flare counts) already give them.
- **The linear model can't use them.** Every term moves the score by less than 0.1%. The one-fold pilot gain (−0.8% on 2024-10) didn't hold on other years: the same terms lose 0.9–1.6% in 2020-10 and 2023-10.
- **Nothing helps CME runs.** As expected: CMEs come from active regions, but a magnetogram doesn't say whether one will erupt, or when.

## Decision

Not built. The magnetogram path closes at path (i).
- **What stays:**
  - the connector change (`SDOConnector(channel="HMIB")`), the history pull and the feature code, as inputs for any path (ii) study;
  - the window plumbing (`Inputs.magnetogram`).
- **Not part of the forecast:** live runs never fetch HMI images, and no shipped regressor uses them.
- **Path (ii) outlook:** a monotone source-speed proxy or a polarity predictor would read the same images, so it would face the same limit. With the trend removed, the strongest signal correlates at only 0.16, which isn't promising.
- **What a real step needs:** a magnetogram with a calibrated field (GONG, or the full HMI FITS), which needs a network where `gong2.nso.edu` or `jsoc1.stanford.edu` answers.
