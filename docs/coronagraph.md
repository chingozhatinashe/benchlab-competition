# Coronagraph (SOHO/LASCO) features, path (i) (not built)

**What it is.** Step 4 of the item-4 plan ([model_review.md](model_review.md), item 4): coronagraph images summarised into generic brightening numbers and fed straight into the speed models, with no CME model ("path (i)").
- feature code: `src/feature_engineering/coronagraph.py`, windows in `feature_engineering.windows` (`coronagraph_candidates`);
- live connector: `src/connectors/lasco.py`; history: `src/training/pull_lasco_images.py`;
- study: `src/training/coronagraph_features.py`; analysis: `src/experiments/coronagraph_features.ipynb`.

**Result:** not built. The best sets gain 0.35% on the hybrid (3 of 6 folds), against a bar of 1% and 4 of 6. The signal is there: on CME runs the hybrid gains about 3%. But the trees also react to brightenings that never reach Earth, and lose about 1% on the other 82% of runs. That trade-off is the case for path (ii), a CME detector that turns the brightening into a quiet, Earth-directed signal.

## Data

- **Source:** the LASCO C2 and C3 512-pixel browse JPEGs, `soho.nascom.nasa.gov/data/REPROCESSING/Completed/YYYY/{c2,c3}/YYYYMMDD/` (already a declared domain). Images are posted about 15 minutes after capture. History and live read the same directories, so the product is identical.
- **Cadence:** the first C2 image of every hour and the first C3 image of every second hour (`coronagraph.select_times`), the same in history and live.
- **History:** 2016-10 → 2026-10, 121 months: 85,934 C2 and 43,244 C3 images (about 98% of the hours), 0.1% unusable. Each image is reduced to its cells and discarded: `src/data/features/lasco_cells.csv`, 26 MB.
- **Live cost:** 79 h of images (the 72 h window plus the 6 h background) is about 120 C2 + 40 C3 images, fetched in about 22 s.

## Features

1. **Cells:** each image in grey levels; the median in 12 position-angle sectors × 2 radial rings above the occulter (C2 100–140 and 140–200 px; C3 45–90 and 90–180 px). Medians ignore stars and cosmic rays. A frame whose cells are nearly black is unusable.
2. **Brightening:** each cell divided by its own median over the previous 6 h (earlier images only), minus 1. Steady streamers, planets and stars cancel. The background is floored at 5 grey levels, so a cell over a partly blanked stretch stays finite (38 of 129k images were affected).
3. **Per-image summaries:** `share` (fraction of outer-ring sectors brightened by more than 10%: a halo brightens them all), `max` and `mean` over the cells. Six signals, `c2share` … `c3mean`.
4. **Windows per t0:** the largest value over images taken in (t0 − h, t0 − 30 min], h = 6, 24, 72 h; NaN with fewer than half the expected images. 18 features, `c2max_max_24h` etc., available for 97% of training forecast times.

**Check before the study:** against the CDAW catalogue (used only as a label), the C2 signals 1 h after the onset of wide CMEs (≥ 120°) separate them from quiet hours: AUC 0.92 (`c2max`), 0.86 (`c2mean`), 0.77 (`c2share`). C3 is weaker (0.53–0.68).

## Method

- **Variants:** five feature sets added to the hybrid's 30 regressors (trees refitted with the shipped settings on the shipped linear model), and two single terms added to the linear model (optional, so a run without them uses the submodel).
- **Folds:** the six yearly walk-forward folds (2019-10 → 2025-10). Score: per-run mean TIMSE / persistence, geometric mean over folds. The eval year is not used.
- **By regime:** the same ratio on CME runs (`training.regime_models` labels, 18% of runs) and on the rest, against the shipped model on the same runs.
- **Keep rule, set before running:** more than 1% better on the six-fold score, winning at least 4 of 6 folds, with the CME runs better and the other runs not worse by more than 0.5%.
- **Run:** `python -m training.coronagraph_features` (about 6 min after the history pull; the pull took about 6 h).

## Results

| Variant | Score | vs shipped | Folds won | CME runs | Other runs |
|---|---|---|---|---|---|
| hybrid + all C2 (9) | 0.4175 | −0.36% | 3 | −3.3% | +1.0% |
| hybrid + C2 max + share (6) | 0.4175 | −0.35% | 3 | −2.8% | +0.7% |
| **hybrid shipped** | **0.4190** | | | | |
| hybrid + C2 max + mean (5) | 0.4199 | +0.22% | 2 | −2.3% | +1.3% |
| hybrid + C2 + C3 max (6) | 0.4224 | +0.81% | 2 | −1.6% | +1.4% |
| hybrid + C2 max (3) | 0.4225 | +0.84% | 2 | −0.3% | +1.1% |
| linear shipped | 0.4256 | | | | |
| linear + `c2max_max_24h` | 0.4269 | +0.31% | 2 | +0.2% | +0.7% |
| linear + `c2max_max_72h` | 0.4270 | +0.33% | 0 | +1.8% | +0.4% |

- **CME runs gain in the hybrid, the rest lose.** With all C2 features, CME runs gain in 4 of 6 folds (up to −8.2% in 2019-10), and the other runs lose in 4 of 6 (up to +3.0% in 2020-10).
- **C3 adds nothing:** adding the C3 maxima to the C2 maxima is no better than C2 alone, as the separation check suggested.
- **The linear model can't use it:** a single window is linear in the brightening, but a CME's effect arrives 1–4 days later, only if Earth-directed, and on a few runs. The coefficient stays small and adds noise.
- **Why the trees lose on quiet runs:** most brightenings are limb CMEs and streamer changes that never reach L1. With about 100 Earth-directed CMEs in nine years, the trees can't learn from the summaries alone which brightenings matter, and spend splits on the others.

## Decision

Not built. The connector, the history, the feature code and the study stay as the inputs for path (ii): a CME detector trained on the CDAW labels, with physical constraints (rotation-invariant sector patterns, monotone in brightening, a bounded 1–4 day arrival), whose output is a feature for the hybrid. That detector targets exactly the quiet-run loss found here. *Done:* not built either ([cme_detector.md](cme_detector.md)). The coronagraph features are not part of the forecast: `lasco_hours` is 0 for the shipped regressors, so live runs don't fetch LASCO.
