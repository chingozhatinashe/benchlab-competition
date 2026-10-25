# CME detector on LASCO images, path (ii) (not built)

**What it is.** Path (ii) of step 4 in the item-4 plan ([model_review.md](model_review.md), item 4). A learned CME detector turns the coronagraph images into CME probabilities, with physical expectations built in as constraints; its outputs are features for the hybrid. It follows path (i) ([coronagraph.md](coronagraph.md)), where raw brightening features helped CME runs by about 3% and hurt the other runs by about 1%.
- model: `src/models/cme_detector.py` (trained with xgboost in dev, evaluated in numpy);
- study: `src/training/cme_detector.py`; analysis: `src/experiments/cme_detector.ipynb`;
- tests: `tests/test_cme_detector.py`.

**Result:** not built. The best set gains 0.21% on the hybrid in 3 of 6 folds, against a bar of 1% and 4 of 6. The image detector works (AUC about 0.80 on unseen years). But its probability of a CME reaching L1 is barely better than chance (AUC 0.49–0.64 for the next 24 h). The images can't tell an Earth-directed CME from one leaving the far side, and a coincident flare is too weak a confirmation. With path (i), this closes the coronagraph route: neither path passes, so the drag-based arrival model (step 5) isn't built.

## Model

**Stage A, the image detector:** per C2 image, P(a CME of width ≥ 270° began in the last 3 h).
- **Inputs:** the image's 12 sector brightenings per ring (`coronagraph.excess`), sorted from largest to smallest, plus the share of brightened outer sectors (25 inputs). Sorting makes the detector independent of the position angle.
- **Labels:** the CDAW catalogue, used for training only. It is published months late, but live runs need only the images.

**Stage B, arrival at L1:** per t0, P(a CME starts at L1 in the next 0–24, 24–48 and 48–72 h).
- **Inputs:** the largest stage-A probability in each past-day bin (0–24 … 72–96 h before t0), and the largest GOES flare within 3 h before each image, binned the same way. A flare marks an on-disk eruption: a front-side CME.
- **Labels:** the CME onsets found in the ACE data (`training.solar_wind_events`): 117 in 2016-10 → 2025-10.

**Physics as constraints, not as a model:**
- every input of both stages is monotone increasing (xgboost `monotone_constraints`): more brightening, a wider one, or a larger flare never lowers a probability;
- stage B sees only images 0–4 days old (the range of CME transit times) in day bins, so it learns the delay rather than assuming one.

**No future data:**
- an image counts for t0 only if taken by t0 − 30 min;
- every probability a model trains on is cross-fitted by year: each training year is scored by a detector fitted on the other training years, as it would be live;
- stage B trains only on t0 < S − 72 h, since its labels look 72 h ahead.

Hyper-parameters were fixed before running (stage A: 300 trees, depth 3; stage B: 200 trees, depth 2).

## Method

- **Variants:** the shipped hybrid's 30 regressors plus
  - the three stage-B probabilities ("arrival"),
  - the four stage-A day-bin maxima ("detector"),
  - or both (seven).
- **Refitting:** the trees are refitted with the shipped settings on the shipped linear model.
- **Folds:** the six yearly walk-forward folds (2019-10 → 2025-10). Score: per-run mean TIMSE / persistence, as a geometric mean over folds. The eval year is not used.
- **By regime:** CME runs and the rest, as in [coronagraph.md](coronagraph.md).
- **Keep rule, set before running:** more than 1% better, winning at least 4 of 6 folds, CME runs better, and the other runs not worse by more than 0.5%.
- **Run:** `python -m training.cme_detector` (about 5 min; needs the LASCO history from `training.pull_lasco_images`).

## Results

| Variant | Score | vs shipped | Folds won | CME runs | Other runs |
|---|---|---|---|---|---|
| hybrid + both (7) | 0.4181 | −0.21% | 3 | −2.1% | +1.1% |
| hybrid + arrival (3) | 0.4187 | −0.08% | 2 | −1.0% | +0.3% |
| **hybrid shipped** | **0.4190** | | | | |
| hybrid + detector (4) | 0.4197 | +0.18% | 3 | −1.6% | +1.5% |

**The stages fold by fold** (AUC on the fold's own images and t0s):

| Fold | Wide CMEs | L1 onsets | Stage A | Stage B 0–24 h | 24–48 h | 48–72 h |
|---|---|---|---|---|---|---|
| 2019-10 | 0 | 7 | (none) | 0.32 | 0.44 | 0.52 |
| 2020-10 | 20 | 12 | 0.79 | 0.63 | 0.62 | 0.59 |
| 2021-10 | 50 | 15 | 0.73 | 0.49 | 0.49 | 0.50 |
| 2022-10 | 81 | 19 | 0.80 | 0.59 | 0.57 | 0.49 |
| 2023-10 | 103 | 28 | 0.80 | 0.64 | 0.64 | 0.58 |
| 2024-10 | 34 | 17 | 0.83 | 0.59 | 0.59 | 0.53 |

- **Stage A works:** about 0.80 on unseen years, against 0.76–0.80 for the raw brightening in the feasibility check.
- **Stage B doesn't:** at most 0.64, near chance in 2021-10, and below chance in 2019-10, a solar-minimum year with no wide CMEs at all, where the 2019 detector trained on 14. Most wide CMEs miss Earth, and 117 L1 onsets in nine years are too few to learn which ones don't.
- **The trade-off of path (i) remains, smaller:** CME runs gain 1–2%, the other runs lose 0.3–1.5%. The arrival probabilities are the quietest feature (+0.3% on other runs) but also gain least.

## Decision

Not built. The coronagraph route closes:
- **paths (i) and (ii) both fall short**, so the drag-based arrival model of step 5 isn't built;
- **kept as the record:** the LASCO connector, the history, `models.cme_detector` and both studies. The forecast doesn't use them, and live runs don't fetch LASCO.

A detector that could tell Earth-directed CMEs apart would need direction information the C2 browse images don't carry: for example, STEREO-A's side view, or on-disk dimming and its location in AIA at full cadence.
