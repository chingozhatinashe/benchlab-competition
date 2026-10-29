# Linear model re-selected on the official metric (shipped model confirmed)

**What it is.** The linear model's variant and regressors were selected on the old per-run TIMSE ([linear_model.md](linear_model.md)). Only its loss weights were re-tuned for the official metric ([official_retune.md](official_retune.md)). This study re-runs the whole selection, scored with the official metric, to see whether it picks something else.
- selection: `python -m training.linear_model select` (`--objective official`, now the default);
- analysis: `src/experiments/official_reselection.ipynb`.

**Result:** it re-selects the shipped model. The only difference, a 1 h instead of a 24 h density window, is 0.31% better (4 of 6 folds), below the 1% bar. **The shipped config stays.**

## Method

- **Score:** the official S_final (`training.competition_score`) against the official truth (`training.pull_ace_1h`), relative to persistence's, as a geometric mean over the six yearly folds (`training.folds.SELECTION`). Each fold fits on the training runs before it. The eval year is not used.
- **Stage 1, the variant:** 12 variants (Gaussian at Box-Cox 1, 0.5, 0, −0.5; gamma; inverse Gaussian; each with and without relative-to-persistence) × 4 ridges. Loss weights `official` at blend 0.25. Regressors: the old fixed shortlist.
- **Stage 2, the regressors:** forward selection over the 117 windowed candidates plus the lead-aligned coronal-hole strip `ch_cm_lead_72h`, which counts as a coronal-hole window. One window per family; stops when a step gains less than 0.5%.
- **Then:** ridge (0, 0.1, 1, 10) × blend (0.1, 0.25, 0.5) on the selected set.
- **Decision rule, as in `official_retune`:** a change ships only if it is more than 1% better than the shipped config and wins at least 4 of 6 folds.
- **Run:** a 500-run pilot (3.5 min, estimate 1.5–2 h), then the full 13,148 runs (about 19 min of compute; stage 1 and the first two steps ran before the process was stopped and resumed from the cache).

## Results

**Stage 1.** The best variant is the shipped one: Gaussian, Box-Cox 1, relative, ridge 0.1 (0.7982). Box-Cox 0.5 is next (0.8000); gamma and inverse Gaussian are 0.7% behind.

**Stage 2:**

| Step | Added | S / persistence | Gain | Shipped model |
|---|---|---|---|---|
| 0 | speed terms only | 0.8631 | | |
| 1 | `ch_cm_lead_72h` | 0.8227 | 4.67% | same |
| 2 | `bt_mean_1h` | 0.8011 | 2.64% | same |
| 3 | `xray_peak_max_72h` | 0.7959 | 0.65% | same |
| 4 | `density_mean_1h` | 0.7901 | 0.72% | `density_mean_24h` |
| 5 | (`icme_mean_3h`) | 0.7885 | 0.20% | stop |

**The density window.** Step 4 scores every density window next to the three shared regressors, with the shipped variant, ridge and blend, so `density_mean_24h` there is the shipped config:

| Window | 1 h | 3 h | 6 h | 12 h | **24 h (shipped)** | 48 h |
|---|---|---|---|---|---|---|
| S / persistence | 0.7901 | 0.7914 | 0.7911 | 0.7922 | **0.7926** | 0.7926 |

**By fold** (S / persistence, refitted on all runs; `src/experiments/official_reselection.csv`):

| Fold | Speed terms only | Shipped | Selected |
|---|---|---|---|
| 2019-10 | 0.8324 | **0.7323** | 0.7326 |
| 2020-10 | 0.7661 | 0.7027 | **0.6958** |
| 2021-10 | 0.8924 | **0.8110** | 0.8133 |
| 2022-10 | 0.9506 | 0.8800 | **0.8776** |
| 2023-10 | 0.9503 | 0.8663 | **0.8613** |
| 2024-10 | 0.8038 | 0.7792 | **0.7766** |
| **Geometric mean** | 0.8631 | 0.7926 | **0.7901** (−0.31%, 4 of 6) |

**Ridge × blend** on the selected set: blend 0.25 with ridge 0.1 is best (0.7901). Blend 0.5 is 1.7% worse, blend 0.1 0.8% worse, and ridge 1 or more is clearly worse.

## Conclusions

- **The old metric didn't distort the selection.** The official metric picks the same variant and the same four regressor families in the same order. The concern in [official_retune.md](official_retune.md) ("features selected on the old metric") is settled.
- **The density window is noise-level:** 1 h vs 24 h is 0.31%, and the windows span 0.25% in all. No change ships.
- **Blend 0.25 and ridge 0.1 are confirmed** on the selected set.
- **The pilot is for timing only.** On 500 runs (about 55 per fold) it picked a fixed 30 h coronal-hole image, density 48 h, pressure and blend 0.5, none of which the full run kept.
- **Code change:** `training.linear_model select` now scores with the official metric by default and writes `*_official*` files, so a selection never overwrites `linear_model_config.json`. `--objective run_timse` reproduces the old selection.
