# Gaussian-process correction (not built)

**What it is.** A Gaussian process that corrects the linear model's forecast, as an alternative to
the XGBoost hybrid's trees:
- model: `src/models/gaussian_process.py` (forecasting is numpy only; experimental, not used live);
- 4-year selection: `src/training/gaussian_process.py`;
- 9-year fold study: `src/training/gaussian_process_folds.py`;
- analysis: `src/experiments/gaussian_process.ipynb`.

**Result:** not built. On the six yearly folds, the best version gains 1.6% over the linear model,
but it stays 1.4% behind the shipped XGBoost hybrid. The version fitted as designed, with
hyper-parameters by marginal likelihood, overfits and is 29–51% worse than the linear model.

## Model

Two versions, both on top of the shipped linear model:

| Version | Forecast | Hyper-parameters |
|---|---|---|
| **ml2** (`models.gaussian_process`, as designed) | linear(k) + Σ_j c_j(x) φ_j(k) | maximum marginal likelihood (ML-II), on 1,500 runs |
| **offset** | linear(k) + c(x), the same at every lead | an isotropic lengthscale and a noise/signal ratio, chosen on the folds |

- **ml2's correction:**
  - the φ_j are the leading principal shapes (3 used) of the training residual trajectories (observed − linear, 72 leads), from a PCA weighted by each run's TIMSE weights;
  - each run's residual becomes scores on those shapes;
  - one zero-mean GP per score, Matérn 5/2 with one lengthscale per input (ARD).
- **offset's correction:** c is a zero-mean GP on each run's TIMSE-weighted mean offset.
- **Inputs x:** standardised regressors, plus the last observed speed and the linear model's mean offset from persistence. Missing values take the training median.
  - set **A**: the linear model's 4 regressors;
  - set **H**: the hybrid's regressors.
- **Weights:** each run's noise variance is σ_n² / w, with w its mean TIMSE weight, the GP counterpart of the TIMSE-weighted loss.
- **Zero means everywhere:** far from the training runs, the forecast is the linear model's.

## First attempt: 4 years, two folds (2026-10-22)

The 4-year selection (`python -m training.gaussian_process select`) used the two old folds
(2023-10, 2024-10), the 4-year linear anchor (fold score 0.4614) and the 4-year hybrid's 25
regressors.
- **ml2:** all sets and component counts scored about 0.63, far worse than the linear model.
- **offset:** a lengthscale × noise grid scored by hand reached 0.4587 (set H, lengthscale 10, noise/signal 100), a 0.6% gain. The 4-year hybrid gained 1.9%.
- **Decision:** the GP code was kept and nothing was built.

## Nine years, six folds (2026-10-23)

`python -m training.gaussian_process_folds` re-scores both versions on the current setup:
- 9 years of history;
- the six yearly folds (2019-10 → 2025-10);
- the shipped 9-year linear model as anchor, and the hybrid's 30 regressors as set H.

| | Fold score | vs linear | vs hybrid | Folds won vs linear |
|---|---|---|---|---|
| hybrid (shipped, refitted per fold) | 0.4258 | −2.9% | | 6 |
| offset, set H, lengthscale 10, noise/signal 10 | 0.4318 | −1.6% | +1.4% | 4 |
| offset, set A, lengthscale 10, noise/signal 1 | 0.4395 | +0.2% | +3.2% | 3 |
| linear (shipped, refitted per fold) | 0.4387 | | +3.0% | |
| ml2, set A (2019 and 2024 folds) | 0.500 | +29% | | 0 |
| ml2, set H (2019 and 2024 folds) | 0.583 | +51% | | 0 |

- **Grid:** lengthscale (1, 3, 10, 30) × noise/signal (1, 10, 100) × set (A, H). The two best settings were picked on these same folds.
- **Memory:** the GP conditions on 4,000 random 6-hourly fit runs per fold. The full set, up to 11,000 runs, needs more memory than the dev laptop has. The linear anchor uses every fit run.
- **ml2 was scored on two folds only;** its hyper-parameter search takes about 7 minutes per fold.

## Why it fails

- **ML-II overfits.** The marginal likelihood drives the noise of the second and third shapes almost to zero (noise/signal below 10⁻⁴). With set A it also picks lengthscales as short as 0.05. The GP nearly interpolates the training runs, storms included, because the TIMSE weights make a few storm runs almost noise-free. It then adds large wrong corrections on new runs. Likelihood is the wrong objective for a TIMSE-scored correction.
- **The offset GP is too weak:**
  - **Size:** a mean of +4 km/s, 90% of runs between −13 and +25 km/s, against training offsets with a 62 km/s spread.
  - **Effect,** pooled TIMSE vs the linear model on the 2024 fold: better on stream arrivals (0.94) and CMEs (0.96), worse on slow wind (1.16) and intermediate wind (1.09).
  - **The hybrid does more:** its trees make the same kind of correction, but nonlinear and lead-dependent, and they win in every fold.

## Decision

- **Not built.** No runtime change.
- **The code is kept for reuse.**
- **A GP would be worth revisiting** only for something the trees can't give, such as calibrated forecast uncertainty, which the competition doesn't score. A sparse (inducing-point) GP could use every fit run.

```bash
XGB_NTHREAD=2 python -m training.gaussian_process_folds --pilot --folds 2024   # one fold, timed
XGB_NTHREAD=2 python -m training.gaussian_process_folds --max-seconds 240      # all folds, resumable
python -m training.gaussian_process_folds --summary
```

## Re-run on the official metric (2026-10-29)

The offset GP was the only re-run candidate with an earlier gain (−1.6% over the old linear model). It was re-scored with the official metric on the re-tuned linear model (`python -m training.gaussian_process_folds`, `--metric official`, now the default; analysis `src/experiments/gaussian_process_official.ipynb`).
- **Anchor:** the submitted linear model (official loss weights, blend 0.25), refitted per fold. Its weights also weight each run's offset.
- **Versions:** the offset GP only, on the same grid (sets A and H × lengthscale 1, 3, 10, 30 × noise/signal 1, 10, 100) and the same 4,000 fit runs per fold. ml2 is left out: it overfits and takes about 7 minutes per fold. The hybrid is no longer a reference.
- **Score:** the official S_final against the official truth, on the six yearly folds. Decision rule, set before running: more than 1% better than the linear model, winning at least 4 of 6 folds.

| | S_final (km/s) | vs linear | Folds won | Mean correction (km/s) |
|---|---|---|---|---|
| offset, set H, lengthscale 3, noise/signal 100 | **88.27** | −0.34% | 3 | −1.3 |
| offset, set A, lengthscale 10, noise/signal 100 | 88.31 | −0.37% | 3 | −0.8 |
| offset, set A, lengthscale 30, noise/signal 100 | 88.33 | −0.24% | 5 | +0.0 |
| offset, set A, lengthscale 3, noise/signal 100 | 88.35 | −0.38% | 3 | −2.0 |
| **linear (submitted, refitted per fold)** | 88.49 | | | |
| the other 20 settings | 88.35–91.13 | −0.33% to +3.04% | 0–5 | |

(The ranking by mean S_final and by the geometric-mean ratio differ slightly, because the folds' scores differ in size.)

**Result: still not built.** No setting comes near the bar: the best is −0.38%, with 3 of 6 folds.
- **The best setting changes from fold to fold.** On their own fold, the winners gain −3.5% (2019-10), −2.2% (2020-10) and −1.8% (2023-10), but each is a different setting. Picking each fold's best in hindsight, which can't be done live, still gains only −1.4%.
- **The corrections are tiny:** the best settings shift the forecast by −2 to +1 km/s on average, against the anchor's +22 km/s mean bias.
- **The re-tune absorbed the GP's old gain.** On the old metric the GP mostly corrected the old linear model's run-level offset. The official-weights linear model is already less biased, and nothing systematic is left for a smooth correction to learn.
