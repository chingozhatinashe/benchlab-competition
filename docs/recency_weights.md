# Recency sample weights (not built)

**What it is.** Sample weights that make the linear model and the hybrid lean on the recent
training years, in case the input–speed relationships drift with the solar cycle (item 2e of
[model_review.md](model_review.md), after Collin et al. 2025).
- weights and fold study: `src/training/recency_weights.py`;
- the hook: `run_weights` in `LinearModel.fit` and `XGBoostModel.fit` (default: none, so the shipped models are unchanged);
- analysis: `src/experiments/recency_weights.ipynb`.

**Result:** not built. Worse at every half-life, for both models.

## Method

- **Weights:** each training run gets 0.5^(age / half-life), with age = years from the run to the end of the training data (the fold start). They multiply the TIMSE weights and are normalised to mean 1, so the trees' `min_child_weight` keeps its meaning.
- **Half-lives:** ∞ (the shipped model), 8, 4, 2 and 1 years.
- **Fitting:** per fold, the linear model (shipped config, data fixes and the regressor cap) and the hybrid's trees on top are refitted on the data before the fold, both with the same half-life. "Hybrid, trees only" weights only the trees, on the unweighted linear model.
- **Folds:** the six yearly walk-forward folds (2019-10 → 2025-10). Score: per-run mean TIMSE / persistence, as a geometric mean over folds. The eval year is not used.
- **Run:** `python -m training.recency_weights` (about 6 min; the 2024 fold alone is 100 s).

## Results

| Half-life | Linear | vs ∞ | Hybrid | vs ∞ | Hybrid, trees only, vs ∞ | Folds won (linear / hybrid) | Effective years |
|---|---|---|---|---|---|---|---|
| **∞ (shipped)** | **0.4375** | | **0.4253** | | | | 5.5 |
| 8 years | 0.4388 | +0.3% | 0.4276 | +0.5% | +0.7% | 3 / 2 | 5.4 |
| 4 years | 0.4404 | +0.6% | 0.4299 | +1.1% | +1.1% | 3 / 1 | 5.0 |
| 2 years | 0.4443 | +1.6% | 0.4355 | +2.4% | +1.4% | 3 / 2 | 4.1 |
| 1 year | 0.4531 | +3.6% | 0.4485 | +5.4% | +3.3% | 2 / 2 | 2.7 |

Effective years: the Kish effective sample size of the weights, averaged over folds.

## Why it fails

- **The folds disagree.** The 2019 fold gains steadily (linear −3.7% at a 1-year half-life), and the 2023 fold a little. The 2021 and 2024 folds lose heavily, by up to 8.5% and 18%.
- **The activity level explains it only partly.** In 2019 the fold year's mean F10.7 matches the two years before it, and recency helps; in 2021 activity jumped by 56%, and it hurts. But 2024 also matches its recent years (ratio 0.98) and loses the most. A cycle-phase similarity weight based on the F10.7 level would not have rescued that fold.
- **Losing data costs more than the drift gains.** It matches the history comparison fold by fold ([evaluation.md](evaluation.md)): the 2024 fold gained 12% from adding the 2016–2020 years and is the one recency hurts most; on the 2023 fold the old years didn't help, and recency gains slightly there.
