# Forward-add test for the hybrid's regressors (none kept)

**What it is.** A check that the hybrid's input pruning didn't drop something the trees could use (item 3 of [model_review.md](model_review.md); the concern is in its section 2).
- study: `src/training/hybrid_forward_add.py`;
- analysis: `src/experiments/hybrid_forward_add.ipynb`.

**Result:** nothing kept. The best addition gains 0.27% in 3 of 6 folds, within the trees' refit noise.

## Method

- **What was dropped:** the hybrid's 30 regressors (set C) keep one window per family, chosen by the linear model's single-feature ranking, plus extra coronal-hole lags. All 24 families are in, so set C dropped only the other windows of each family: 90, of which 17 are coronal-hole lags, timed separately in [lead_aligned_ch.md](lead_aligned_ch.md).
- **Candidates:** for each of the 19 families with unused windows, the one furthest (in log hours) from the window in use, for example `speed_mean_1h` next to `speed_mean_24h`. Each is added alone; "+ all" adds all 19.
- **Fitting:** the trees are refitted with the shipped settings on the shipped linear model.
- **Folds:** the six yearly walk-forward folds. Score: per-run mean TIMSE / persistence, as a geometric mean over folds. The eval year is not used.
- **Keep rule, set before running:** more than 0.5% better and winning at least 4 of 6 folds.
- **Run:** `python -m training.hybrid_forward_add` (about 25 min).

## Results

| Added | Score | vs shipped | Folds won |
|---|---|---|---|
| `log_p310_mean_48h` | 0.4241 | −0.27% | 3 |
| `speed_mean_1h` | 0.4244 | −0.21% | 5 |
| `log_t_ratio_mean_48h` | 0.4247 | −0.15% | 3 |
| 4 more | 0.4249–0.4252 | −0.09% to −0.03% | 3–4 |
| **shipped** | **0.4253** | | |
| 9 more | 0.4254–0.4273 | +0.02% to +0.47% | 2–5 |
| `f107_mean_648h` | 0.4257 | +0.10% | 5 |
| `pressure_mean_48h` | 0.4302 | +1.15% | 2 |
| all 19 | 0.4303 | +1.18% | 2 |

- **The 2024 pilot fold misled:** there the 27-day F10.7 mean gained 2.2% and "+ all" 2.6%. The F10.7 mean loses 4.7% in the 2020 fold, the first after the solar minimum, where its level was new to the trees.
- **Adding everything hurts:** with 49 inputs the trees spend splits on redundant windows.
- **Conclusion:** the pruning didn't throw away signal. The hybrid's 30 inputs stay.
