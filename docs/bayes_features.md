# Bayes features: the generative model's fitted quantities as linear-model regressors (not built)

**The question (user, 2026-11-15).** In the [Bayes stream](bayes_stream.md) study, `linear + bayes` gave the submitted linear model only the generative model's blended mean path, and tied the submitted model. Would its fitted parameters do better?

**Status (2026-11-15): not built.** The best variant, the ambient level alone, is 0.26% better than the submitted model and wins 4 of 6 folds; the decision rule needs more than 1%.
- **Study:** `python -m training.bayes_features` (`--pilot --folds 2024`; `--summary`).
- **Tests:** `tests/test_bayes_features.py`.
- **Analysis:** `src/experiments/bayes_features.ipynb`; results in `src/experiments/bayes_features.csv`.

## From parameters to features

The fitted global parameters are the same for every run of a fold (one posterior per fold), so as regressors they would be constant columns, absorbed by the intercept. What varies run by run is what those parameters imply from each run's inputs. Per run, averaged over 32 posterior draws of the fold's cached additive fit (no refit):

| Signal | Definition | Carries |
|---|---|---|
| `ambient` | a = 250 + 250 sigmoid(w_a · [1, q20, rec27]), every lead | the level the slow wind relaxes towards |
| `slow` | a + ρᵏ(v0 − a) | the slow-wind path from the current speed |
| `parr` | P(N ≥ 1) × P(first arrival ≤ k) | stream timing without the amplitude |
| `stream` | Σᵤ P(N ≥ 1) πᵤ E(x(u)) profile(k; u), one arrival, E and D at their population medians given x(u) | the expected stream excess |
| `mean` | slow + stream | the additive model's mean path (the `linear + bayes` signal without simulation noise) |

Each is a lead-aligned regressor `structural_bayes_<name>_lead_0h` of the submitted config: lead k reads the column for its 6-hour block, as `structural_speed` in the [structural model](structural_model.md). The fit runs' features come from the same fold's posterior, which is fitted on the runs before the fold.

Per-event marks (each event's own excess and decay) are not usable: they exist only for training events with detected arrivals, not at forecast time.

## Variants and decision rule (set before running)

- Each signal alone; `slow + parr` (the quiet-wind part and the timing, without amplitudes); `components` (ambient, slow, parr, stream).
- `forward`: greedy forward selection over the five signals by the geometric mean over the six folds.
- Keep if more than 1% better than the submitted model and winning at least 4 of 6 folds. The eval year is not used.

## 2024 pilot

151 s (features 136 s, fits 30 s), peak memory 0.8 GB. The forward selection on that fold alone picked `ambient + slow` (−0.78%); on six folds it stops after `ambient`.

## Six folds (2026-11-15)

About 2 min per fold.

| Variant | vs reference (geometric mean) | Folds won | slow wind | intermediate | fast stream | stream arrival | CME |
|---|---|---|---|---|---|---|---|
| ambient (= forward) | −0.26% | 4 | −0.50% | −0.06% | +0.65% | −0.59% | −0.29% |
| parr | −0.08% | 3 | +1.41% | +0.45% | −0.43% | −0.34% | −0.46% |
| mean | −0.07% | 3 | +0.01% | −0.14% | −0.05% | −0.13% | −0.05% |
| stream | +0.01% | 3 | +1.50% | +0.35% | −0.49% | −0.16% | −0.29% |
| components | +0.18% | 3 | +2.19% | +0.33% | +1.83% | −0.41% | −0.48% |
| slow + parr | +0.23% | 3 | +2.23% | +0.26% | +0.66% | −0.06% | −0.35% |
| slow | +0.33% | 2 | +0.97% | −0.06% | +1.18% | +0.16% | +0.20% |

S_final per fold (the percentages against the reference):

| Fold | reference | ambient | slow | parr | stream | mean | slow + parr | components |
|---|---|---|---|---|---|---|---|---|
| 2019 | 60.01 | 59.41 (−1.0%) | 60.80 (+1.3%) | 57.97 (−3.4%) | 58.01 (−3.3%) | 59.87 (−0.2%) | 58.76 (−2.1%) | 58.19 (−3.0%) |
| 2020 | 68.86 | 68.21 (−0.9%) | 69.13 (+0.4%) | 67.67 (−1.7%) | 67.68 (−1.7%) | 68.65 (−0.3%) | 67.88 (−1.4%) | 67.33 (−2.2%) |
| 2021 | 84.93 | 85.25 (+0.4%) | 85.16 (+0.3%) | 85.44 (+0.6%) | 85.57 (+0.7%) | 84.89 (−0.1%) | 85.57 (+0.8%) | 86.22 (+1.5%) |
| 2022 | 88.89 | 88.83 (−0.1%) | 89.11 (+0.2%) | 92.29 (+3.8%) | 92.34 (+3.9%) | 88.96 (+0.1%) | 92.59 (+4.2%) | 92.72 (+4.3%) |
| 2023 | 111.38 | 111.91 (+0.5%) | 111.34 (−0.0%) | 110.62 (−0.7%) | 110.27 (−1.0%) | 111.48 (+0.1%) | 110.45 (−0.8%) | 110.82 (−0.5%) |
| 2024 | 107.81 | 107.35 (−0.4%) | 107.60 (−0.2%) | 108.97 (+1.1%) | 109.55 (+1.6%) | 107.83 (+0.0%) | 108.81 (+0.9%) | 109.08 (+1.2%) |

- **The ambient level is the only signal that helps, and only a little:** −0.26%, 4 of 6 folds. The forward selection adds nothing after it.
- **The slow-wind path doesn't help on its own** (+0.33%), probably because the linear model already has the current speed and its recurrence terms, which carry the relaxation.
- **The stream signals swing with the fold:** `parr` is 3.4% better on 2019 and 3.8% worse on 2022, `stream` similar. They help at stream arrivals and fast streams but cost in slow wind (+1.4–1.5%), so on average they cancel. The generative model's arrival skill also falls over the folds (AUC 0.83 on 2019, 0.53 on 2023; see the Bayes stream notebook).
- **Splitting the path beats the blended path only through the ambient level** (−0.26% against −0.07% for `mean`).

## Latent fit, 2024 (2026-11-16)

The same features from the latent-arrival fit (`bayes latent`: arrivals summed out, 300 + 200 draws per chain, R-hat 1.005; `--fit latent`), against the observed-arrival fit's, on the 2024 fold only. The other folds would need a latent refit each (~4 h), and were not run. Results: `src/experiments/bayes_features_latent.csv`.

| added to the linear model | latent fit | observed fit |
|---|---|---|
| ambient | −0.10% | −0.43% |
| slow | −0.31% | −0.19% |
| parr | +1.19% | +1.08% |
| stream | +1.51% | +1.61% |
| mean | +0.28% | +0.02% |
| slow + parr | +1.00% | +0.93% |
| components | +1.38% | +1.18% |
| forward | slow + ambient −0.44% | ambient −0.43% |

- **Latent arrivals don't make the stream signals useful:** `parr` and `stream` still cost 1–1.5%, better at stream arrivals but 3–5% worse in slow and intermediate wind.
- **Only the quiet-wind part helps, as with the observed fit:** the best set reaches the same −0.4% on this fold, where the observed fit's ambient level gave −0.26% over six folds.
- **Features beat the forecast as a regressor:** the latent mean path as one regressor (`linear + bayes latent`, docs/bayes_stream.md) was +0.32% on 2024.

**Possible next step (not planned):** the ambient level is a smooth function of three inputs (q20, rec27 and a constant) through a fitted sigmoid. A cheaper version is that sigmoid term directly in the linear model (Config.terms), without the Bayesian fit; at 0.26% it is unlikely to pass the rule.
