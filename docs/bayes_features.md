# Bayes features: the generative model's fitted quantities as linear-model regressors (not built)

**The question (user, 2026-11-15).** In the [Bayes stream](bayes_stream.md) study, `linear + bayes` gave the submitted linear model only the generative model's blended mean path, and tied the submitted model. Would its fitted parameters do better?

**Status (2026-11-15): not built.** The best variant, the ambient level alone, is 0.26% better than the submitted model and wins 4 of 6 folds; the decision rule needs more than 1%. From the latent-arrival fit, P(arrival by lead) + the mean path is 0.41% better (4 of 6 folds), see [Latent fit](#latent-fit-six-folds-2026-11-16); built as an option, not submitted ([Bayes model](bayes_model.md)).
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

## Latent fit, six folds (2026-11-16)

The same features from the latent-arrival fit (`bayes latent`: arrivals summed out, 300 + 200 draws per chain, every fold converged, R-hat at most 1.015; `--fit latent`), from each fold's cached latent posterior (training.bayes_stream). Results: `src/experiments/bayes_features_latent.csv`.

| added to the linear model | latent fit | folds won | observed fit | folds won |
|---|---|---|---|---|
| parr | **−0.38%** | 4 | −0.08% | 3 |
| mean | −0.23% | 3 | −0.07% | 3 |
| stream | −0.18% | 3 | +0.01% | 3 |
| slow + parr | −0.18% | 4 | +0.23% | 3 |
| ambient | −0.02% | 3 | −0.26% | 4 |
| components | +0.07% | 3 | +0.18% | 3 |
| slow | +0.17% | 2 | +0.33% | 2 |
| forward | parr + mean **−0.41%** | 4 | ambient −0.26% | 4 |

`parr` per fold (S_final against the reference): 2019 −2.3%, 2020 −2.2%, 2021 −0.2%, 2022 +1.8%, 2023 −0.5%, 2024 +1.2% (observed fit: −3.4%, −1.7%, +0.6%, +3.8%, −0.7%, +1.1%). Forward set by regime: fast stream −2.5%, CME −0.7%, stream arrival −0.4%, slow wind −0.3%, intermediate +0.1%.

- **Latent arrivals make the stream signals more useful:** `parr`, `stream` and `mean` all do better from the latent fit, mostly in fast streams; the swing between folds is smaller.
- **The quiet-wind signals go the other way:** the ambient level no longer helps (−0.02%); the latent fit puts more of the variation in streams.
- **Not built under the rule** (−0.41% < 1%), but the best Bayes-derived result: features beat the forecast as a regressor (`linear + bayes latent` −0.23%, docs/bayes_stream.md), and the 2024-only check (forward: slow + ambient −0.44%) chose differently from the six folds.
- **Built anyway as an option (user, 2026-11-16):** the forward set (`parr` + `mean` from a latent fit on all the training data) ships behind `SWP_MODEL=bayes`, not as the submitted model; see [Bayes model](bayes_model.md).

**Possible next step (not planned):** the ambient level is a smooth function of three inputs (q20, rec27 and a constant) through a fitted sigmoid. A cheaper version is that sigmoid term directly in the linear model (Config.terms), without the Bayesian fit; at 0.26% it is unlikely to pass the rule.
