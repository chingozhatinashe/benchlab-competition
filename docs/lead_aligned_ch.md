# Lead-aligned coronal-hole features (built)

**What it is.** The coronal-hole input timed against each forecast hour instead of against t0
(item 3 of [model_review.md](model_review.md), after Collin et al. 2025).
- fold study and the stitched model: `src/training/lead_aligned_ch.py`;
- analysis: `src/experiments/lead_aligned_ch.ipynb`;
- tests: `tests/test_lead_aligned_ch.py`.

**Result:** the linear model gains 2.7% and the hybrid 1.7% on the folds, each winning 5 of 6. Built on 2026-10-24: the shipped linear model uses `ch_cm_lead_72h` ([linear_model.md](linear_model.md), data fix 4). After retraining, the six-fold scores are linear 0.4256 and hybrid 0.4190.

## Method

- **Now:** the linear model uses the central-meridian strip (`ch_cm`) of one image, 24 h before t0, for every lead. For lead k that image is 25 + k hours older than the hour being forecast: 26 h at lead 1, 97 h at lead 72.
- **Lead-aligned at delay D:** lead k uses the strip of the image at lag L_k = D − k before t0, rounded up to the 6-hour image grid and at least 0. With the 1-hour image latency, the image is D + 1 to D + 6 hours older than the forecast hour.
- **No model change for the test.** The linear model fits each lead on its own, so a model whose lead k uses the column at L_k equals, lead by lead, one fitted with that single column. `Stitched` fits one `LinearModel` per distinct L_k (13 for D = 72 h) and takes each lead from its model. It has the interface the hybrid needs, so the hybrid's trees can be refitted on top.
- **Variants** on the shipped config (data fixes and the regressor cap included):
  - `at D`: `ch_cm_at_24h` replaced by the lead-aligned strip, D = 60 … 120 h;
  - `+ at D`: the lead-aligned strip added to `ch_cm_at_24h`;
  - 24-hour window means of the strip: tried on the 2024 pilot fold only, where they were worse (+1% to +6%).
- **Folds:** the six yearly walk-forward folds (2019-10 → 2025-10). Score: per-run mean TIMSE / persistence, as a geometric mean over folds. The eval year is not used.
- **Run:** `python -m training.lead_aligned_ch` (about 25 min for the linear variants), then `--hybrid "at 72" "at 78" "+ at 72"` (about 15 min).

## Results

**Linear model, by delay:**

| D (h) | 60 | 66 | **72** | 78 | 84 | 90 | 96 | 108 | 120 |
|---|---|---|---|---|---|---|---|---|---|
| replace, vs shipped | −0.9% | −2.0% | **−2.7%** | −2.4% | −0.9% | +1.1% | +3.1% | +6.2% | +7.1% |
| folds won | 4 | 5 | **5** | 5 | 3 | 2 | 0 | 0 | 0 |
| add, vs shipped | −1.8% | −2.2% | −2.4% | −2.2% | −1.5% | −0.8% | −0.5% | −0.2% | −0.4% |

**Best variants, both models:**

| | Score | vs shipped | Folds won |
|---|---|---|---|
| linear, shipped | 0.4375 | | |
| **linear, at 72** | **0.4257** | **−2.7%** | 5 |
| linear, at 78 | 0.4269 | −2.4% | 5 |
| linear, + at 72 | 0.4271 | −2.4% | 4 |
| hybrid, shipped | 0.4253 | | |
| **hybrid, at 72** | **0.4181** | **−1.7%** | 5 |
| hybrid, at 78 | 0.4214 | −0.9% | 2 |
| hybrid, + at 72 | 0.4202 | −1.2% | 5 |

- **The fold that doesn't gain:** the hybrid at 72 h loses the 2024 fold (+2.3%); the linear model is level there.
- **Selection:** D was chosen among 11 values on the same folds, which flatters the winner a little. The smooth curve and 5 of 6 folds make noise unlikely.

## Interpretation

- **72 h is shorter than the slow-wind travel time** of 4–5 days. The strip matters most for fast streams, whose wind takes about 2.5–3 days, and the actual image delay is 73–78 h. Collin et al.'s 4–7 days are measured to the target day in OMNI daily means, a coarser timing.
- **The gain is at the far leads.** At lead 48 both models use the same image. At lead 72 the shipped image is 97 h older than the forecast hour, the aligned one 73 h, and its coefficient stays high: 23.5 against 13.7 km/s per standard deviation (2024 fold).
- **Replacing beats adding:** the fixed-lag strip adds nothing once the aligned one is in.

## To build it

1. **Per-lead regressors in `LinearModel`:** a lead-aligned feature (for example `ch_cm_lead_72h`) takes the column at L_k for lead k. This is one fit instead of 13, and the submodel without the coronal hole stays.
2. **Live:** fetch the images at lags 0, 6, …, 72 h (13 instead of the 4 the hybrid needs now), each a small 512-pixel browse JPEG, with the same fallbacks to older images. Check the time and MB with `inference.check_features`.
3. **Retrain** the linear model and the hybrid's trees on it. Optionally give the trees the aligned strip per row as well.
