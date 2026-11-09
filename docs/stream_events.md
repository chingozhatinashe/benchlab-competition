# Stream events: arrival, duration and a free speed profile, several additive events (not built)

**The question (user, 2026-11-09).** A follow-up to the [structural model](structural_model.md):
- (i) model each stream event in three parts: an arrival time, a duration, and the speeds over that duration, without assuming a shape;
- (ii) allow several events within the 72 h window, with overlapping effects adding.

The loss also includes the detected stream arrivals (the user's choice), so the arrival rate is anchored apart from the event size.

**Result: not built, stopped after the one-fold pilot (2024).**
- **Every variant is worse than the submitted model,** including as an input to it: +2.5% to +20%.
- **The arrival labels didn't anchor the rate.** The model expects about 5× the arrivals that were detected.
- **Several additive events make it worse** than a single first arrival.

The six-fold run (about 2.3 h) was not run (user decision, 2026-11-09). The Bayesian model ([bayes_stream.md](bayes_stream.md)) followed instead.

- **Model:** `src/models/stream_events.py` (numpy; experimental, not used live).
- **Study:** `python -m training.stream_events` (`--pilot --folds 2024`; about 25 min per fold).
- **Analysis:** `src/experiments/stream_events.ipynb`; results in `src/experiments/stream_events.csv` and `stream_events_params.csv` (2024 fold only).
- **Dev requirement:** PyTorch (CPU), as for the structural model.

## Model

At t0, for leads k = 1…72, where x(u) holds v1's inputs for hour t0 + u:

    v(k) = a + e0 · ψ(k / D_c) + Σ_{u=1..72} λ(u) · A(u) · φ((k − u + L) / (D(u) + L))

| Part | Form |
|---|---|
| Ambient wind a | as in v1: 250–500 km/s from the 27-day q20 and the recurrence |
| Current event | e0 = last speed − a, ending after D_c = 6 + 24·softplus(w · x(1)) h, along a free relaxation curve ψ (piecewise linear on 7 knots, ψ(0) = 1, ψ(1) = 0) |
| Arrival time | Poisson intensity λ(u) = exp(w · x(u)) per hour, so any number of events; the expected effects add |
| Duration | D(u) = 6 + 24·softplus(w · x(u)) h, from the inputs at arrival |
| Speeds over the duration | peak excess A(u) = 100·softplus(w · x(u)) km/s times a free profile φ: piecewise linear on 10 knots, normalised to a maximum of 1, 0 at both ends, starting L = 12 h before the labelled arrival (the labels mark the midpoint of the rise) |

- **Single-event variant:** only the first arrival counts, weighted by (1 − e^{−λ(u)}) · exp(−Σ_{s<u} λ(s)).
- **Training, in one step:**
  - the speed loss (official weights, blend 0.25, despiked, as v1) plus β × the Poisson NLL of the detected arrivals;
  - the label weight is 0 in CME and possible-CME hours, and from 48 h before the fold start on;
  - torch L-BFGS, the 72 × 72 event tensor in chunks of runs.
- **Pilot settings:** chunks of 500 runs, fit on every second fit run (12-hourly), at most 100 iterations. The first pilot, with full settings, was stopped by the system for low memory after one variant (about 17 min per fit).
- **Checks** (`tests/test_stream_events.py`):
  - the piecewise profile;
  - events vanish outside their duration and add when they overlap;
  - numpy/torch parity and chunking;
  - recovery with Poisson labels;
  - the asset round trip.

## Results (2024 fold only)

**Score** (% change in S_final against the submitted model, which scores 107.81; negative = better):

| Variant | vs submitted | Slow wind | Fast stream | Stream arrival | CME |
|---|---|---|---|---|---|
| Events without the current-sheet distance | +2.5% | +8.5% | −7.5% | −0.1% | +7.0% |
| Linear + events (as a regressor) | +3.2% | +6.5% | −7.2% | +1.6% | +8.4% |
| Single event (first arrival only) | +12.2% | +21.0% | −10.0% | +9.8% | +20.0% |
| Events, speed loss only (β = 0) | +18.0% | +34.6% | −4.0% | +10.5% | +25.8% |
| Events with labels (β = 1, the main variant) | +20.2% | +35.8% | +3.6% | +11.0% | +28.2% |

**Fitted quantities:**

| Variant | Mean λ /h | Expected arrivals (observed 372) | Arrival AUC | LL skill vs constant rate | Duration (h) | Peak excess (km/s) |
|---|---|---|---|---|---|---|
| Events with labels | 0.019 | 1935 | 0.65 | −0.39 | 69 | 330 |
| Speed loss only | 0.041 | 4162 | 0.65 | −1.18 | 64 | 190 |
| Single event | 0.019 | 1906 | 0.63 | −0.47 | 75 | 495 |
| Without current-sheet distance | 0.009 | 940 | 0.69 | −0.07 | 70 | 435 |

The detected arrival rate in the fit period is 0.0028 per hour.

**Learned profiles** (with labels):
- **φ:** rises to its peak within the first ~20% of the event, then declines slowly to about 0.4 and rises a little before the end.
- **ψ:** relaxes almost linearly over the remaining duration.

## Why it fails

- **The speed loss swamps the arrival labels.** At β = 1 the label term is about 3% of the loss. The fit raises the rate about 7× above the detected rate and uses the events as a smooth background boost to the speed, not as discrete streams. Out of sample, the arrival-timing likelihood is worse than a constant rate.
- **Additive events pile up.** Durations average about 69 h, so events cover the whole window. Their expected effects add on slow-wind runs, which become 35% worse. The first-arrival-only variant does better (+12%) because its weights can't sum above 1.
- **The free profile doesn't help either.** It learns a plausible shape (a fast rise, a slow decline), but the expected value over many uncertain arrivals smooths it out, as in v1.
- **As an input it adds nothing:** +3.2% on this fold.

## What it suggests

Rate, timing and size need their own likelihood terms and priors, not one speed loss with an added label term. That led to the Bayesian generative model in [bayes_stream.md](bayes_stream.md), where the detected arrivals are data for the count and timing levels, and each event's peak, rise and decay are latent with population priors.
