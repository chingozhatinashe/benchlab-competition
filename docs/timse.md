# TIMSE: Threshold-Integrated Mean Squared Error

TIMSE is the official ranking metric for the Benchlab Solar Wind Prediction Tournament. Lower is better.
The implementation is `timse()` in `src/training/metrics.py`.

## Definition

Source: Benchlab participant briefing, 1 September 2026, slide 29. The PDF is kept locally as `docs/Briefing.pdf` and is not committed.

Given observed speeds vᵢ and forecasts v̂ᵢ (km/s):

1. **Thresholds.** v\*₁ ≤ v\*₂ ≤ … ≤ v\*ₘ are the observed speeds, sorted.
2. **Tail MSE.** For each threshold, Sⱼ = { i : vᵢ ≥ v\*ⱼ } and

   Mⱼ = (1 / Nⱼ) · Σ_{i ∈ Sⱼ} (vᵢ − v̂ᵢ)², where Nⱼ = |Sⱼ|.

3. **Integrate across thresholds.**

   TIMSE = Σ_{j=1}^{m−1} (v\*_{j+1} − v\*ⱼ) · Mⱼ

The briefing describes it as measuring squared error across the whole speed distribution, giving more weight to rarer high-speed conditions, with weights derived from the observed distribution rather than chosen by hand.

## Clarifications not in the briefing

From the organisers:

- **Thresholds are the sorted observed values.** They are not a fixed grid or quantiles.
- **Missing observations are not thresholds.** *This may change.* In the code, a missing observation (NaN) is dropped from the error sets as well, since it has no error. The rule lives in one function, `_valid_pairs`.

## Implementation notes

- **Ties.** A repeated observed value adds a zero-width gap, so de-duplicating thresholds would not change the score. Sⱼ includes every observation equal to v\*ⱼ.
- **Top threshold.** v\*ₘ only closes the last gap. Mₘ is never used.
- **Units and comparability.** The result is in (km/s)³, and it scales with the spread of observed speeds. Scores are only comparable over the same set of observations, such as the same validation window.
- **Pooling.** Inputs are flattened, so a (runs × 72 lead hours) block is scored as one pooled set. How Benchlab pools runs and lead times for the leaderboard is not stated in the briefing.
- **Edge cases.** No observations gives NaN. A single observation, or all observations equal, gives 0.0 because the sum has no non-zero gaps. A missing forecast where an observation exists raises `ValueError`.
