# Metric verification against the organisers' scorer (2026-11-17)

**The question.** The organisers published their scoring code with the starter kit (`scoring_demo/swp_scoring`, "an unmodified copy of what scores the leaderboard"; upstream PR #6, synced in PR #60). Does our implementation of the official metric (`training.competition_score`, [official_score.md](official_score.md)) score exactly as theirs does?

**Result: yes, bit for bit, after two edge-case fixes.** There is one remaining difference, in the persistence rule. It doesn't affect model scores, only the persistence baseline and the fill for failed runs.
- **The metric:** identical on the submitted model's eval year (S_final 117.98389265936824 both ways), on the organisers' demo, and on synthetic cases with ties, missing observations and few runs.
- **Two edge cases now follow their code:** how the quantiles are computed, and a lead whose observations are all equal.
- **The truth:** our parser and our pulled ACE files agree exactly with theirs.
- **The persistence rule differs:** they use the hour stamped t0; we use the hour before. That decision is open (see the end).
- **Study:** `python -m training.metric_verification` (about 1 min). Results: `src/experiments/metric_verification.json`. Tests: `tests/test_metric_verification.py` (skipped without `scoring_demo/`).

**Running their code.** It needs Python 3.11 (`tomllib`); on our 3.10 venv, `tomli` (the same parser, already installed with pytest) is registered as `tomllib`. Their files are not modified.

## 1. The organisers' demo

Their demo window: 356 cycles, Jul–Sep 2026, three baselines (`scoring_demo/data`).

| baseline | expected report | their code, rerun | our code |
|---|---|---|---|
| climatology | 106.455950 | 106.455950 | 106.455950 |
| persistence | 107.768121 | 107.768121 | 107.768121 |
| recurrence 27 d | 93.509428 | 93.509428 | 93.509428 |

- **Their code runs correctly here:** the rerun reproduces their expected report.
- **Our pipeline gives the same scores** on the same forecast files: our truth parser, `verifying` and `score`.
- **The observation matrices are identical.**

## 2. The metric on the same matrices

Both metrics on identical observation and forecast matrices:

| case | S_final, ours − theirs (relative) | before the fixes |
|---|---|---|
| submitted model, eval year (1,448 runs) | 0 | 3 × 10⁻⁹ |
| continuous synthetic speeds | 0 | 4 × 10⁻⁸ |
| 0.1 km/s grid (ACE's resolution) | 0 | 2 × 10⁻⁸ |
| 1 km/s grid (ties) | 0 | 2 × 10⁻⁷ |
| 10 km/s grid (many ties), 300 runs | 0 | 0 |
| 12 runs | 0 | 0 |
| one lead with all-equal observations | 0 | **1.4%** |

**The two fixes** in `training.competition_score`:
1. **Quantiles.** We computed the quantile positions in integer arithmetic, so that a quantile landing exactly on an observation stays on it. Their `np.quantile` works in floating point and can land a hair above, leaving that observation out of v ≥ q_k: 8 of the 101 quantiles of evenly spaced data do. We now call `np.quantile` as they do. The effect on real scores is about 10⁻⁹.
2. **All-equal observations.** We gave such a lead TIMSE NaN, which dropped it from S_final. They give TIMSE 0, so S = 0 and the lead still counts. We now do the same. This matters only with very few runs, such as the first days of the competition or a short staging window.

None of this changes a model choice: every reported score moves by less than 10⁻⁸.

## 3. The truth

- **Parsing:** on the demo's four monthly files, our parser (`competition_score.parse_monthly`) and theirs (`swp_scoring/sources/ace_archive.py`) give the same valid hours and speeds. Both treat status ≠ 0 and −9999.9 as missing, keep the last row of a duplicated timestamp, and label each hour by its start.
- **Our pulled files:** our truth (`training.pull_ace_1h`, `src/data/ace_1h`) matches their archive snapshot exactly: 2,022 valid hours, none in one only, no value differs.

## 4. Persistence: the one remaining difference

| | our rule | their rule |
|---|---|---|
| persistence is | the latest valid hour labelled ≤ t0 − 1 h | the latest valid hour labelled ≤ t0, within 72 h |
| eval-year runs where the two differ | — | 1,119 of 1,448 (mean 9.9 km/s) |
| persistence S_final, eval year | 157.28 | **155.70** (1.0% better) |
| the submitted model with 10% of runs failed (scored as persistence) | 121.40 | 121.25 |
| the submitted model | 117.98 | 117.98 |

- **What their rule does:** the hour labelled t0 averages t0 to t0 + 59 min. So their persistence uses up to an hour after t0, which our rule avoids, following the spec's "no observation after t0". It is their definition, and it is what the leaderboard uses.
- **Model scores aren't affected:** persistence only fills failed runs, and our models never see it.
- **What it does change:**
  - **The persistence baseline score:** 1.0% better under their rule.
  - **The scores of failed runs:** 0.1% on the example above.
  - **Every ratio to persistence we report** ("0.78 of persistence"): their persistence is better, so our models' ratios would rise by about 1%.

  Model choices compare variants against the same persistence, so none would change.

**Decision (user, 2026-11-17): keep our rule.** Persistence stays the latest valid hour labelled ≤ t0 − 1 h (`competition_score.PERSISTENCE_LAG`). That follows the spec's "no observation after t0", and it keeps every reported ratio comparable with the earlier ones.
- **Where staging differs from the leaderboard:** in persistence scores, and in failed runs (scored as persistence). Under the organisers' rule both would score about 1% better.
- **Model scores are unaffected.**
