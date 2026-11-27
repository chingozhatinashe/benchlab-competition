# Staging report 2026-11-27

Generated 2026-11-27 13:02 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-11-27 11:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 28 | 1 | 52.9 | 1.236 | 66.8 | 42.3 | 58.7 | 72.8 | 42.8 | 39.4 |
| bayes-option | bayes | 28 | 1 | 45.8 | 1.070 | 52.6 | 40.3 | 37.1 | 59.4 | 42.8 | 39.4 |
| bayes-option | default | 28 | 1 | 45.6 | 1.065 | 52.4 | 40.1 | 36.8 | 58.9 | 42.8 | 39.4 |
| benchlab | dryrun | 1 | 0 | 0.0 | — | 69.9 | 0.0 | 69.7 | 0.0 | 0.0 | 0.0 |
| benchlab | live | 19 | 0 | 41.5 | 1.127 | 47.1 | 37.2 | 31.5 | 51.3 | 36.8 | 45.1 |
| freeze | baseline | 27 | 1 | 54.1 | 1.377 | 67.5 | 43.6 | 59.6 | 74.8 | 39.3 | 39.3 |
| freeze | bayes | 27 | 1 | 46.0 | 1.173 | 52.9 | 40.4 | 38.6 | 60.5 | 39.3 | 39.3 |
| freeze | default | 27 | 1 | 45.9 | 1.169 | 52.6 | 40.3 | 38.3 | 60.1 | 39.3 | 39.3 |
| linear-v2 | baseline | 115 | 1 | 108.4 | 0.822 | 82.7 | 142.6 | 33.8 | 138.5 | 131.9 | 127.0 |
| linear-v2 | default | 115 | 1 | 91.6 | 0.694 | 72.4 | 117.0 | 30.0 | 124.3 | 131.9 | 127.0 |
| near-miss-a | baseline | 27 | 1 | 54.1 | 1.377 | 67.5 | 43.6 | 59.6 | 74.8 | 39.3 | 39.3 |
| near-miss-a | bayes | 27 | 1 | 46.0 | 1.173 | 52.9 | 40.4 | 38.6 | 60.5 | 39.3 | 39.3 |
| near-miss-a | default | 27 | 1 | 48.0 | 1.223 | 54.3 | 42.7 | 40.0 | 65.5 | 39.3 | 39.3 |
| near-miss-b | baseline | 27 | 1 | 55.3 | 1.410 | 68.4 | 45.2 | 60.5 | 77.6 | 39.3 | 39.3 |
| near-miss-b | bayes | 27 | 1 | 46.9 | 1.195 | 53.6 | 41.4 | 39.2 | 62.2 | 39.3 | 39.3 |
| near-miss-b | default | 27 | 1 | 42.1 | 1.073 | 47.8 | 37.4 | 32.9 | 54.3 | 39.3 | 39.3 |
| near-miss-c | baseline | 26 | 1 | 55.0 | 1.470 | 68.2 | 44.7 | 60.6 | 75.4 | 37.4 | 39.6 |
| near-miss-c | bayes | 26 | 1 | 42.3 | 1.130 | 47.6 | 37.9 | 33.2 | 55.0 | 37.4 | 39.6 |
| near-miss-c | default | 26 | 1 | 42.5 | 1.135 | 47.9 | 38.0 | 33.5 | 54.8 | 37.4 | 39.6 |
| pfss-dhcs | baseline | 66 | 1 | 69.2 | 0.858 | 72.1 | 68.1 | 49.8 | 78.8 | 80.7 | 83.0 |
| pfss-dhcs | default | 66 | 1 | 65.3 | 0.810 | 60.3 | 71.4 | 43.5 | 82.3 | 80.7 | 83.0 |
| pfss-shadow | baseline | 68 | 1 | 84.2 | 0.951 | 76.9 | 94.5 | 42.5 | 86.0 | 88.5 | 92.9 |
| pfss-shadow | default | 68 | 1 | 75.2 | 0.850 | 70.8 | 80.9 | 51.8 | 98.3 | 88.5 | 92.9 |
| pfss-shadow-2 | baseline | 68 | 1 | 84.2 | 0.951 | 76.9 | 94.5 | 42.5 | 86.0 | 88.5 | 92.9 |
| pfss-shadow-2 | default | 68 | 1 | 75.2 | 0.850 | 70.8 | 80.9 | 51.8 | 98.3 | 88.5 | 92.9 |
| pfss-shadow-3 | baseline | 68 | 1 | 84.2 | 0.951 | 76.9 | 94.5 | 42.5 | 86.0 | 88.5 | 92.9 |
| pfss-shadow-3 | default | 68 | 1 | 75.2 | 0.850 | 70.8 | 80.9 | 51.8 | 98.3 | 88.5 | 92.9 |
| submitted | baseline | 19 | 0 | 60.0 | 1.628 | 72.7 | 49.9 | 66.4 | 78.8 | 36.8 | 45.1 |
| submitted | bayes | 19 | 0 | 41.8 | 1.134 | 47.2 | 37.6 | 31.7 | 50.8 | 36.8 | 45.1 |
| submitted | default | 19 | 0 | 41.8 | 1.134 | 47.2 | 37.6 | 31.7 | 50.8 | 36.8 | 45.1 |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 19 | 0 | 59.0 | 1.601 | 72.2 | 48.7 | 65.6 | 78.5 | 36.8 | 45.1 |
| bayes-option | bayes | 19 | 0 | 46.6 | 1.266 | 53.4 | 41.3 | 39.0 | 59.1 | 36.8 | 45.1 |
| bayes-option | default | 19 | 0 | 46.4 | 1.260 | 53.2 | 41.1 | 38.5 | 58.8 | 36.8 | 45.1 |
| benchlab | live | 19 | 0 | 41.5 | 1.127 | 47.1 | 37.2 | 31.5 | 51.3 | 36.8 | 45.1 |
| freeze | baseline | 19 | 0 | 59.0 | 1.601 | 72.2 | 48.7 | 65.6 | 78.5 | 36.8 | 45.1 |
| freeze | bayes | 19 | 0 | 46.6 | 1.266 | 53.4 | 41.3 | 39.0 | 59.1 | 36.8 | 45.1 |
| freeze | default | 19 | 0 | 46.4 | 1.260 | 53.2 | 41.1 | 38.5 | 58.8 | 36.8 | 45.1 |
| linear-v2 | baseline | 19 | 0 | 60.0 | 1.630 | 73.0 | 49.8 | 66.7 | 80.5 | 36.8 | 45.1 |
| linear-v2 | default | 19 | 0 | 58.8 | 1.596 | 68.2 | 51.2 | 56.7 | 82.4 | 36.8 | 45.1 |
| near-miss-a | baseline | 19 | 0 | 59.0 | 1.601 | 72.2 | 48.7 | 65.6 | 78.5 | 36.8 | 45.1 |
| near-miss-a | bayes | 19 | 0 | 46.6 | 1.266 | 53.4 | 41.3 | 39.0 | 59.1 | 36.8 | 45.1 |
| near-miss-a | default | 19 | 0 | 47.6 | 1.293 | 54.3 | 42.4 | 38.8 | 62.6 | 36.8 | 45.1 |
| near-miss-b | baseline | 19 | 0 | 61.0 | 1.657 | 73.3 | 51.3 | 66.8 | 82.7 | 36.8 | 45.1 |
| near-miss-b | bayes | 19 | 0 | 48.1 | 1.307 | 54.4 | 43.2 | 39.7 | 61.9 | 36.8 | 45.1 |
| near-miss-b | default | 19 | 0 | 42.2 | 1.145 | 47.7 | 37.9 | 31.8 | 51.9 | 36.8 | 45.1 |
| near-miss-c | baseline | 19 | 0 | 59.0 | 1.601 | 72.2 | 48.7 | 65.6 | 78.5 | 36.8 | 45.1 |
| near-miss-c | bayes | 19 | 0 | 41.3 | 1.123 | 47.0 | 37.1 | 31.4 | 51.1 | 36.8 | 45.1 |
| near-miss-c | default | 19 | 0 | 41.5 | 1.127 | 47.2 | 37.2 | 31.4 | 50.9 | 36.8 | 45.1 |
| pfss-dhcs | baseline | 19 | 0 | 59.0 | 1.601 | 72.2 | 48.7 | 65.6 | 78.5 | 36.8 | 45.1 |
| pfss-dhcs | default | 19 | 0 | 46.4 | 1.260 | 53.2 | 41.1 | 38.5 | 58.8 | 36.8 | 45.1 |
| pfss-shadow | baseline | 19 | 0 | 59.0 | 1.601 | 72.2 | 48.7 | 65.6 | 78.5 | 36.8 | 45.1 |
| pfss-shadow | default | 19 | 0 | 58.3 | 1.584 | 67.6 | 50.9 | 54.3 | 80.7 | 36.8 | 45.1 |
| pfss-shadow-2 | baseline | 19 | 0 | 59.0 | 1.601 | 72.2 | 48.7 | 65.6 | 78.5 | 36.8 | 45.1 |
| pfss-shadow-2 | default | 19 | 0 | 58.3 | 1.584 | 67.6 | 50.9 | 54.3 | 80.7 | 36.8 | 45.1 |
| pfss-shadow-3 | baseline | 19 | 0 | 59.0 | 1.601 | 72.2 | 48.7 | 65.6 | 78.5 | 36.8 | 45.1 |
| pfss-shadow-3 | default | 19 | 0 | 58.3 | 1.584 | 67.6 | 50.9 | 54.3 | 80.7 | 36.8 | 45.1 |
| submitted | baseline | 19 | 0 | 60.0 | 1.628 | 72.7 | 49.9 | 66.4 | 78.8 | 36.8 | 45.1 |
| submitted | bayes | 19 | 0 | 41.8 | 1.134 | 47.2 | 37.6 | 31.7 | 50.8 | 36.8 | 45.1 |
| submitted | default | 19 | 0 | 41.8 | 1.134 | 47.2 | 37.6 | 31.7 | 50.8 | 36.8 | 45.1 |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 28 | 0 | 0 | 8.0 | 246.7 | 5.5 | — | — | — |
| bayes-option | bayes | 28 | 0 | 0 | 7.0 | 165.1 | 5.5 | 98.4% | — | — |
| bayes-option | default | 28 | 0 | 0 | 9.7 | 247.1 | 5.5 | 98.4% | — | — |
| benchlab | live | 28 | 0 | 0 | 42.5 | 68.0 | 2.7 | 100.0% | — | — |
| freeze | baseline | 28 | 0 | 0 | 6.7 | 9.3 | 5.5 | — | — | — |
| freeze | bayes | 28 | 0 | 0 | 7.1 | 10.3 | 5.5 | 100.0% | — | — |
| freeze | default | 28 | 0 | 0 | 9.7 | 11.6 | 5.5 | 100.0% | — | — |
| linear-v2 | baseline | 28 | 0 | 0 | 7.7 | 246.7 | 5.5 | — | — | — |
| linear-v2 | default | 28 | 0 | 0 | 8.1 | 247.2 | 5.5 | 94.2% | — | — |
| near-miss-a | baseline | 28 | 0 | 0 | 7.0 | 9.5 | 5.5 | — | — | — |
| near-miss-a | bayes | 28 | 0 | 0 | 7.0 | 10.6 | 5.5 | 100.0% | — | — |
| near-miss-a | default | 28 | 0 | 0 | 9.7 | 11.5 | 5.5 | 100.0% | — | — |
| near-miss-b | baseline | 28 | 0 | 0 | 7.2 | 22.4 | 5.5 | — | — | — |
| near-miss-b | bayes | 28 | 0 | 0 | 7.0 | 10.0 | 5.5 | 100.0% | — | — |
| near-miss-b | default | 28 | 0 | 0 | 9.7 | 25.4 | 5.5 | 100.0% | — | — |
| near-miss-c | baseline | 28 | 0 | 0 | 7.5 | 23.9 | 5.5 | — | — | — |
| near-miss-c | bayes | 28 | 0 | 0 | 6.9 | 11.1 | 5.5 | 100.0% | — | — |
| near-miss-c | default | 28 | 0 | 0 | 8.6 | 11.6 | 5.5 | 100.0% | — | — |
| pfss-dhcs | baseline | 28 | 0 | 0 | 7.8 | 9.9 | 5.5 | — | — | — |
| pfss-dhcs | default | 28 | 0 | 0 | 9.8 | 11.5 | 5.5 | 100.0% | — | — |
| pfss-shadow | baseline | 28 | 0 | 0 | 8.0 | 9.6 | 5.5 | — | — | — |
| pfss-shadow | default | 28 | 0 | 0 | 42.8 | 43.1 | 5.5 | 100.0% | pfss ×28 | — |
| pfss-shadow-2 | baseline | 28 | 0 | 0 | 6.7 | 9.3 | 5.5 | — | — | — |
| pfss-shadow-2 | default | 28 | 0 | 0 | 47.0 | 47.7 | 5.5 | 100.0% | pfss ×28 | — |
| pfss-shadow-3 | baseline | 28 | 0 | 0 | 5.2 | 8.2 | 5.5 | — | — | — |
| pfss-shadow-3 | default | 28 | 0 | 0 | 10.4 | 11.6 | 5.5 | 100.0% | — | — |
| submitted | baseline | 28 | 0 | 0 | 7.2 | 246.7 | 5.5 | — | — | — |
| submitted | bayes | 28 | 0 | 0 | 7.0 | 164.9 | 5.5 | 98.9% | — | — |
| submitted | default | 28 | 0 | 0 | 7.6 | 165.7 | 5.5 | 98.9% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 41 | 1 | 0 | 7.8 | 246.7 | 5.5 | — | — | — |
| bayes-option | bayes | 41 | 1 | 0 | 7.0 | 165.1 | 5.5 | 98.9% | — | — |
| bayes-option | default | 41 | 1 | 0 | 9.7 | 247.1 | 5.5 | 98.9% | — | — |
| benchlab | dryrun | 1 | 0 | 0 | 43.0 | 43.0 | 5.6 | 100.0% | — | — |
| benchlab | live | 32 | 0 | 0 | 42.0 | 68.0 | 2.7 | 100.0% | — | — |
| freeze | baseline | 40 | 1 | 0 | 7.1 | 25.9 | 5.5 | — | — | — |
| freeze | bayes | 40 | 1 | 0 | 7.1 | 10.6 | 5.5 | 100.0% | — | — |
| freeze | default | 40 | 1 | 0 | 9.7 | 11.6 | 5.5 | 100.0% | — | — |
| linear-v2 | baseline | 128 | 1 | 0 | 6.8 | 246.7 | 5.5 | — | — | — |
| linear-v2 | default | 128 | 1 | 0 | 7.1 | 247.2 | 5.5 | 98.7% | — | — |
| near-miss-a | baseline | 40 | 1 | 0 | 6.0 | 9.5 | 5.5 | — | — | — |
| near-miss-a | bayes | 40 | 1 | 0 | 7.0 | 10.7 | 5.5 | 100.0% | — | — |
| near-miss-a | default | 40 | 1 | 0 | 9.7 | 11.7 | 5.5 | 100.0% | — | — |
| near-miss-b | baseline | 40 | 1 | 0 | 7.2 | 246.7 | 5.5 | — | — | — |
| near-miss-b | bayes | 40 | 1 | 0 | 7.0 | 165.0 | 5.5 | 98.9% | — | — |
| near-miss-b | default | 40 | 1 | 0 | 9.7 | 247.4 | 5.5 | 99.2% | — | — |
| near-miss-c | baseline | 39 | 1 | 0 | 7.2 | 23.9 | 5.5 | — | — | — |
| near-miss-c | bayes | 39 | 1 | 0 | 7.0 | 11.1 | 5.5 | 100.0% | — | — |
| near-miss-c | default | 39 | 1 | 0 | 9.2 | 11.6 | 5.5 | 100.0% | — | — |
| pfss-dhcs | baseline | 79 | 1 | 0 | 7.0 | 22.4 | 5.5 | — | — | — |
| pfss-dhcs | default | 79 | 1 | 0 | 9.9 | 137.2 | 5.5 | 100.0% | — | — |
| pfss-shadow | baseline | 81 | 1 | 0 | 7.0 | 23.2 | 5.4 | — | — | — |
| pfss-shadow | default | 81 | 1 | 0 | 42.8 | 95.1 | 5.4 | 100.0% | pfss ×80 | — |
| pfss-shadow-2 | baseline | 81 | 1 | 0 | 6.4 | 9.3 | 5.4 | — | — | — |
| pfss-shadow-2 | default | 81 | 1 | 0 | 47.0 | 105.6 | 5.4 | 100.0% | pfss ×80 | — |
| pfss-shadow-3 | baseline | 81 | 1 | 0 | 5.6 | 9.9 | 5.4 | — | — | — |
| pfss-shadow-3 | default | 81 | 1 | 0 | 9.8 | 81.6 | 5.4 | 100.0% | — | — |
| submitted | baseline | 32 | 0 | 0 | 7.4 | 246.7 | 5.5 | — | — | — |
| submitted | bayes | 32 | 0 | 0 | 7.0 | 164.9 | 5.5 | 99.0% | — | — |
| submitted | default | 32 | 0 | 0 | 7.7 | 165.7 | 5.5 | 99.0% | — | — |

## Schedule

Last cycle: 2026-11-27 12:00 UTC
Missed cycles: 2 (10-26 12Z, 10-27 06Z)

## Alerts, last 7 days

| t0 | candidate | variant | kind | detail |
|---|---|---|---|---|
| 20261120T180000Z | submitted | bayes | few regressors | 29/42 observed; missing ['ch_cm_at_0h', 'ch_cm_at_6h', 'ch_cm_at_12h', 'ch_cm_at_18h', 'ch_cm_at_24h', 'ch_cm_at_30h', 'ch_cm_at_36h', 'ch_cm_at_42h', 'ch_cm_at_48h', 'ch_cm_at_54h', 'ch_cm_at_60h', 'ch_cm_at_66h', 'ch_cm_at_72h'] |
| 20261120T180000Z | submitted | default | few regressors | 29/42 observed; missing ['ch_cm_at_0h', 'ch_cm_at_6h', 'ch_cm_at_12h', 'ch_cm_at_18h', 'ch_cm_at_24h', 'ch_cm_at_30h', 'ch_cm_at_36h', 'ch_cm_at_42h', 'ch_cm_at_48h', 'ch_cm_at_54h', 'ch_cm_at_60h', 'ch_cm_at_66h', 'ch_cm_at_72h'] |
| 20261123T060000Z | linear-v2 | default | few regressors | 3/16 observed; missing ['ch_cm_at_0h', 'ch_cm_at_6h', 'ch_cm_at_12h', 'ch_cm_at_18h', 'ch_cm_at_24h', 'ch_cm_at_30h', 'ch_cm_at_36h', 'ch_cm_at_42h', 'ch_cm_at_48h', 'ch_cm_at_54h', 'ch_cm_at_60h', 'ch_cm_at_66h', 'ch_cm_at_72h'] |
| 20261124T000000Z | linear-v2 | default | few regressors | 3/16 observed; missing ['ch_cm_at_0h', 'ch_cm_at_6h', 'ch_cm_at_12h', 'ch_cm_at_18h', 'ch_cm_at_24h', 'ch_cm_at_30h', 'ch_cm_at_36h', 'ch_cm_at_42h', 'ch_cm_at_48h', 'ch_cm_at_54h', 'ch_cm_at_60h', 'ch_cm_at_66h', 'ch_cm_at_72h'] |
| 20261126T060000Z | bayes-option | bayes | few regressors | 16/29 observed; missing ['ch_cm_at_0h', 'ch_cm_at_6h', 'ch_cm_at_12h', 'ch_cm_at_18h', 'ch_cm_at_24h', 'ch_cm_at_30h', 'ch_cm_at_36h', 'ch_cm_at_42h', 'ch_cm_at_48h', 'ch_cm_at_54h', 'ch_cm_at_60h', 'ch_cm_at_66h', 'ch_cm_at_72h'] |
| 20261126T060000Z | bayes-option | default | few regressors | 16/29 observed; missing ['ch_cm_at_0h', 'ch_cm_at_6h', 'ch_cm_at_12h', 'ch_cm_at_18h', 'ch_cm_at_24h', 'ch_cm_at_30h', 'ch_cm_at_36h', 'ch_cm_at_42h', 'ch_cm_at_48h', 'ch_cm_at_54h', 'ch_cm_at_60h', 'ch_cm_at_66h', 'ch_cm_at_72h'] |

## Other events, last 7 days (counts; details in events/)

source not ok: 56, vs staging: 28

