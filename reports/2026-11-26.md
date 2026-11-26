# Staging report 2026-11-26

Generated 2026-11-26 13:05 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-11-26 11:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 24 | 1 | 52.0 | 1.181 | 67.1 | 40.9 | 58.2 | 71.3 | 44.0 | 38.5 |
| bayes-option | bayes | 24 | 1 | 46.2 | 1.049 | 54.5 | 39.7 | 41.6 | 59.1 | 44.0 | 38.5 |
| bayes-option | default | 24 | 1 | 45.9 | 1.042 | 54.1 | 39.5 | 41.2 | 58.4 | 44.0 | 38.5 |
| benchlab | dryrun | 1 | 0 | 0.0 | — | 69.9 | 0.0 | 69.7 | 0.0 | 0.0 | 0.0 |
| benchlab | live | 15 | 0 | 42.1 | 1.109 | 49.3 | 36.7 | 37.4 | 50.6 | 38.0 | 44.9 |
| freeze | baseline | 23 | 1 | 53.2 | 1.321 | 67.9 | 42.2 | 59.4 | 73.3 | 40.3 | 38.3 |
| freeze | bayes | 23 | 1 | 46.5 | 1.155 | 54.9 | 40.0 | 43.6 | 60.4 | 40.3 | 38.3 |
| freeze | default | 23 | 1 | 46.2 | 1.148 | 54.4 | 39.7 | 43.2 | 59.7 | 40.3 | 38.3 |
| linear-v2 | baseline | 111 | 1 | 108.7 | 0.817 | 83.2 | 142.5 | 32.8 | 138.7 | 133.1 | 127.7 |
| linear-v2 | default | 111 | 1 | 92.3 | 0.693 | 73.3 | 117.2 | 30.4 | 125.0 | 133.1 | 127.7 |
| near-miss-a | baseline | 23 | 1 | 53.2 | 1.321 | 67.9 | 42.2 | 59.4 | 73.3 | 40.3 | 38.3 |
| near-miss-a | bayes | 23 | 1 | 46.5 | 1.155 | 54.9 | 40.0 | 43.6 | 60.4 | 40.3 | 38.3 |
| near-miss-a | default | 23 | 1 | 48.5 | 1.205 | 55.9 | 42.6 | 44.5 | 65.0 | 40.3 | 38.3 |
| near-miss-b | baseline | 23 | 1 | 54.7 | 1.358 | 68.9 | 44.0 | 60.4 | 76.5 | 40.3 | 38.3 |
| near-miss-b | bayes | 23 | 1 | 47.5 | 1.179 | 55.6 | 41.1 | 44.2 | 62.4 | 40.3 | 38.3 |
| near-miss-b | default | 23 | 1 | 42.7 | 1.060 | 49.3 | 37.3 | 37.1 | 53.9 | 40.3 | 38.3 |
| near-miss-c | baseline | 22 | 1 | 54.3 | 1.414 | 68.7 | 43.4 | 60.5 | 73.9 | 38.4 | 38.7 |
| near-miss-c | bayes | 22 | 1 | 43.0 | 1.122 | 49.3 | 37.9 | 37.7 | 54.9 | 38.4 | 38.7 |
| near-miss-c | default | 22 | 1 | 43.1 | 1.124 | 49.4 | 38.0 | 37.9 | 54.5 | 38.4 | 38.7 |
| pfss-dhcs | baseline | 62 | 1 | 69.4 | 0.846 | 72.4 | 68.1 | 49.0 | 78.6 | 82.0 | 83.8 |
| pfss-dhcs | default | 62 | 1 | 65.9 | 0.804 | 61.3 | 71.5 | 45.6 | 82.6 | 82.0 | 83.8 |
| pfss-shadow | baseline | 64 | 1 | 84.5 | 0.941 | 77.3 | 94.5 | 41.4 | 86.1 | 89.8 | 93.6 |
| pfss-shadow | default | 64 | 1 | 76.0 | 0.847 | 72.2 | 80.9 | 53.9 | 99.1 | 89.8 | 93.6 |
| pfss-shadow-2 | baseline | 64 | 1 | 84.5 | 0.941 | 77.3 | 94.5 | 41.4 | 86.1 | 89.8 | 93.6 |
| pfss-shadow-2 | default | 64 | 1 | 76.0 | 0.847 | 72.2 | 80.9 | 53.9 | 99.1 | 89.8 | 93.6 |
| pfss-shadow-3 | baseline | 64 | 1 | 84.5 | 0.941 | 77.3 | 94.5 | 41.4 | 86.1 | 89.8 | 93.6 |
| pfss-shadow-3 | default | 64 | 1 | 76.0 | 0.847 | 72.2 | 80.9 | 53.9 | 99.1 | 89.8 | 93.6 |
| submitted | baseline | 15 | 0 | 60.3 | 1.589 | 74.6 | 49.6 | 68.0 | 77.7 | 38.0 | 44.9 |
| submitted | bayes | 15 | 0 | 42.5 | 1.120 | 49.4 | 37.3 | 37.7 | 50.1 | 38.0 | 44.9 |
| submitted | default | 15 | 0 | 42.5 | 1.120 | 49.4 | 37.3 | 37.7 | 50.1 | 38.0 | 44.9 |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 15 | 0 | 59.1 | 1.557 | 74.0 | 48.0 | 67.1 | 77.3 | 38.0 | 44.9 |
| bayes-option | bayes | 15 | 0 | 47.2 | 1.244 | 56.3 | 40.5 | 46.3 | 58.9 | 38.0 | 44.9 |
| bayes-option | default | 15 | 0 | 46.7 | 1.231 | 55.9 | 40.0 | 45.7 | 58.3 | 38.0 | 44.9 |
| benchlab | live | 15 | 0 | 42.1 | 1.109 | 49.3 | 36.7 | 37.4 | 50.6 | 38.0 | 44.9 |
| freeze | baseline | 15 | 0 | 59.1 | 1.557 | 74.0 | 48.0 | 67.1 | 77.3 | 38.0 | 44.9 |
| freeze | bayes | 15 | 0 | 47.2 | 1.244 | 56.3 | 40.5 | 46.3 | 58.9 | 38.0 | 44.9 |
| freeze | default | 15 | 0 | 46.7 | 1.231 | 55.9 | 40.0 | 45.7 | 58.3 | 38.0 | 44.9 |
| linear-v2 | baseline | 15 | 0 | 59.7 | 1.573 | 74.3 | 48.7 | 67.6 | 78.8 | 38.0 | 44.9 |
| linear-v2 | default | 15 | 0 | 61.2 | 1.613 | 73.1 | 52.2 | 65.2 | 84.1 | 38.0 | 44.9 |
| near-miss-a | baseline | 15 | 0 | 59.1 | 1.557 | 74.0 | 48.0 | 67.1 | 77.3 | 38.0 | 44.9 |
| near-miss-a | bayes | 15 | 0 | 47.2 | 1.244 | 56.3 | 40.5 | 46.3 | 58.9 | 38.0 | 44.9 |
| near-miss-a | default | 15 | 0 | 48.2 | 1.270 | 56.6 | 41.9 | 45.2 | 61.6 | 38.0 | 44.9 |
| near-miss-b | baseline | 15 | 0 | 61.6 | 1.623 | 75.2 | 51.3 | 68.5 | 82.3 | 38.0 | 44.9 |
| near-miss-b | bayes | 15 | 0 | 49.1 | 1.294 | 57.4 | 42.9 | 47.2 | 62.3 | 38.0 | 44.9 |
| near-miss-b | default | 15 | 0 | 42.7 | 1.126 | 49.8 | 37.5 | 37.7 | 51.0 | 38.0 | 44.9 |
| near-miss-c | baseline | 15 | 0 | 59.1 | 1.557 | 74.0 | 48.0 | 67.1 | 77.3 | 38.0 | 44.9 |
| near-miss-c | bayes | 15 | 0 | 41.9 | 1.104 | 49.1 | 36.5 | 37.3 | 50.3 | 38.0 | 44.9 |
| near-miss-c | default | 15 | 0 | 41.9 | 1.103 | 49.2 | 36.5 | 37.3 | 49.9 | 38.0 | 44.9 |
| pfss-dhcs | baseline | 15 | 0 | 59.1 | 1.557 | 74.0 | 48.0 | 67.1 | 77.3 | 38.0 | 44.9 |
| pfss-dhcs | default | 15 | 0 | 46.7 | 1.231 | 55.9 | 40.0 | 45.7 | 58.3 | 38.0 | 44.9 |
| pfss-shadow | baseline | 15 | 0 | 59.1 | 1.557 | 74.0 | 48.0 | 67.1 | 77.3 | 38.0 | 44.9 |
| pfss-shadow | default | 15 | 0 | 60.7 | 1.598 | 72.6 | 51.6 | 63.9 | 82.9 | 38.0 | 44.9 |
| pfss-shadow-2 | baseline | 15 | 0 | 59.1 | 1.557 | 74.0 | 48.0 | 67.1 | 77.3 | 38.0 | 44.9 |
| pfss-shadow-2 | default | 15 | 0 | 60.7 | 1.598 | 72.6 | 51.6 | 63.9 | 82.9 | 38.0 | 44.9 |
| pfss-shadow-3 | baseline | 15 | 0 | 59.1 | 1.557 | 74.0 | 48.0 | 67.1 | 77.3 | 38.0 | 44.9 |
| pfss-shadow-3 | default | 15 | 0 | 60.7 | 1.598 | 72.6 | 51.6 | 63.9 | 82.9 | 38.0 | 44.9 |
| submitted | baseline | 15 | 0 | 60.3 | 1.589 | 74.6 | 49.6 | 68.0 | 77.7 | 38.0 | 44.9 |
| submitted | bayes | 15 | 0 | 42.5 | 1.120 | 49.4 | 37.3 | 37.7 | 50.1 | 38.0 | 44.9 |
| submitted | default | 15 | 0 | 42.5 | 1.120 | 49.4 | 37.3 | 37.7 | 50.1 | 38.0 | 44.9 |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 28 | 0 | 0 | 8.0 | 246.7 | 5.5 | — | — | — |
| bayes-option | bayes | 28 | 0 | 0 | 7.0 | 165.1 | 5.5 | 98.4% | — | — |
| bayes-option | default | 28 | 0 | 0 | 9.7 | 247.1 | 5.5 | 98.4% | — | — |
| benchlab | dryrun | 1 | 0 | 0 | 43.0 | 43.0 | 5.6 | 100.0% | — | — |
| benchlab | live | 28 | 0 | 0 | 42.0 | 50.0 | 2.7 | 100.0% | — | — |
| freeze | baseline | 28 | 0 | 0 | 7.0 | 8.5 | 5.5 | — | — | — |
| freeze | bayes | 28 | 0 | 0 | 7.1 | 10.3 | 5.5 | 100.0% | — | — |
| freeze | default | 28 | 0 | 0 | 9.8 | 11.6 | 5.5 | 100.0% | — | — |
| linear-v2 | baseline | 28 | 0 | 0 | 7.7 | 246.7 | 5.5 | — | — | — |
| linear-v2 | default | 28 | 0 | 0 | 8.1 | 247.2 | 5.5 | 94.2% | — | — |
| near-miss-a | baseline | 28 | 0 | 0 | 6.0 | 9.5 | 5.5 | — | — | — |
| near-miss-a | bayes | 28 | 0 | 0 | 7.1 | 10.7 | 5.5 | 100.0% | — | — |
| near-miss-a | default | 28 | 0 | 0 | 9.7 | 11.7 | 5.5 | 100.0% | — | — |
| near-miss-b | baseline | 28 | 0 | 0 | 7.2 | 246.7 | 5.5 | — | — | — |
| near-miss-b | bayes | 28 | 0 | 0 | 7.0 | 165.0 | 5.5 | 98.4% | — | — |
| near-miss-b | default | 28 | 0 | 0 | 9.7 | 247.4 | 5.5 | 98.9% | — | — |
| near-miss-c | baseline | 28 | 0 | 0 | 8.0 | 23.9 | 5.5 | — | — | — |
| near-miss-c | bayes | 28 | 0 | 0 | 6.9 | 11.1 | 5.5 | 100.0% | — | — |
| near-miss-c | default | 28 | 0 | 0 | 9.6 | 11.3 | 5.5 | 100.0% | — | — |
| pfss-dhcs | baseline | 28 | 0 | 0 | 7.8 | 9.9 | 5.5 | — | — | — |
| pfss-dhcs | default | 28 | 0 | 0 | 9.8 | 14.9 | 5.5 | 100.0% | — | — |
| pfss-shadow | baseline | 28 | 0 | 0 | 8.0 | 9.9 | 5.5 | — | — | — |
| pfss-shadow | default | 28 | 0 | 0 | 42.7 | 43.1 | 5.5 | 100.0% | pfss ×28 | — |
| pfss-shadow-2 | baseline | 28 | 0 | 0 | 5.3 | 9.3 | 5.5 | — | — | — |
| pfss-shadow-2 | default | 28 | 0 | 0 | 47.0 | 47.7 | 5.5 | 100.0% | pfss ×28 | — |
| pfss-shadow-3 | baseline | 28 | 0 | 0 | 5.4 | 8.2 | 5.5 | — | — | — |
| pfss-shadow-3 | default | 28 | 0 | 0 | 10.4 | 11.6 | 5.5 | 100.0% | — | — |
| submitted | baseline | 28 | 0 | 0 | 7.2 | 246.7 | 5.5 | — | — | — |
| submitted | bayes | 28 | 0 | 0 | 7.0 | 164.9 | 5.5 | 98.9% | — | — |
| submitted | default | 28 | 0 | 0 | 7.7 | 165.7 | 5.5 | 98.9% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 37 | 1 | 0 | 7.8 | 246.7 | 5.5 | — | — | — |
| bayes-option | bayes | 37 | 1 | 0 | 7.0 | 165.1 | 5.5 | 98.8% | — | — |
| bayes-option | default | 37 | 1 | 0 | 9.7 | 247.1 | 5.5 | 98.8% | — | — |
| benchlab | dryrun | 1 | 0 | 0 | 43.0 | 43.0 | 5.6 | 100.0% | — | — |
| benchlab | live | 28 | 0 | 0 | 42.0 | 50.0 | 2.7 | 100.0% | — | — |
| freeze | baseline | 36 | 1 | 0 | 7.1 | 25.9 | 5.5 | — | — | — |
| freeze | bayes | 36 | 1 | 0 | 7.1 | 10.6 | 5.5 | 100.0% | — | — |
| freeze | default | 36 | 1 | 0 | 9.8 | 11.6 | 5.5 | 100.0% | — | — |
| linear-v2 | baseline | 124 | 1 | 0 | 6.8 | 246.7 | 5.5 | — | — | — |
| linear-v2 | default | 124 | 1 | 0 | 7.1 | 247.2 | 5.5 | 98.7% | — | — |
| near-miss-a | baseline | 36 | 1 | 0 | 5.5 | 9.5 | 5.5 | — | — | — |
| near-miss-a | bayes | 36 | 1 | 0 | 7.1 | 10.7 | 5.5 | 100.0% | — | — |
| near-miss-a | default | 36 | 1 | 0 | 9.7 | 11.7 | 5.5 | 100.0% | — | — |
| near-miss-b | baseline | 36 | 1 | 0 | 7.2 | 246.7 | 5.6 | — | — | — |
| near-miss-b | bayes | 36 | 1 | 0 | 7.0 | 165.0 | 5.6 | 98.7% | — | — |
| near-miss-b | default | 36 | 1 | 0 | 9.7 | 247.4 | 5.6 | 99.1% | — | — |
| near-miss-c | baseline | 35 | 1 | 0 | 7.5 | 23.9 | 5.5 | — | — | — |
| near-miss-c | bayes | 35 | 1 | 0 | 7.0 | 11.1 | 5.5 | 100.0% | — | — |
| near-miss-c | default | 35 | 1 | 0 | 9.6 | 11.3 | 5.5 | 100.0% | — | — |
| pfss-dhcs | baseline | 75 | 1 | 0 | 7.0 | 22.4 | 5.5 | — | — | — |
| pfss-dhcs | default | 75 | 1 | 0 | 9.9 | 137.2 | 5.5 | 100.0% | — | — |
| pfss-shadow | baseline | 77 | 1 | 0 | 7.0 | 23.2 | 5.4 | — | — | — |
| pfss-shadow | default | 77 | 1 | 0 | 42.8 | 95.1 | 5.4 | 100.0% | pfss ×76 | — |
| pfss-shadow-2 | baseline | 77 | 1 | 0 | 6.4 | 9.3 | 5.5 | — | — | — |
| pfss-shadow-2 | default | 77 | 1 | 0 | 47.0 | 105.6 | 5.5 | 100.0% | pfss ×76 | — |
| pfss-shadow-3 | baseline | 77 | 1 | 0 | 5.7 | 9.9 | 5.4 | — | — | — |
| pfss-shadow-3 | default | 77 | 1 | 0 | 9.8 | 81.6 | 5.4 | 100.0% | — | — |
| submitted | baseline | 28 | 0 | 0 | 7.2 | 246.7 | 5.5 | — | — | — |
| submitted | bayes | 28 | 0 | 0 | 7.0 | 164.9 | 5.5 | 98.9% | — | — |
| submitted | default | 28 | 0 | 0 | 7.7 | 165.7 | 5.5 | 98.9% | — | — |

## Schedule

Last cycle: 2026-11-26 12:00 UTC
Missed cycles: 2 (10-26 12Z, 10-27 06Z)

## Alerts, last 7 days

| t0 | candidate | variant | kind | detail |
|---|---|---|---|---|
| 20261119T180000Z | near-miss-b | bayes | few regressors | 16/29 observed; missing ['ch_cm_at_0h', 'ch_cm_at_6h', 'ch_cm_at_12h', 'ch_cm_at_18h', 'ch_cm_at_24h', 'ch_cm_at_30h', 'ch_cm_at_36h', 'ch_cm_at_42h', 'ch_cm_at_48h', 'ch_cm_at_54h', 'ch_cm_at_60h', 'ch_cm_at_66h', 'ch_cm_at_72h'] |
| 20261119T180000Z | near-miss-b | default | few regressors | 29/42 observed; missing ['ch_cm_at_0h', 'ch_cm_at_6h', 'ch_cm_at_12h', 'ch_cm_at_18h', 'ch_cm_at_24h', 'ch_cm_at_30h', 'ch_cm_at_36h', 'ch_cm_at_42h', 'ch_cm_at_48h', 'ch_cm_at_54h', 'ch_cm_at_60h', 'ch_cm_at_66h', 'ch_cm_at_72h'] |
| 20261120T180000Z | submitted | bayes | few regressors | 29/42 observed; missing ['ch_cm_at_0h', 'ch_cm_at_6h', 'ch_cm_at_12h', 'ch_cm_at_18h', 'ch_cm_at_24h', 'ch_cm_at_30h', 'ch_cm_at_36h', 'ch_cm_at_42h', 'ch_cm_at_48h', 'ch_cm_at_54h', 'ch_cm_at_60h', 'ch_cm_at_66h', 'ch_cm_at_72h'] |
| 20261120T180000Z | submitted | default | few regressors | 29/42 observed; missing ['ch_cm_at_0h', 'ch_cm_at_6h', 'ch_cm_at_12h', 'ch_cm_at_18h', 'ch_cm_at_24h', 'ch_cm_at_30h', 'ch_cm_at_36h', 'ch_cm_at_42h', 'ch_cm_at_48h', 'ch_cm_at_54h', 'ch_cm_at_60h', 'ch_cm_at_66h', 'ch_cm_at_72h'] |
| 20261123T060000Z | linear-v2 | default | few regressors | 3/16 observed; missing ['ch_cm_at_0h', 'ch_cm_at_6h', 'ch_cm_at_12h', 'ch_cm_at_18h', 'ch_cm_at_24h', 'ch_cm_at_30h', 'ch_cm_at_36h', 'ch_cm_at_42h', 'ch_cm_at_48h', 'ch_cm_at_54h', 'ch_cm_at_60h', 'ch_cm_at_66h', 'ch_cm_at_72h'] |
| 20261124T000000Z | linear-v2 | default | few regressors | 3/16 observed; missing ['ch_cm_at_0h', 'ch_cm_at_6h', 'ch_cm_at_12h', 'ch_cm_at_18h', 'ch_cm_at_24h', 'ch_cm_at_30h', 'ch_cm_at_36h', 'ch_cm_at_42h', 'ch_cm_at_48h', 'ch_cm_at_54h', 'ch_cm_at_60h', 'ch_cm_at_66h', 'ch_cm_at_72h'] |
| 20261126T060000Z | bayes-option | bayes | few regressors | 16/29 observed; missing ['ch_cm_at_0h', 'ch_cm_at_6h', 'ch_cm_at_12h', 'ch_cm_at_18h', 'ch_cm_at_24h', 'ch_cm_at_30h', 'ch_cm_at_36h', 'ch_cm_at_42h', 'ch_cm_at_48h', 'ch_cm_at_54h', 'ch_cm_at_60h', 'ch_cm_at_66h', 'ch_cm_at_72h'] |
| 20261126T060000Z | bayes-option | default | few regressors | 16/29 observed; missing ['ch_cm_at_0h', 'ch_cm_at_6h', 'ch_cm_at_12h', 'ch_cm_at_18h', 'ch_cm_at_24h', 'ch_cm_at_30h', 'ch_cm_at_36h', 'ch_cm_at_42h', 'ch_cm_at_48h', 'ch_cm_at_54h', 'ch_cm_at_60h', 'ch_cm_at_66h', 'ch_cm_at_72h'] |

## Other events, last 7 days (counts; details in events/)

source not ok: 56, vs staging: 28, eligibility: 1, candidate added: 1

