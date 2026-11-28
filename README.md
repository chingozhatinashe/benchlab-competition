# Staging report 2026-11-28

Generated 2026-11-28 12:16 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-11-28 10:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 32 | 1 | 55.3 | 1.308 | 68.4 | 45.2 | 59.5 | 79.9 | 42.3 | 42.7 |
| bayes-option | bayes | 32 | 1 | 50.0 | 1.182 | 57.1 | 44.2 | 39.3 | 71.1 | 42.3 | 42.7 |
| bayes-option | default | 32 | 1 | 50.0 | 1.182 | 57.2 | 44.1 | 39.2 | 71.1 | 42.3 | 42.7 |
| benchlab | dryrun | 1 | 0 | 0.0 | — | 69.9 | 0.0 | 69.7 | 0.0 | 0.0 | 0.0 |
| benchlab | live | 23 | 0 | 48.4 | 1.311 | 54.3 | 43.8 | 35.2 | 69.9 | 36.9 | 48.6 |
| freeze | baseline | 31 | 1 | 56.4 | 1.442 | 69.0 | 46.5 | 60.4 | 81.9 | 39.1 | 42.7 |
| freeze | bayes | 31 | 1 | 50.3 | 1.286 | 57.4 | 44.5 | 40.7 | 72.6 | 39.1 | 42.7 |
| freeze | default | 31 | 1 | 50.3 | 1.287 | 57.5 | 44.4 | 40.6 | 72.6 | 39.1 | 42.7 |
| linear-v2 | baseline | 119 | 1 | 108.0 | 0.826 | 82.6 | 141.8 | 34.6 | 138.7 | 130.8 | 126.2 |
| linear-v2 | default | 119 | 1 | 91.7 | 0.701 | 72.8 | 116.6 | 30.8 | 125.2 | 130.8 | 126.2 |
| near-miss-a | baseline | 31 | 1 | 56.4 | 1.442 | 69.0 | 46.5 | 60.4 | 81.9 | 39.1 | 42.7 |
| near-miss-a | bayes | 31 | 1 | 50.3 | 1.286 | 57.4 | 44.5 | 40.7 | 72.6 | 39.1 | 42.7 |
| near-miss-a | default | 31 | 1 | 53.1 | 1.356 | 60.2 | 47.1 | 43.1 | 79.2 | 39.1 | 42.7 |
| near-miss-b | baseline | 31 | 1 | 57.5 | 1.470 | 69.7 | 47.9 | 61.2 | 84.3 | 39.1 | 42.7 |
| near-miss-b | bayes | 31 | 1 | 51.0 | 1.305 | 58.0 | 45.4 | 41.2 | 73.9 | 39.1 | 42.7 |
| near-miss-b | default | 31 | 1 | 47.0 | 1.202 | 53.5 | 41.8 | 35.7 | 67.6 | 39.1 | 42.7 |
| near-miss-c | baseline | 30 | 1 | 57.4 | 1.531 | 69.6 | 47.8 | 61.2 | 82.9 | 37.5 | 43.0 |
| near-miss-c | bayes | 30 | 1 | 47.3 | 1.262 | 53.2 | 42.5 | 35.8 | 68.7 | 37.5 | 43.0 |
| near-miss-c | default | 30 | 1 | 47.5 | 1.268 | 53.7 | 42.5 | 36.2 | 68.5 | 37.5 | 43.0 |
| pfss-dhcs | baseline | 70 | 1 | 69.5 | 0.871 | 72.6 | 68.3 | 50.3 | 80.4 | 79.8 | 82.7 |
| pfss-dhcs | default | 70 | 1 | 66.3 | 0.830 | 62.0 | 71.5 | 44.3 | 85.2 | 79.8 | 82.7 |
| pfss-shadow | baseline | 72 | 1 | 84.3 | 0.962 | 77.2 | 94.6 | 43.3 | 87.5 | 87.6 | 92.5 |
| pfss-shadow | default | 72 | 1 | 75.6 | 0.863 | 71.6 | 80.9 | 52.1 | 100.0 | 87.6 | 92.5 |
| pfss-shadow-2 | baseline | 72 | 1 | 84.3 | 0.962 | 77.2 | 94.6 | 43.3 | 87.5 | 87.6 | 92.5 |
| pfss-shadow-2 | default | 72 | 1 | 75.6 | 0.863 | 71.6 | 80.9 | 52.1 | 100.1 | 87.6 | 92.5 |
| pfss-shadow-3 | baseline | 72 | 1 | 84.3 | 0.962 | 77.2 | 94.6 | 43.3 | 87.5 | 87.6 | 92.5 |
| pfss-shadow-3 | default | 72 | 1 | 75.6 | 0.863 | 71.6 | 80.9 | 52.1 | 100.0 | 87.6 | 92.5 |
| submitted | baseline | 23 | 0 | 62.4 | 1.693 | 73.9 | 53.2 | 66.4 | 88.1 | 36.9 | 48.6 |
| submitted | bayes | 23 | 0 | 48.6 | 1.318 | 54.4 | 44.2 | 35.4 | 69.6 | 36.9 | 48.6 |
| submitted | default | 23 | 0 | 48.6 | 1.318 | 54.4 | 44.2 | 35.4 | 69.6 | 36.9 | 48.6 |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 23 | 0 | 61.6 | 1.670 | 73.4 | 52.2 | 65.7 | 87.8 | 36.9 | 48.6 |
| bayes-option | bayes | 23 | 0 | 52.4 | 1.421 | 59.2 | 47.1 | 41.7 | 75.6 | 36.9 | 48.6 |
| bayes-option | default | 23 | 0 | 52.5 | 1.422 | 59.4 | 47.1 | 41.5 | 75.9 | 36.9 | 48.6 |
| benchlab | live | 23 | 0 | 48.4 | 1.311 | 54.3 | 43.8 | 35.2 | 69.9 | 36.9 | 48.6 |
| freeze | baseline | 23 | 0 | 61.6 | 1.670 | 73.4 | 52.2 | 65.7 | 87.8 | 36.9 | 48.6 |
| freeze | bayes | 23 | 0 | 52.4 | 1.421 | 59.2 | 47.1 | 41.7 | 75.6 | 36.9 | 48.6 |
| freeze | default | 23 | 0 | 52.5 | 1.422 | 59.4 | 47.1 | 41.5 | 75.9 | 36.9 | 48.6 |
| linear-v2 | baseline | 23 | 0 | 62.4 | 1.692 | 74.1 | 53.1 | 66.6 | 89.3 | 36.9 | 48.6 |
| linear-v2 | default | 23 | 0 | 62.3 | 1.687 | 70.9 | 55.2 | 57.2 | 93.3 | 36.9 | 48.6 |
| near-miss-a | baseline | 23 | 0 | 61.6 | 1.670 | 73.4 | 52.2 | 65.7 | 87.8 | 36.9 | 48.6 |
| near-miss-a | bayes | 23 | 0 | 52.4 | 1.421 | 59.2 | 47.1 | 41.7 | 75.6 | 36.9 | 48.6 |
| near-miss-a | default | 23 | 0 | 54.8 | 1.484 | 62.0 | 49.2 | 43.1 | 81.9 | 36.9 | 48.6 |
| near-miss-b | baseline | 23 | 0 | 63.2 | 1.712 | 74.3 | 54.2 | 66.7 | 90.9 | 36.9 | 48.6 |
| near-miss-b | bayes | 23 | 0 | 53.5 | 1.451 | 59.9 | 48.5 | 42.3 | 77.5 | 36.9 | 48.6 |
| near-miss-b | default | 23 | 0 | 48.9 | 1.326 | 55.0 | 44.2 | 35.7 | 70.3 | 36.9 | 48.6 |
| near-miss-c | baseline | 23 | 0 | 61.6 | 1.670 | 73.4 | 52.2 | 65.7 | 87.8 | 36.9 | 48.6 |
| near-miss-c | bayes | 23 | 0 | 48.2 | 1.308 | 54.2 | 43.7 | 35.1 | 69.8 | 36.9 | 48.6 |
| near-miss-c | default | 23 | 0 | 48.4 | 1.312 | 54.6 | 43.7 | 35.4 | 69.6 | 36.9 | 48.6 |
| pfss-dhcs | baseline | 23 | 0 | 61.6 | 1.670 | 73.4 | 52.2 | 65.7 | 87.8 | 36.9 | 48.6 |
| pfss-dhcs | default | 23 | 0 | 52.5 | 1.422 | 59.4 | 47.1 | 41.5 | 75.9 | 36.9 | 48.6 |
| pfss-shadow | baseline | 23 | 0 | 61.6 | 1.670 | 73.4 | 52.2 | 65.7 | 87.8 | 36.9 | 48.6 |
| pfss-shadow | default | 23 | 0 | 61.9 | 1.678 | 70.4 | 54.9 | 55.2 | 92.0 | 36.9 | 48.6 |
| pfss-shadow-2 | baseline | 23 | 0 | 61.6 | 1.670 | 73.4 | 52.2 | 65.7 | 87.8 | 36.9 | 48.6 |
| pfss-shadow-2 | default | 23 | 0 | 61.9 | 1.678 | 70.4 | 54.9 | 55.2 | 92.0 | 36.9 | 48.6 |
| pfss-shadow-3 | baseline | 23 | 0 | 61.6 | 1.670 | 73.4 | 52.2 | 65.7 | 87.8 | 36.9 | 48.6 |
| pfss-shadow-3 | default | 23 | 0 | 61.9 | 1.678 | 70.4 | 54.9 | 55.2 | 92.0 | 36.9 | 48.6 |
| submitted | baseline | 23 | 0 | 62.4 | 1.693 | 73.9 | 53.2 | 66.4 | 88.1 | 36.9 | 48.6 |
| submitted | bayes | 23 | 0 | 48.6 | 1.318 | 54.4 | 44.2 | 35.4 | 69.6 | 36.9 | 48.6 |
| submitted | default | 23 | 0 | 48.6 | 1.318 | 54.4 | 44.2 | 35.4 | 69.6 | 36.9 | 48.6 |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 28 | 0 | 0 | 8.0 | 246.7 | 5.5 | — | — | — |
| bayes-option | bayes | 28 | 0 | 0 | 7.0 | 165.1 | 5.5 | 98.4% | — | — |
| bayes-option | default | 28 | 0 | 0 | 9.7 | 247.1 | 5.5 | 98.4% | — | — |
| benchlab | live | 27 | 0 | 0 | 42.0 | 101.0 | 2.7 | 100.0% | — | — |
| freeze | baseline | 28 | 0 | 0 | 6.0 | 9.3 | 5.5 | — | — | — |
| freeze | bayes | 28 | 0 | 0 | 7.1 | 10.3 | 5.5 | 100.0% | — | — |
| freeze | default | 28 | 0 | 0 | 9.6 | 11.6 | 5.5 | 100.0% | — | — |
| linear-v2 | baseline | 28 | 0 | 0 | 8.0 | 246.7 | 5.5 | — | — | — |
| linear-v2 | default | 28 | 0 | 0 | 8.6 | 247.2 | 5.5 | 94.2% | — | — |
| near-miss-a | baseline | 28 | 0 | 0 | 7.6 | 9.5 | 5.5 | — | — | — |
| near-miss-a | bayes | 28 | 0 | 0 | 7.0 | 10.6 | 5.5 | 100.0% | — | — |
| near-miss-a | default | 28 | 0 | 0 | 9.7 | 11.5 | 5.5 | 100.0% | — | — |
| near-miss-b | baseline | 28 | 0 | 0 | 7.0 | 22.0 | 5.5 | — | — | — |
| near-miss-b | bayes | 28 | 0 | 0 | 7.1 | 10.0 | 5.5 | 100.0% | — | — |
| near-miss-b | default | 28 | 0 | 0 | 9.7 | 25.4 | 5.5 | 100.0% | — | — |
| near-miss-c | baseline | 28 | 0 | 0 | 5.8 | 23.9 | 5.5 | — | — | — |
| near-miss-c | bayes | 28 | 0 | 0 | 7.0 | 11.1 | 5.5 | 100.0% | — | — |
| near-miss-c | default | 28 | 0 | 0 | 8.4 | 11.6 | 5.5 | 100.0% | — | — |
| pfss-dhcs | baseline | 28 | 0 | 0 | 8.0 | 9.9 | 5.5 | — | — | — |
| pfss-dhcs | default | 28 | 0 | 0 | 9.8 | 11.2 | 5.5 | 100.0% | — | — |
| pfss-shadow | baseline | 28 | 0 | 0 | 5.7 | 9.6 | 5.5 | — | — | — |
| pfss-shadow | default | 28 | 0 | 0 | 42.8 | 43.1 | 5.5 | 100.0% | pfss ×28 | — |
| pfss-shadow-2 | baseline | 28 | 0 | 0 | 6.8 | 9.3 | 5.5 | — | — | — |
| pfss-shadow-2 | default | 28 | 0 | 0 | 47.0 | 47.7 | 5.5 | 100.0% | pfss ×28 | — |
| pfss-shadow-3 | baseline | 28 | 0 | 0 | 5.1 | 8.2 | 5.5 | — | — | — |
| pfss-shadow-3 | default | 28 | 0 | 0 | 10.1 | 11.6 | 5.5 | 100.0% | — | — |
| submitted | baseline | 28 | 0 | 0 | 6.8 | 14.9 | 5.5 | — | — | — |
| submitted | bayes | 28 | 0 | 0 | 6.9 | 11.5 | 5.5 | 100.0% | — | — |
| submitted | default | 28 | 0 | 0 | 7.6 | 11.6 | 5.5 | 100.0% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 45 | 1 | 0 | 7.8 | 246.7 | 5.5 | — | — | — |
| bayes-option | bayes | 45 | 1 | 0 | 7.0 | 165.1 | 5.5 | 99.0% | — | — |
| bayes-option | default | 45 | 1 | 0 | 9.7 | 247.1 | 5.5 | 99.0% | — | — |
| benchlab | dryrun | 1 | 0 | 0 | 43.0 | 43.0 | 5.6 | 100.0% | — | — |
| benchlab | live | 35 | 0 | 0 | 42.0 | 101.0 | 2.7 | 100.0% | — | — |
| freeze | baseline | 44 | 1 | 0 | 7.1 | 25.9 | 5.5 | — | — | — |
| freeze | bayes | 44 | 1 | 0 | 7.1 | 10.6 | 5.5 | 100.0% | — | — |
| freeze | default | 44 | 1 | 0 | 9.7 | 11.6 | 5.5 | 100.0% | — | — |
| linear-v2 | baseline | 132 | 1 | 0 | 6.8 | 246.7 | 5.5 | — | — | — |
| linear-v2 | default | 132 | 1 | 0 | 7.1 | 247.2 | 5.5 | 98.8% | — | — |
| near-miss-a | baseline | 44 | 1 | 0 | 6.1 | 9.5 | 5.5 | — | — | — |
| near-miss-a | bayes | 44 | 1 | 0 | 7.0 | 10.7 | 5.5 | 100.0% | — | — |
| near-miss-a | default | 44 | 1 | 0 | 9.7 | 11.7 | 5.5 | 100.0% | — | — |
| near-miss-b | baseline | 44 | 1 | 0 | 7.1 | 246.7 | 5.5 | — | — | — |
| near-miss-b | bayes | 44 | 1 | 0 | 7.0 | 165.0 | 5.5 | 99.0% | — | — |
| near-miss-b | default | 44 | 1 | 0 | 9.7 | 247.4 | 5.5 | 99.3% | — | — |
| near-miss-c | baseline | 43 | 1 | 0 | 7.2 | 23.9 | 5.5 | — | — | — |
| near-miss-c | bayes | 43 | 1 | 0 | 7.0 | 11.1 | 5.5 | 100.0% | — | — |
| near-miss-c | default | 43 | 1 | 0 | 9.2 | 11.6 | 5.5 | 100.0% | — | — |
| pfss-dhcs | baseline | 83 | 1 | 0 | 7.0 | 22.4 | 5.5 | — | — | — |
| pfss-dhcs | default | 83 | 1 | 0 | 9.9 | 137.2 | 5.5 | 100.0% | — | — |
| pfss-shadow | baseline | 85 | 1 | 0 | 7.0 | 23.2 | 5.4 | — | — | — |
| pfss-shadow | default | 85 | 1 | 0 | 42.8 | 95.1 | 5.4 | 100.0% | pfss ×84 | — |
| pfss-shadow-2 | baseline | 85 | 1 | 0 | 6.5 | 9.3 | 5.5 | — | — | — |
| pfss-shadow-2 | default | 85 | 1 | 0 | 47.0 | 105.6 | 5.5 | 100.0% | pfss ×84 | — |
| pfss-shadow-3 | baseline | 85 | 1 | 0 | 5.6 | 9.9 | 5.5 | — | — | — |
| pfss-shadow-3 | default | 85 | 1 | 0 | 9.8 | 81.6 | 5.5 | 100.0% | — | — |
| submitted | baseline | 36 | 0 | 0 | 7.4 | 246.7 | 5.5 | — | — | — |
| submitted | bayes | 36 | 0 | 0 | 7.0 | 164.9 | 5.5 | 99.1% | — | — |
| submitted | default | 36 | 0 | 0 | 7.6 | 165.7 | 5.5 | 99.1% | — | — |

## Schedule

Last cycle: 2026-11-28 12:00 UTC
Missed cycles: 2 (10-26 12Z, 10-27 06Z)

## Alerts, last 7 days

| t0 | candidate | variant | kind | detail |
|---|---|---|---|---|
| 20261123T060000Z | linear-v2 | default | few regressors | 3/16 observed; missing ['ch_cm_at_0h', 'ch_cm_at_6h', 'ch_cm_at_12h', 'ch_cm_at_18h', 'ch_cm_at_24h', 'ch_cm_at_30h', 'ch_cm_at_36h', 'ch_cm_at_42h', 'ch_cm_at_48h', 'ch_cm_at_54h', 'ch_cm_at_60h', 'ch_cm_at_66h', 'ch_cm_at_72h'] |
| 20261124T000000Z | linear-v2 | default | few regressors | 3/16 observed; missing ['ch_cm_at_0h', 'ch_cm_at_6h', 'ch_cm_at_12h', 'ch_cm_at_18h', 'ch_cm_at_24h', 'ch_cm_at_30h', 'ch_cm_at_36h', 'ch_cm_at_42h', 'ch_cm_at_48h', 'ch_cm_at_54h', 'ch_cm_at_60h', 'ch_cm_at_66h', 'ch_cm_at_72h'] |
| 20261126T060000Z | bayes-option | bayes | few regressors | 16/29 observed; missing ['ch_cm_at_0h', 'ch_cm_at_6h', 'ch_cm_at_12h', 'ch_cm_at_18h', 'ch_cm_at_24h', 'ch_cm_at_30h', 'ch_cm_at_36h', 'ch_cm_at_42h', 'ch_cm_at_48h', 'ch_cm_at_54h', 'ch_cm_at_60h', 'ch_cm_at_66h', 'ch_cm_at_72h'] |
| 20261126T060000Z | bayes-option | default | few regressors | 16/29 observed; missing ['ch_cm_at_0h', 'ch_cm_at_6h', 'ch_cm_at_12h', 'ch_cm_at_18h', 'ch_cm_at_24h', 'ch_cm_at_30h', 'ch_cm_at_36h', 'ch_cm_at_42h', 'ch_cm_at_48h', 'ch_cm_at_54h', 'ch_cm_at_60h', 'ch_cm_at_66h', 'ch_cm_at_72h'] |

## Other events, last 7 days (counts; details in events/)

source not ok: 56, vs staging: 27

