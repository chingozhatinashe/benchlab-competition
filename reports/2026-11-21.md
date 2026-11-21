# Staging report 2026-11-21

Generated 2026-11-21 11:41 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-11-21 10:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 4 | 0 | 26.1 | 0.499 | 31.5 | 23.4 | 25.0 | 52.3 | 52.3 | 22.5 |
| bayes-option | bayes | 4 | 0 | 29.0 | 0.553 | 37.8 | 25.0 | 4.2 | 41.0 | 52.3 | 22.5 |
| bayes-option | default | 4 | 0 | 29.0 | 0.555 | 37.8 | 25.0 | 5.3 | 41.7 | 52.3 | 22.5 |
| freeze | baseline | 3 | 0 | 27.4 | 0.753 | 33.3 | 25.2 | 27.0 | 57.2 | 36.5 | 17.0 |
| freeze | bayes | 3 | 0 | 26.6 | 0.731 | 35.8 | 23.1 | 14.1 | 44.4 | 36.5 | 17.0 |
| freeze | default | 3 | 0 | 27.0 | 0.739 | 35.8 | 23.5 | 15.2 | 45.3 | 36.5 | 17.0 |
| linear-v2 | baseline | 91 | 0 | 110.9 | 0.791 | 86.0 | 143.3 | 24.6 | 141.2 | 140.2 | 133.7 |
| linear-v2 | default | 91 | 0 | 92.5 | 0.660 | 73.3 | 117.6 | 21.9 | 124.9 | 140.2 | 133.7 |
| near-miss-a | baseline | 3 | 0 | 27.4 | 0.753 | 33.3 | 25.2 | 27.0 | 57.2 | 36.5 | 17.0 |
| near-miss-a | bayes | 3 | 0 | 26.6 | 0.731 | 35.8 | 23.1 | 14.1 | 44.4 | 36.5 | 17.0 |
| near-miss-a | default | 3 | 0 | 30.0 | 0.824 | 38.4 | 26.7 | 26.0 | 58.5 | 36.5 | 17.0 |
| near-miss-b | baseline | 3 | 0 | 27.4 | 0.753 | 33.3 | 25.2 | 27.0 | 57.2 | 36.5 | 17.0 |
| near-miss-b | bayes | 3 | 0 | 26.6 | 0.731 | 35.8 | 23.1 | 14.1 | 44.4 | 36.5 | 17.0 |
| near-miss-b | default | 3 | 0 | 23.7 | 0.650 | 31.1 | 20.7 | 15.8 | 41.8 | 36.5 | 17.0 |
| near-miss-c | baseline | 2 | 0 | 21.3 | 1.218 | 36.1 | 19.8 | 30.7 | 42.1 | 17.5 | 10.2 |
| near-miss-c | bayes | 2 | 0 | 17.9 | 1.024 | 32.8 | 15.8 | 23.5 | 35.0 | 17.5 | 10.2 |
| near-miss-c | default | 2 | 0 | 18.3 | 1.046 | 33.1 | 16.1 | 25.4 | 35.5 | 17.5 | 10.2 |
| pfss-dhcs | baseline | 42 | 0 | 65.1 | 0.760 | 73.0 | 64.7 | 31.1 | 69.3 | 85.7 | 84.8 |
| pfss-dhcs | default | 42 | 0 | 61.0 | 0.712 | 64.1 | 65.0 | 46.1 | 77.8 | 85.7 | 84.8 |
| pfss-shadow | baseline | 44 | 0 | 82.5 | 0.879 | 81.6 | 92.2 | 18.6 | 68.5 | 93.8 | 95.6 |
| pfss-shadow | default | 44 | 0 | 69.1 | 0.736 | 72.0 | 73.3 | 49.7 | 83.7 | 93.8 | 95.6 |
| pfss-shadow-2 | baseline | 44 | 0 | 82.5 | 0.879 | 81.6 | 92.2 | 18.6 | 68.5 | 93.8 | 95.6 |
| pfss-shadow-2 | default | 44 | 0 | 69.1 | 0.736 | 72.0 | 73.3 | 49.7 | 83.7 | 93.8 | 95.6 |
| pfss-shadow-3 | baseline | 44 | 0 | 82.5 | 0.879 | 81.6 | 92.2 | 18.6 | 68.5 | 93.8 | 95.6 |
| pfss-shadow-3 | default | 44 | 0 | 69.1 | 0.736 | 72.0 | 73.3 | 49.7 | 83.7 | 93.8 | 95.6 |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 2 | 0 | 21.3 | 1.218 | 36.1 | 19.8 | 30.7 | 42.1 | 17.5 | 10.2 |
| bayes-option | bayes | 2 | 0 | 19.9 | 1.142 | 36.5 | 17.8 | 26.2 | 39.0 | 17.5 | 10.2 |
| bayes-option | default | 2 | 0 | 20.0 | 1.148 | 36.7 | 17.9 | 27.2 | 39.2 | 17.5 | 10.2 |
| freeze | baseline | 2 | 0 | 21.3 | 1.218 | 36.1 | 19.8 | 30.7 | 42.1 | 17.5 | 10.2 |
| freeze | bayes | 2 | 0 | 19.9 | 1.142 | 36.5 | 17.8 | 26.2 | 39.0 | 17.5 | 10.2 |
| freeze | default | 2 | 0 | 20.0 | 1.148 | 36.7 | 17.9 | 27.2 | 39.2 | 17.5 | 10.2 |
| linear-v2 | baseline | 2 | 0 | 21.3 | 1.218 | 36.1 | 19.8 | 30.7 | 42.1 | 17.5 | 10.2 |
| linear-v2 | default | 2 | 0 | 20.7 | 1.188 | 37.8 | 18.3 | 31.2 | 40.5 | 17.5 | 10.2 |
| near-miss-a | baseline | 2 | 0 | 21.3 | 1.218 | 36.1 | 19.8 | 30.7 | 42.1 | 17.5 | 10.2 |
| near-miss-a | bayes | 2 | 0 | 19.9 | 1.142 | 36.5 | 17.8 | 26.2 | 39.0 | 17.5 | 10.2 |
| near-miss-a | default | 2 | 0 | 23.7 | 1.357 | 41.5 | 21.6 | 36.4 | 46.6 | 17.5 | 10.2 |
| near-miss-b | baseline | 2 | 0 | 21.3 | 1.218 | 36.1 | 19.8 | 30.7 | 42.1 | 17.5 | 10.2 |
| near-miss-b | bayes | 2 | 0 | 19.9 | 1.142 | 36.5 | 17.8 | 26.2 | 39.0 | 17.5 | 10.2 |
| near-miss-b | default | 2 | 0 | 18.3 | 1.046 | 33.1 | 16.1 | 25.4 | 35.5 | 17.5 | 10.2 |
| near-miss-c | baseline | 2 | 0 | 21.3 | 1.218 | 36.1 | 19.8 | 30.7 | 42.1 | 17.5 | 10.2 |
| near-miss-c | bayes | 2 | 0 | 17.9 | 1.024 | 32.8 | 15.8 | 23.5 | 35.0 | 17.5 | 10.2 |
| near-miss-c | default | 2 | 0 | 18.3 | 1.046 | 33.1 | 16.1 | 25.4 | 35.5 | 17.5 | 10.2 |
| pfss-dhcs | baseline | 2 | 0 | 21.3 | 1.218 | 36.1 | 19.8 | 30.7 | 42.1 | 17.5 | 10.2 |
| pfss-dhcs | default | 2 | 0 | 20.0 | 1.148 | 36.7 | 17.9 | 27.2 | 39.2 | 17.5 | 10.2 |
| pfss-shadow | baseline | 2 | 0 | 21.3 | 1.218 | 36.1 | 19.8 | 30.7 | 42.1 | 17.5 | 10.2 |
| pfss-shadow | default | 2 | 0 | 20.7 | 1.188 | 37.8 | 18.3 | 31.2 | 40.5 | 17.5 | 10.2 |
| pfss-shadow-2 | baseline | 2 | 0 | 21.3 | 1.218 | 36.1 | 19.8 | 30.7 | 42.1 | 17.5 | 10.2 |
| pfss-shadow-2 | default | 2 | 0 | 20.7 | 1.188 | 37.8 | 18.3 | 31.2 | 40.5 | 17.5 | 10.2 |
| pfss-shadow-3 | baseline | 2 | 0 | 21.3 | 1.218 | 36.1 | 19.8 | 30.7 | 42.1 | 17.5 | 10.2 |
| pfss-shadow-3 | default | 2 | 0 | 20.7 | 1.188 | 37.8 | 18.3 | 31.2 | 40.5 | 17.5 | 10.2 |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 16 | 1 | 0 | 7.4 | 8.9 | 5.6 | — | — | — |
| bayes-option | bayes | 16 | 1 | 0 | 7.1 | 22.1 | 5.6 | 100.0% | — | — |
| bayes-option | default | 16 | 1 | 0 | 10.0 | 11.8 | 5.6 | 100.0% | — | — |
| benchlab | dryrun | 1 | 0 | 0 | 43.0 | 43.0 | 5.6 | 100.0% | — | — |
| benchlab | live | 7 | 0 | 0 | 42.0 | 50.0 | 2.5 | 100.0% | — | — |
| freeze | baseline | 15 | 1 | 0 | 8.1 | 25.9 | 5.7 | — | — | — |
| freeze | bayes | 15 | 1 | 0 | 7.0 | 10.6 | 5.7 | 100.0% | — | — |
| freeze | default | 15 | 1 | 0 | 9.7 | 11.5 | 5.7 | 100.0% | — | — |
| linear-v2 | baseline | 28 | 1 | 0 | 7.0 | 9.4 | 5.5 | — | — | — |
| linear-v2 | default | 28 | 1 | 0 | 6.8 | 101.8 | 5.5 | 100.0% | — | — |
| near-miss-a | baseline | 15 | 1 | 0 | 3.9 | 9.3 | 5.5 | — | — | — |
| near-miss-a | bayes | 15 | 1 | 0 | 7.2 | 10.7 | 5.5 | 100.0% | — | — |
| near-miss-a | default | 15 | 1 | 0 | 9.2 | 11.7 | 5.5 | 100.0% | — | — |
| near-miss-b | baseline | 15 | 1 | 0 | 8.0 | 246.7 | 5.6 | — | — | — |
| near-miss-b | bayes | 15 | 1 | 0 | 7.0 | 165.0 | 5.6 | 96.8% | — | — |
| near-miss-b | default | 15 | 1 | 0 | 9.7 | 247.4 | 5.6 | 97.8% | — | — |
| near-miss-c | baseline | 14 | 1 | 0 | 7.7 | 8.8 | 5.6 | — | — | — |
| near-miss-c | bayes | 14 | 1 | 0 | 7.4 | 10.2 | 5.6 | 100.0% | — | — |
| near-miss-c | default | 14 | 1 | 0 | 10.1 | 11.1 | 5.6 | 100.0% | — | — |
| pfss-dhcs | baseline | 28 | 1 | 0 | 7.2 | 9.7 | 5.5 | — | — | — |
| pfss-dhcs | default | 28 | 1 | 0 | 9.8 | 137.2 | 5.5 | 100.0% | — | — |
| pfss-shadow | baseline | 28 | 1 | 0 | 6.9 | 23.2 | 5.5 | — | — | — |
| pfss-shadow | default | 28 | 1 | 0 | 42.8 | 95.1 | 5.5 | 100.0% | pfss ×27 | — |
| pfss-shadow-2 | baseline | 28 | 1 | 0 | 3.9 | 9.3 | 5.5 | — | — | — |
| pfss-shadow-2 | default | 28 | 1 | 0 | 46.9 | 105.6 | 5.5 | 100.0% | pfss ×27 | — |
| pfss-shadow-3 | baseline | 28 | 1 | 0 | 7.1 | 9.9 | 5.5 | — | — | — |
| pfss-shadow-3 | default | 28 | 1 | 0 | 9.5 | 81.6 | 5.5 | 100.0% | — | — |
| submitted | baseline | 7 | 0 | 0 | 8.1 | 246.7 | 5.6 | — | — | — |
| submitted | bayes | 7 | 0 | 0 | 8.0 | 164.9 | 5.6 | 95.6% | — | — |
| submitted | default | 7 | 0 | 0 | 7.9 | 165.7 | 5.6 | 95.6% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 16 | 1 | 0 | 7.4 | 8.9 | 5.6 | — | — | — |
| bayes-option | bayes | 16 | 1 | 0 | 7.1 | 22.1 | 5.6 | 100.0% | — | — |
| bayes-option | default | 16 | 1 | 0 | 10.0 | 11.8 | 5.6 | 100.0% | — | — |
| benchlab | dryrun | 1 | 0 | 0 | 43.0 | 43.0 | 5.6 | 100.0% | — | — |
| benchlab | live | 7 | 0 | 0 | 42.0 | 50.0 | 2.5 | 100.0% | — | — |
| freeze | baseline | 15 | 1 | 0 | 8.1 | 25.9 | 5.7 | — | — | — |
| freeze | bayes | 15 | 1 | 0 | 7.0 | 10.6 | 5.7 | 100.0% | — | — |
| freeze | default | 15 | 1 | 0 | 9.7 | 11.5 | 5.7 | 100.0% | — | — |
| linear-v2 | baseline | 103 | 1 | 0 | 6.6 | 33.3 | 5.4 | — | — | — |
| linear-v2 | default | 103 | 1 | 0 | 6.8 | 101.8 | 5.4 | 100.0% | — | — |
| near-miss-a | baseline | 15 | 1 | 0 | 3.9 | 9.3 | 5.5 | — | — | — |
| near-miss-a | bayes | 15 | 1 | 0 | 7.2 | 10.7 | 5.5 | 100.0% | — | — |
| near-miss-a | default | 15 | 1 | 0 | 9.2 | 11.7 | 5.5 | 100.0% | — | — |
| near-miss-b | baseline | 15 | 1 | 0 | 8.0 | 246.7 | 5.6 | — | — | — |
| near-miss-b | bayes | 15 | 1 | 0 | 7.0 | 165.0 | 5.6 | 96.8% | — | — |
| near-miss-b | default | 15 | 1 | 0 | 9.7 | 247.4 | 5.6 | 97.8% | — | — |
| near-miss-c | baseline | 14 | 1 | 0 | 7.7 | 8.8 | 5.6 | — | — | — |
| near-miss-c | bayes | 14 | 1 | 0 | 7.4 | 10.2 | 5.6 | 100.0% | — | — |
| near-miss-c | default | 14 | 1 | 0 | 10.1 | 11.1 | 5.6 | 100.0% | — | — |
| pfss-dhcs | baseline | 54 | 1 | 0 | 6.8 | 22.4 | 5.4 | — | — | — |
| pfss-dhcs | default | 54 | 1 | 0 | 10.3 | 137.2 | 5.4 | 100.0% | — | — |
| pfss-shadow | baseline | 56 | 1 | 0 | 7.0 | 23.2 | 5.4 | — | — | — |
| pfss-shadow | default | 56 | 1 | 0 | 42.8 | 95.1 | 5.4 | 100.0% | pfss ×55 | — |
| pfss-shadow-2 | baseline | 56 | 1 | 0 | 6.3 | 9.3 | 5.4 | — | — | — |
| pfss-shadow-2 | default | 56 | 1 | 0 | 47.0 | 105.6 | 5.4 | 100.0% | pfss ×55 | — |
| pfss-shadow-3 | baseline | 56 | 1 | 0 | 6.0 | 9.9 | 5.4 | — | — | — |
| pfss-shadow-3 | default | 56 | 1 | 0 | 9.6 | 81.6 | 5.4 | 100.0% | — | — |
| submitted | baseline | 7 | 0 | 0 | 8.1 | 246.7 | 5.6 | — | — | — |
| submitted | bayes | 7 | 0 | 0 | 8.0 | 164.9 | 5.6 | 95.6% | — | — |
| submitted | default | 7 | 0 | 0 | 7.9 | 165.7 | 5.6 | 95.6% | — | — |

## Schedule

Last cycle: 2026-11-21 06:00 UTC
Missed cycles: 2 (10-26 12Z, 10-27 06Z)

## Alerts, last 7 days

| t0 | candidate | variant | kind | detail |
|---|---|---|---|---|
| 20261119T120000Z | bayes-option | baseline | build failed | the image did not build; scored as persistence |
| 20261119T120000Z | bayes-option | bayes | build failed | the image did not build; scored as persistence |
| 20261119T120000Z | bayes-option | default | build failed | the image did not build; scored as persistence |
| 20261119T120000Z | freeze | baseline | build failed | the image did not build; scored as persistence |
| 20261119T120000Z | freeze | bayes | build failed | the image did not build; scored as persistence |
| 20261119T120000Z | freeze | default | build failed | the image did not build; scored as persistence |
| 20261119T120000Z | linear-v2 | baseline | build failed | the image did not build; scored as persistence |
| 20261119T120000Z | linear-v2 | default | build failed | the image did not build; scored as persistence |
| 20261119T120000Z | near-miss-a | baseline | build failed | the image did not build; scored as persistence |
| 20261119T120000Z | near-miss-a | bayes | build failed | the image did not build; scored as persistence |
| 20261119T120000Z | near-miss-a | default | build failed | the image did not build; scored as persistence |
| 20261119T120000Z | near-miss-b | baseline | build failed | the image did not build; scored as persistence |
| 20261119T120000Z | near-miss-b | bayes | build failed | the image did not build; scored as persistence |
| 20261119T120000Z | near-miss-b | default | build failed | the image did not build; scored as persistence |
| 20261119T120000Z | near-miss-c | baseline | build failed | the image did not build; scored as persistence |
| 20261119T120000Z | near-miss-c | bayes | build failed | the image did not build; scored as persistence |
| 20261119T120000Z | near-miss-c | default | build failed | the image did not build; scored as persistence |
| 20261119T120000Z | pfss-dhcs | baseline | build failed | the image did not build; scored as persistence |
| 20261119T120000Z | pfss-dhcs | default | build failed | the image did not build; scored as persistence |
| 20261119T120000Z | pfss-shadow-2 | baseline | build failed | the image did not build; scored as persistence |
| 20261119T120000Z | pfss-shadow-2 | default | build failed | the image did not build; scored as persistence |
| 20261119T120000Z | pfss-shadow-3 | baseline | build failed | the image did not build; scored as persistence |
| 20261119T120000Z | pfss-shadow-3 | default | build failed | the image did not build; scored as persistence |
| 20261119T120000Z | pfss-shadow | baseline | build failed | the image did not build; scored as persistence |
| 20261119T120000Z | pfss-shadow | default | build failed | the image did not build; scored as persistence |
| 20261119T180000Z | near-miss-b | bayes | few regressors | 16/29 observed; missing ['ch_cm_at_0h', 'ch_cm_at_6h', 'ch_cm_at_12h', 'ch_cm_at_18h', 'ch_cm_at_24h', 'ch_cm_at_30h', 'ch_cm_at_36h', 'ch_cm_at_42h', 'ch_cm_at_48h', 'ch_cm_at_54h', 'ch_cm_at_60h', 'ch_cm_at_66h', 'ch_cm_at_72h'] |
| 20261119T180000Z | near-miss-b | default | few regressors | 29/42 observed; missing ['ch_cm_at_0h', 'ch_cm_at_6h', 'ch_cm_at_12h', 'ch_cm_at_18h', 'ch_cm_at_24h', 'ch_cm_at_30h', 'ch_cm_at_36h', 'ch_cm_at_42h', 'ch_cm_at_48h', 'ch_cm_at_54h', 'ch_cm_at_60h', 'ch_cm_at_66h', 'ch_cm_at_72h'] |
| 20261120T180000Z | submitted | bayes | few regressors | 29/42 observed; missing ['ch_cm_at_0h', 'ch_cm_at_6h', 'ch_cm_at_12h', 'ch_cm_at_18h', 'ch_cm_at_24h', 'ch_cm_at_30h', 'ch_cm_at_36h', 'ch_cm_at_42h', 'ch_cm_at_48h', 'ch_cm_at_54h', 'ch_cm_at_60h', 'ch_cm_at_66h', 'ch_cm_at_72h'] |
| 20261120T180000Z | submitted | default | few regressors | 29/42 observed; missing ['ch_cm_at_0h', 'ch_cm_at_6h', 'ch_cm_at_12h', 'ch_cm_at_18h', 'ch_cm_at_24h', 'ch_cm_at_30h', 'ch_cm_at_36h', 'ch_cm_at_42h', 'ch_cm_at_48h', 'ch_cm_at_54h', 'ch_cm_at_60h', 'ch_cm_at_66h', 'ch_cm_at_72h'] |

## Other events, last 7 days (counts; details in events/)

source not ok: 54, no forecast in log: 19, vs staging: 7, candidate added: 6, eligibility: 1

