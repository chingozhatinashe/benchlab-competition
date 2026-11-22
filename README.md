# Staging report 2026-11-22

Generated 2026-11-22 11:56 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-11-22 10:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 8 | 0 | 36.9 | 0.723 | 43.5 | 31.8 | 35.7 | 59.7 | 51.0 | 21.9 |
| bayes-option | bayes | 8 | 0 | 41.0 | 0.803 | 49.0 | 34.5 | 27.8 | 54.4 | 51.0 | 21.9 |
| bayes-option | default | 8 | 0 | 40.8 | 0.800 | 48.7 | 34.5 | 28.2 | 53.7 | 51.0 | 21.9 |
| freeze | baseline | 7 | 0 | 39.1 | 0.933 | 45.4 | 34.3 | 37.8 | 64.5 | 41.9 | 19.1 |
| freeze | bayes | 7 | 0 | 41.8 | 0.997 | 50.1 | 35.3 | 34.8 | 59.6 | 41.9 | 19.1 |
| freeze | default | 7 | 0 | 41.7 | 0.995 | 49.8 | 35.3 | 35.2 | 58.9 | 41.9 | 19.1 |
| linear-v2 | baseline | 95 | 0 | 110.2 | 0.793 | 85.1 | 143.3 | 25.2 | 140.5 | 139.1 | 132.5 |
| linear-v2 | default | 95 | 0 | 92.3 | 0.664 | 72.8 | 117.7 | 23.1 | 124.6 | 139.1 | 132.5 |
| near-miss-a | baseline | 7 | 0 | 39.1 | 0.933 | 45.4 | 34.3 | 37.8 | 64.5 | 41.9 | 19.1 |
| near-miss-a | bayes | 7 | 0 | 41.8 | 0.997 | 50.1 | 35.3 | 34.8 | 59.6 | 41.9 | 19.1 |
| near-miss-a | default | 7 | 0 | 45.8 | 1.092 | 54.3 | 39.1 | 43.0 | 68.5 | 41.9 | 19.1 |
| near-miss-b | baseline | 7 | 0 | 39.1 | 0.933 | 45.4 | 34.3 | 37.8 | 64.5 | 41.9 | 19.1 |
| near-miss-b | bayes | 7 | 0 | 41.8 | 0.997 | 50.1 | 35.3 | 34.8 | 59.6 | 41.9 | 19.1 |
| near-miss-b | default | 7 | 0 | 39.2 | 0.935 | 47.6 | 32.7 | 34.4 | 55.1 | 41.9 | 19.1 |
| near-miss-c | baseline | 6 | 0 | 41.5 | 1.135 | 47.2 | 37.0 | 40.1 | 67.8 | 36.5 | 17.0 |
| near-miss-c | bayes | 6 | 0 | 41.9 | 1.146 | 49.3 | 36.1 | 38.2 | 62.3 | 36.5 | 17.0 |
| near-miss-c | default | 6 | 0 | 42.1 | 1.152 | 49.8 | 36.1 | 39.8 | 61.9 | 36.5 | 17.0 |
| pfss-dhcs | baseline | 46 | 0 | 64.4 | 0.768 | 71.8 | 64.5 | 30.9 | 68.8 | 83.9 | 83.0 |
| pfss-dhcs | default | 46 | 0 | 60.6 | 0.723 | 63.6 | 64.9 | 46.4 | 77.4 | 83.9 | 83.0 |
| pfss-shadow | baseline | 48 | 0 | 81.5 | 0.884 | 80.3 | 91.8 | 18.9 | 67.9 | 92.2 | 93.7 |
| pfss-shadow | default | 48 | 0 | 68.6 | 0.744 | 71.2 | 73.1 | 49.9 | 83.3 | 92.2 | 93.7 |
| pfss-shadow-2 | baseline | 48 | 0 | 81.5 | 0.884 | 80.3 | 91.8 | 18.9 | 67.9 | 92.2 | 93.7 |
| pfss-shadow-2 | default | 48 | 0 | 68.6 | 0.744 | 71.2 | 73.1 | 49.9 | 83.3 | 92.2 | 93.7 |
| pfss-shadow-3 | baseline | 48 | 0 | 81.5 | 0.884 | 80.3 | 91.8 | 18.9 | 67.9 | 92.2 | 93.7 |
| pfss-shadow-3 | default | 48 | 0 | 68.6 | 0.744 | 71.2 | 73.1 | 49.9 | 83.3 | 92.2 | 93.7 |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 6 | 0 | 41.5 | 1.135 | 47.2 | 37.0 | 40.1 | 67.8 | 36.5 | 17.0 |
| bayes-option | bayes | 6 | 0 | 44.4 | 1.216 | 51.7 | 38.9 | 41.3 | 67.9 | 36.5 | 17.0 |
| bayes-option | default | 6 | 0 | 44.2 | 1.211 | 51.4 | 38.8 | 41.6 | 66.6 | 36.5 | 17.0 |
| freeze | baseline | 6 | 0 | 41.5 | 1.135 | 47.2 | 37.0 | 40.1 | 67.8 | 36.5 | 17.0 |
| freeze | bayes | 6 | 0 | 44.4 | 1.216 | 51.7 | 38.9 | 41.3 | 67.9 | 36.5 | 17.0 |
| freeze | default | 6 | 0 | 44.2 | 1.211 | 51.4 | 38.8 | 41.6 | 66.6 | 36.5 | 17.0 |
| linear-v2 | baseline | 6 | 0 | 41.5 | 1.135 | 47.2 | 37.0 | 40.1 | 67.8 | 36.5 | 17.0 |
| linear-v2 | default | 6 | 0 | 46.8 | 1.280 | 54.9 | 40.4 | 46.3 | 74.0 | 36.5 | 17.0 |
| near-miss-a | baseline | 6 | 0 | 41.5 | 1.135 | 47.2 | 37.0 | 40.1 | 67.8 | 36.5 | 17.0 |
| near-miss-a | bayes | 6 | 0 | 44.4 | 1.216 | 51.7 | 38.9 | 41.3 | 67.9 | 36.5 | 17.0 |
| near-miss-a | default | 6 | 0 | 48.9 | 1.339 | 56.7 | 43.1 | 48.4 | 74.4 | 36.5 | 17.0 |
| near-miss-b | baseline | 6 | 0 | 41.5 | 1.135 | 47.2 | 37.0 | 40.1 | 67.8 | 36.5 | 17.0 |
| near-miss-b | bayes | 6 | 0 | 44.4 | 1.216 | 51.7 | 38.9 | 41.3 | 67.9 | 36.5 | 17.0 |
| near-miss-b | default | 6 | 0 | 42.1 | 1.152 | 49.8 | 36.1 | 39.8 | 61.9 | 36.5 | 17.0 |
| near-miss-c | baseline | 6 | 0 | 41.5 | 1.135 | 47.2 | 37.0 | 40.1 | 67.8 | 36.5 | 17.0 |
| near-miss-c | bayes | 6 | 0 | 41.9 | 1.146 | 49.3 | 36.1 | 38.2 | 62.3 | 36.5 | 17.0 |
| near-miss-c | default | 6 | 0 | 42.1 | 1.152 | 49.8 | 36.1 | 39.8 | 61.9 | 36.5 | 17.0 |
| pfss-dhcs | baseline | 6 | 0 | 41.5 | 1.135 | 47.2 | 37.0 | 40.1 | 67.8 | 36.5 | 17.0 |
| pfss-dhcs | default | 6 | 0 | 44.2 | 1.211 | 51.4 | 38.8 | 41.6 | 66.6 | 36.5 | 17.0 |
| pfss-shadow | baseline | 6 | 0 | 41.5 | 1.135 | 47.2 | 37.0 | 40.1 | 67.8 | 36.5 | 17.0 |
| pfss-shadow | default | 6 | 0 | 46.8 | 1.280 | 54.9 | 40.4 | 46.3 | 74.0 | 36.5 | 17.0 |
| pfss-shadow-2 | baseline | 6 | 0 | 41.5 | 1.135 | 47.2 | 37.0 | 40.1 | 67.8 | 36.5 | 17.0 |
| pfss-shadow-2 | default | 6 | 0 | 46.8 | 1.280 | 54.9 | 40.4 | 46.3 | 74.0 | 36.5 | 17.0 |
| pfss-shadow-3 | baseline | 6 | 0 | 41.5 | 1.135 | 47.2 | 37.0 | 40.1 | 67.8 | 36.5 | 17.0 |
| pfss-shadow-3 | default | 6 | 0 | 46.8 | 1.280 | 54.9 | 40.4 | 46.3 | 74.0 | 36.5 | 17.0 |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 20 | 1 | 0 | 7.6 | 8.9 | 5.6 | — | — | — |
| bayes-option | bayes | 20 | 1 | 0 | 7.1 | 22.1 | 5.6 | 100.0% | — | — |
| bayes-option | default | 20 | 1 | 0 | 9.9 | 40.2 | 5.6 | 100.0% | — | — |
| benchlab | dryrun | 1 | 0 | 0 | 43.0 | 43.0 | 5.6 | 100.0% | — | — |
| benchlab | live | 11 | 0 | 0 | 46.0 | 50.0 | 2.5 | 100.0% | — | — |
| freeze | baseline | 19 | 1 | 0 | 7.8 | 25.9 | 5.6 | — | — | — |
| freeze | bayes | 19 | 1 | 0 | 7.1 | 10.6 | 5.6 | 100.0% | — | — |
| freeze | default | 19 | 1 | 0 | 9.7 | 11.5 | 5.6 | 100.0% | — | — |
| linear-v2 | baseline | 28 | 1 | 0 | 6.7 | 9.4 | 5.5 | — | — | — |
| linear-v2 | default | 28 | 1 | 0 | 6.7 | 9.3 | 5.5 | 100.0% | — | — |
| near-miss-a | baseline | 19 | 1 | 0 | 4.0 | 9.3 | 5.5 | — | — | — |
| near-miss-a | bayes | 19 | 1 | 0 | 7.2 | 10.7 | 5.5 | 100.0% | — | — |
| near-miss-a | default | 19 | 1 | 0 | 9.7 | 11.7 | 5.5 | 100.0% | — | — |
| near-miss-b | baseline | 19 | 1 | 0 | 7.6 | 246.7 | 5.5 | — | — | — |
| near-miss-b | bayes | 19 | 1 | 0 | 6.8 | 165.0 | 5.5 | 97.5% | — | — |
| near-miss-b | default | 19 | 1 | 0 | 9.4 | 247.4 | 5.5 | 98.3% | — | — |
| near-miss-c | baseline | 18 | 1 | 0 | 7.7 | 8.8 | 5.6 | — | — | — |
| near-miss-c | bayes | 18 | 1 | 0 | 7.2 | 10.2 | 5.6 | 100.0% | — | — |
| near-miss-c | default | 18 | 1 | 0 | 9.9 | 11.1 | 5.6 | 100.0% | — | — |
| pfss-dhcs | baseline | 28 | 1 | 0 | 6.1 | 9.7 | 5.5 | — | — | — |
| pfss-dhcs | default | 28 | 1 | 0 | 9.9 | 14.9 | 5.5 | 100.0% | — | — |
| pfss-shadow | baseline | 28 | 1 | 0 | 8.0 | 23.2 | 5.5 | — | — | — |
| pfss-shadow | default | 28 | 1 | 0 | 42.7 | 43.0 | 5.5 | 100.0% | pfss ×27 | — |
| pfss-shadow-2 | baseline | 28 | 1 | 0 | 3.9 | 9.3 | 5.5 | — | — | — |
| pfss-shadow-2 | default | 28 | 1 | 0 | 46.9 | 47.4 | 5.5 | 100.0% | pfss ×27 | — |
| pfss-shadow-3 | baseline | 28 | 1 | 0 | 7.1 | 9.9 | 5.5 | — | — | — |
| pfss-shadow-3 | default | 28 | 1 | 0 | 9.8 | 11.4 | 5.5 | 100.0% | — | — |
| submitted | baseline | 11 | 0 | 0 | 7.9 | 246.7 | 5.5 | — | — | — |
| submitted | bayes | 11 | 0 | 0 | 7.9 | 164.9 | 5.5 | 97.2% | — | — |
| submitted | default | 11 | 0 | 0 | 7.9 | 165.7 | 5.5 | 97.2% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 20 | 1 | 0 | 7.6 | 8.9 | 5.6 | — | — | — |
| bayes-option | bayes | 20 | 1 | 0 | 7.1 | 22.1 | 5.6 | 100.0% | — | — |
| bayes-option | default | 20 | 1 | 0 | 9.9 | 40.2 | 5.6 | 100.0% | — | — |
| benchlab | dryrun | 1 | 0 | 0 | 43.0 | 43.0 | 5.6 | 100.0% | — | — |
| benchlab | live | 11 | 0 | 0 | 46.0 | 50.0 | 2.5 | 100.0% | — | — |
| freeze | baseline | 19 | 1 | 0 | 7.8 | 25.9 | 5.6 | — | — | — |
| freeze | bayes | 19 | 1 | 0 | 7.1 | 10.6 | 5.6 | 100.0% | — | — |
| freeze | default | 19 | 1 | 0 | 9.7 | 11.5 | 5.6 | 100.0% | — | — |
| linear-v2 | baseline | 107 | 1 | 0 | 6.7 | 33.3 | 5.4 | — | — | — |
| linear-v2 | default | 107 | 1 | 0 | 6.8 | 101.8 | 5.4 | 100.0% | — | — |
| near-miss-a | baseline | 19 | 1 | 0 | 4.0 | 9.3 | 5.5 | — | — | — |
| near-miss-a | bayes | 19 | 1 | 0 | 7.2 | 10.7 | 5.5 | 100.0% | — | — |
| near-miss-a | default | 19 | 1 | 0 | 9.7 | 11.7 | 5.5 | 100.0% | — | — |
| near-miss-b | baseline | 19 | 1 | 0 | 7.6 | 246.7 | 5.5 | — | — | — |
| near-miss-b | bayes | 19 | 1 | 0 | 6.8 | 165.0 | 5.5 | 97.5% | — | — |
| near-miss-b | default | 19 | 1 | 0 | 9.4 | 247.4 | 5.5 | 98.3% | — | — |
| near-miss-c | baseline | 18 | 1 | 0 | 7.7 | 8.8 | 5.6 | — | — | — |
| near-miss-c | bayes | 18 | 1 | 0 | 7.2 | 10.2 | 5.6 | 100.0% | — | — |
| near-miss-c | default | 18 | 1 | 0 | 9.9 | 11.1 | 5.6 | 100.0% | — | — |
| pfss-dhcs | baseline | 58 | 1 | 0 | 6.8 | 22.4 | 5.4 | — | — | — |
| pfss-dhcs | default | 58 | 1 | 0 | 10.4 | 137.2 | 5.4 | 100.0% | — | — |
| pfss-shadow | baseline | 60 | 1 | 0 | 7.0 | 23.2 | 5.4 | — | — | — |
| pfss-shadow | default | 60 | 1 | 0 | 42.8 | 95.1 | 5.4 | 100.0% | pfss ×59 | — |
| pfss-shadow-2 | baseline | 60 | 1 | 0 | 6.1 | 9.3 | 5.4 | — | — | — |
| pfss-shadow-2 | default | 60 | 1 | 0 | 47.0 | 105.6 | 5.4 | 100.0% | pfss ×59 | — |
| pfss-shadow-3 | baseline | 60 | 1 | 0 | 6.0 | 9.9 | 5.4 | — | — | — |
| pfss-shadow-3 | default | 60 | 1 | 0 | 9.8 | 81.6 | 5.4 | 100.0% | — | — |
| submitted | baseline | 11 | 0 | 0 | 7.9 | 246.7 | 5.5 | — | — | — |
| submitted | bayes | 11 | 0 | 0 | 7.9 | 164.9 | 5.5 | 97.2% | — | — |
| submitted | default | 11 | 0 | 0 | 7.9 | 165.7 | 5.5 | 97.2% | — | — |

## Schedule

Last cycle: 2026-11-22 06:00 UTC
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

source not ok: 54, no forecast in log: 17, vs staging: 11, candidate added: 6, eligibility: 1

