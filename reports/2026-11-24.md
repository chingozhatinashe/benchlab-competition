# Staging report 2026-11-24

Generated 2026-11-24 14:13 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-11-24 13:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 17 | 1 | 45.2 | 0.991 | 57.1 | 36.6 | 49.6 | 67.5 | 45.6 | 30.0 |
| bayes-option | bayes | 17 | 1 | 44.3 | 0.972 | 52.0 | 38.5 | 40.3 | 60.8 | 45.6 | 30.0 |
| bayes-option | default | 17 | 1 | 43.9 | 0.962 | 51.4 | 38.2 | 39.8 | 59.6 | 45.6 | 30.0 |
| benchlab | dryrun | 1 | 0 | 0.0 | — | 69.9 | 0.0 | 69.7 | 0.0 | 0.0 | 0.0 |
| benchlab | live | 8 | 0 | 41.7 | 1.032 | 49.6 | 36.5 | 43.4 | 54.1 | 40.5 | 34.9 |
| freeze | baseline | 16 | 1 | 46.5 | 1.130 | 58.2 | 38.1 | 51.1 | 69.8 | 41.2 | 29.3 |
| freeze | bayes | 16 | 1 | 44.8 | 1.089 | 52.5 | 39.0 | 43.4 | 62.8 | 41.2 | 29.3 |
| freeze | default | 16 | 1 | 44.4 | 1.078 | 52.0 | 38.6 | 42.9 | 61.6 | 41.2 | 29.3 |
| linear-v2 | baseline | 104 | 1 | 108.7 | 0.803 | 83.1 | 142.6 | 29.0 | 139.0 | 135.3 | 129.1 |
| linear-v2 | default | 104 | 1 | 91.9 | 0.679 | 72.6 | 117.1 | 27.3 | 125.0 | 135.3 | 129.1 |
| near-miss-a | baseline | 16 | 1 | 46.5 | 1.130 | 58.2 | 38.1 | 51.1 | 69.8 | 41.2 | 29.3 |
| near-miss-a | bayes | 16 | 1 | 44.8 | 1.089 | 52.5 | 39.0 | 43.4 | 62.8 | 41.2 | 29.3 |
| near-miss-a | default | 16 | 1 | 47.5 | 1.154 | 55.5 | 41.6 | 47.8 | 68.0 | 41.2 | 29.3 |
| near-miss-b | baseline | 16 | 1 | 48.6 | 1.181 | 59.9 | 40.4 | 52.6 | 74.3 | 41.2 | 29.3 |
| near-miss-b | bayes | 16 | 1 | 46.1 | 1.120 | 53.8 | 40.3 | 44.3 | 65.3 | 41.2 | 29.3 |
| near-miss-b | default | 16 | 1 | 41.2 | 1.002 | 48.7 | 35.8 | 39.7 | 55.2 | 41.2 | 29.3 |
| near-miss-c | baseline | 15 | 1 | 47.8 | 1.228 | 59.3 | 39.3 | 52.6 | 70.5 | 38.9 | 29.4 |
| near-miss-c | bayes | 15 | 1 | 41.9 | 1.077 | 48.9 | 36.7 | 40.9 | 56.9 | 38.9 | 29.4 |
| near-miss-c | default | 15 | 1 | 41.9 | 1.077 | 48.9 | 36.7 | 41.2 | 56.1 | 38.9 | 29.4 |
| pfss-dhcs | baseline | 55 | 1 | 68.5 | 0.814 | 70.2 | 68.6 | 41.3 | 77.7 | 84.1 | 84.0 |
| pfss-dhcs | default | 55 | 1 | 65.8 | 0.782 | 61.0 | 71.5 | 45.1 | 83.5 | 84.1 | 84.0 |
| pfss-shadow | baseline | 57 | 1 | 84.5 | 0.915 | 76.5 | 95.5 | 32.8 | 85.8 | 92.4 | 94.5 |
| pfss-shadow | default | 57 | 1 | 75.0 | 0.812 | 70.4 | 80.7 | 51.5 | 99.5 | 92.4 | 94.5 |
| pfss-shadow-2 | baseline | 57 | 1 | 84.5 | 0.915 | 76.5 | 95.5 | 32.8 | 85.8 | 92.4 | 94.5 |
| pfss-shadow-2 | default | 57 | 1 | 75.0 | 0.812 | 70.4 | 80.7 | 51.6 | 99.5 | 92.4 | 94.5 |
| pfss-shadow-3 | baseline | 57 | 1 | 84.5 | 0.915 | 76.5 | 95.5 | 32.8 | 85.8 | 92.4 | 94.5 |
| pfss-shadow-3 | default | 57 | 1 | 75.0 | 0.812 | 70.4 | 80.7 | 51.5 | 99.5 | 92.4 | 94.5 |
| submitted | baseline | 8 | 0 | 54.8 | 1.356 | 67.8 | 46.3 | 63.0 | 76.1 | 40.5 | 34.9 |
| submitted | bayes | 8 | 0 | 42.1 | 1.040 | 50.1 | 36.9 | 44.0 | 53.4 | 40.5 | 34.9 |
| submitted | default | 8 | 0 | 42.1 | 1.040 | 50.1 | 36.9 | 44.0 | 53.4 | 40.5 | 34.9 |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 8 | 0 | 53.3 | 1.318 | 66.0 | 44.7 | 61.0 | 75.5 | 40.5 | 34.9 |
| bayes-option | bayes | 8 | 0 | 47.4 | 1.171 | 55.3 | 42.0 | 49.7 | 66.0 | 40.5 | 34.9 |
| bayes-option | default | 8 | 0 | 46.5 | 1.150 | 54.5 | 41.2 | 48.6 | 64.3 | 40.5 | 34.9 |
| benchlab | live | 8 | 0 | 41.7 | 1.032 | 49.6 | 36.5 | 43.4 | 54.1 | 40.5 | 34.9 |
| freeze | baseline | 8 | 0 | 53.3 | 1.318 | 66.0 | 44.7 | 61.0 | 75.5 | 40.5 | 34.9 |
| freeze | bayes | 8 | 0 | 47.4 | 1.171 | 55.3 | 42.0 | 49.7 | 66.0 | 40.5 | 34.9 |
| freeze | default | 8 | 0 | 46.5 | 1.150 | 54.5 | 41.2 | 48.6 | 64.3 | 40.5 | 34.9 |
| linear-v2 | baseline | 8 | 0 | 53.3 | 1.318 | 66.0 | 44.7 | 61.0 | 75.5 | 40.5 | 34.9 |
| linear-v2 | default | 8 | 0 | 60.1 | 1.486 | 70.0 | 53.3 | 65.7 | 91.3 | 40.5 | 34.9 |
| near-miss-a | baseline | 8 | 0 | 53.3 | 1.318 | 66.0 | 44.7 | 61.0 | 75.5 | 40.5 | 34.9 |
| near-miss-a | bayes | 8 | 0 | 47.4 | 1.171 | 55.3 | 42.0 | 49.7 | 66.0 | 40.5 | 34.9 |
| near-miss-a | default | 8 | 0 | 49.1 | 1.215 | 57.6 | 43.4 | 52.3 | 69.1 | 40.5 | 34.9 |
| near-miss-b | baseline | 8 | 0 | 57.9 | 1.432 | 68.6 | 50.6 | 63.6 | 85.0 | 40.5 | 34.9 |
| near-miss-b | bayes | 8 | 0 | 50.4 | 1.245 | 57.4 | 45.7 | 51.3 | 71.2 | 40.5 | 34.9 |
| near-miss-b | default | 8 | 0 | 42.5 | 1.050 | 50.3 | 37.3 | 43.7 | 54.2 | 40.5 | 34.9 |
| near-miss-c | baseline | 8 | 0 | 53.3 | 1.318 | 66.0 | 44.7 | 61.0 | 75.5 | 40.5 | 34.9 |
| near-miss-c | bayes | 8 | 0 | 41.4 | 1.024 | 49.3 | 36.2 | 43.1 | 53.7 | 40.5 | 34.9 |
| near-miss-c | default | 8 | 0 | 41.2 | 1.017 | 49.1 | 36.0 | 42.9 | 52.4 | 40.5 | 34.9 |
| pfss-dhcs | baseline | 8 | 0 | 53.3 | 1.318 | 66.0 | 44.7 | 61.0 | 75.5 | 40.5 | 34.9 |
| pfss-dhcs | default | 8 | 0 | 46.5 | 1.150 | 54.5 | 41.2 | 48.6 | 64.3 | 40.5 | 34.9 |
| pfss-shadow | baseline | 8 | 0 | 53.3 | 1.318 | 66.0 | 44.7 | 61.0 | 75.5 | 40.5 | 34.9 |
| pfss-shadow | default | 8 | 0 | 60.1 | 1.486 | 70.0 | 53.3 | 65.7 | 91.3 | 40.5 | 34.9 |
| pfss-shadow-2 | baseline | 8 | 0 | 53.3 | 1.318 | 66.0 | 44.7 | 61.0 | 75.5 | 40.5 | 34.9 |
| pfss-shadow-2 | default | 8 | 0 | 60.1 | 1.486 | 70.0 | 53.3 | 65.7 | 91.3 | 40.5 | 34.9 |
| pfss-shadow-3 | baseline | 8 | 0 | 53.3 | 1.318 | 66.0 | 44.7 | 61.0 | 75.5 | 40.5 | 34.9 |
| pfss-shadow-3 | default | 8 | 0 | 60.1 | 1.486 | 70.0 | 53.3 | 65.7 | 91.3 | 40.5 | 34.9 |
| submitted | baseline | 8 | 0 | 54.8 | 1.356 | 67.8 | 46.3 | 63.0 | 76.1 | 40.5 | 34.9 |
| submitted | bayes | 8 | 0 | 42.1 | 1.040 | 50.1 | 36.9 | 44.0 | 53.4 | 40.5 | 34.9 |
| submitted | default | 8 | 0 | 42.1 | 1.040 | 50.1 | 36.9 | 44.0 | 53.4 | 40.5 | 34.9 |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 28 | 1 | 0 | 7.6 | 9.1 | 5.5 | — | — | — |
| bayes-option | bayes | 28 | 1 | 0 | 7.1 | 22.1 | 5.5 | 100.0% | — | — |
| bayes-option | default | 28 | 1 | 0 | 9.7 | 40.2 | 5.5 | 100.0% | — | — |
| benchlab | dryrun | 1 | 0 | 0 | 43.0 | 43.0 | 5.6 | 100.0% | — | — |
| benchlab | live | 20 | 0 | 0 | 42.5 | 50.0 | 2.9 | 100.0% | — | — |
| freeze | baseline | 28 | 1 | 0 | 7.7 | 25.9 | 5.6 | — | — | — |
| freeze | bayes | 28 | 1 | 0 | 7.1 | 10.6 | 5.6 | 100.0% | — | — |
| freeze | default | 28 | 1 | 0 | 9.8 | 11.5 | 5.6 | 100.0% | — | — |
| linear-v2 | baseline | 28 | 1 | 0 | 7.3 | 246.7 | 5.5 | — | — | — |
| linear-v2 | default | 28 | 1 | 0 | 7.4 | 247.2 | 5.5 | 94.0% | — | — |
| near-miss-a | baseline | 28 | 1 | 0 | 6.1 | 9.5 | 5.5 | — | — | — |
| near-miss-a | bayes | 28 | 1 | 0 | 7.1 | 10.7 | 5.5 | 100.0% | — | — |
| near-miss-a | default | 28 | 1 | 0 | 9.7 | 11.7 | 5.5 | 100.0% | — | — |
| near-miss-b | baseline | 28 | 1 | 0 | 7.9 | 246.7 | 5.6 | — | — | — |
| near-miss-b | bayes | 28 | 1 | 0 | 6.9 | 165.0 | 5.6 | 98.3% | — | — |
| near-miss-b | default | 28 | 1 | 0 | 9.5 | 247.4 | 5.6 | 98.9% | — | — |
| near-miss-c | baseline | 27 | 1 | 0 | 7.1 | 23.9 | 5.5 | — | — | — |
| near-miss-c | bayes | 27 | 1 | 0 | 7.0 | 10.2 | 5.5 | 100.0% | — | — |
| near-miss-c | default | 27 | 1 | 0 | 9.7 | 11.1 | 5.5 | 100.0% | — | — |
| pfss-dhcs | baseline | 28 | 1 | 0 | 7.5 | 9.9 | 5.5 | — | — | — |
| pfss-dhcs | default | 28 | 1 | 0 | 9.9 | 14.9 | 5.5 | 100.0% | — | — |
| pfss-shadow | baseline | 28 | 1 | 0 | 8.0 | 23.2 | 5.6 | — | — | — |
| pfss-shadow | default | 28 | 1 | 0 | 42.8 | 43.1 | 5.6 | 100.0% | pfss ×27 | — |
| pfss-shadow-2 | baseline | 28 | 1 | 0 | 3.9 | 9.3 | 5.5 | — | — | — |
| pfss-shadow-2 | default | 28 | 1 | 0 | 47.0 | 47.5 | 5.5 | 100.0% | pfss ×27 | — |
| pfss-shadow-3 | baseline | 28 | 1 | 0 | 5.9 | 8.2 | 5.5 | — | — | — |
| pfss-shadow-3 | default | 28 | 1 | 0 | 10.2 | 11.6 | 5.5 | 100.0% | — | — |
| submitted | baseline | 20 | 0 | 0 | 7.6 | 246.7 | 5.5 | — | — | — |
| submitted | bayes | 20 | 0 | 0 | 7.0 | 164.9 | 5.5 | 98.5% | — | — |
| submitted | default | 20 | 0 | 0 | 7.7 | 165.7 | 5.5 | 98.5% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 29 | 1 | 0 | 7.8 | 9.1 | 5.5 | — | — | — |
| bayes-option | bayes | 29 | 1 | 0 | 7.0 | 22.1 | 5.5 | 100.0% | — | — |
| bayes-option | default | 29 | 1 | 0 | 9.8 | 40.2 | 5.5 | 100.0% | — | — |
| benchlab | dryrun | 1 | 0 | 0 | 43.0 | 43.0 | 5.6 | 100.0% | — | — |
| benchlab | live | 20 | 0 | 0 | 42.5 | 50.0 | 2.9 | 100.0% | — | — |
| freeze | baseline | 28 | 1 | 0 | 7.7 | 25.9 | 5.6 | — | — | — |
| freeze | bayes | 28 | 1 | 0 | 7.1 | 10.6 | 5.6 | 100.0% | — | — |
| freeze | default | 28 | 1 | 0 | 9.8 | 11.5 | 5.6 | 100.0% | — | — |
| linear-v2 | baseline | 116 | 1 | 0 | 6.7 | 246.7 | 5.5 | — | — | — |
| linear-v2 | default | 116 | 1 | 0 | 6.9 | 247.2 | 5.5 | 98.6% | — | — |
| near-miss-a | baseline | 28 | 1 | 0 | 6.1 | 9.5 | 5.5 | — | — | — |
| near-miss-a | bayes | 28 | 1 | 0 | 7.1 | 10.7 | 5.5 | 100.0% | — | — |
| near-miss-a | default | 28 | 1 | 0 | 9.7 | 11.7 | 5.5 | 100.0% | — | — |
| near-miss-b | baseline | 28 | 1 | 0 | 7.9 | 246.7 | 5.6 | — | — | — |
| near-miss-b | bayes | 28 | 1 | 0 | 6.9 | 165.0 | 5.6 | 98.3% | — | — |
| near-miss-b | default | 28 | 1 | 0 | 9.5 | 247.4 | 5.6 | 98.9% | — | — |
| near-miss-c | baseline | 27 | 1 | 0 | 7.1 | 23.9 | 5.5 | — | — | — |
| near-miss-c | bayes | 27 | 1 | 0 | 7.0 | 10.2 | 5.5 | 100.0% | — | — |
| near-miss-c | default | 27 | 1 | 0 | 9.7 | 11.1 | 5.5 | 100.0% | — | — |
| pfss-dhcs | baseline | 67 | 1 | 0 | 7.0 | 22.4 | 5.4 | — | — | — |
| pfss-dhcs | default | 67 | 1 | 0 | 10.2 | 137.2 | 5.4 | 100.0% | — | — |
| pfss-shadow | baseline | 69 | 1 | 0 | 7.0 | 23.2 | 5.4 | — | — | — |
| pfss-shadow | default | 69 | 1 | 0 | 42.8 | 95.1 | 5.4 | 100.0% | pfss ×68 | — |
| pfss-shadow-2 | baseline | 69 | 1 | 0 | 6.2 | 9.3 | 5.4 | — | — | — |
| pfss-shadow-2 | default | 69 | 1 | 0 | 47.0 | 105.6 | 5.4 | 100.0% | pfss ×68 | — |
| pfss-shadow-3 | baseline | 69 | 1 | 0 | 5.9 | 9.9 | 5.4 | — | — | — |
| pfss-shadow-3 | default | 69 | 1 | 0 | 9.8 | 81.6 | 5.4 | 100.0% | — | — |
| submitted | baseline | 20 | 0 | 0 | 7.6 | 246.7 | 5.5 | — | — | — |
| submitted | bayes | 20 | 0 | 0 | 7.0 | 164.9 | 5.5 | 98.5% | — | — |
| submitted | default | 20 | 0 | 0 | 7.7 | 165.7 | 5.5 | 98.5% | — | — |

## Schedule

Last cycle: 2026-11-24 12:00 UTC
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
| 20261123T060000Z | linear-v2 | default | few regressors | 3/16 observed; missing ['ch_cm_at_0h', 'ch_cm_at_6h', 'ch_cm_at_12h', 'ch_cm_at_18h', 'ch_cm_at_24h', 'ch_cm_at_30h', 'ch_cm_at_36h', 'ch_cm_at_42h', 'ch_cm_at_48h', 'ch_cm_at_54h', 'ch_cm_at_60h', 'ch_cm_at_66h', 'ch_cm_at_72h'] |
| 20261124T000000Z | linear-v2 | default | few regressors | 3/16 observed; missing ['ch_cm_at_0h', 'ch_cm_at_6h', 'ch_cm_at_12h', 'ch_cm_at_18h', 'ch_cm_at_24h', 'ch_cm_at_30h', 'ch_cm_at_36h', 'ch_cm_at_42h', 'ch_cm_at_48h', 'ch_cm_at_54h', 'ch_cm_at_60h', 'ch_cm_at_66h', 'ch_cm_at_72h'] |

## Other events, last 7 days (counts; details in events/)

source not ok: 54, vs staging: 20, no forecast in log: 8, candidate added: 5, eligibility: 1

