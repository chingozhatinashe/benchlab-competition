# Staging report 2026-11-25

Generated 2026-11-25 13:25 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-11-25 12:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 20 | 1 | 47.9 | 1.104 | 61.8 | 38.1 | 52.8 | 72.8 | 43.4 | 33.3 |
| bayes-option | bayes | 20 | 1 | 44.7 | 1.030 | 53.6 | 38.5 | 40.5 | 62.7 | 43.4 | 33.3 |
| bayes-option | default | 20 | 1 | 44.3 | 1.021 | 53.1 | 38.2 | 40.1 | 61.7 | 43.4 | 33.3 |
| benchlab | dryrun | 1 | 0 | 0.0 | — | 69.9 | 0.0 | 69.7 | 0.0 | 0.0 | 0.0 |
| benchlab | live | 11 | 0 | 43.2 | 1.152 | 53.4 | 36.7 | 43.8 | 56.9 | 37.5 | 39.9 |
| freeze | baseline | 19 | 1 | 49.2 | 1.248 | 62.8 | 39.5 | 54.1 | 75.1 | 39.4 | 33.0 |
| freeze | bayes | 19 | 1 | 45.2 | 1.147 | 54.0 | 39.0 | 43.0 | 64.6 | 39.4 | 33.0 |
| freeze | default | 19 | 1 | 44.8 | 1.137 | 53.6 | 38.7 | 42.5 | 63.6 | 39.4 | 33.0 |
| linear-v2 | baseline | 107 | 1 | 108.6 | 0.810 | 83.1 | 142.3 | 30.8 | 139.0 | 134.1 | 128.3 |
| linear-v2 | default | 107 | 1 | 92.1 | 0.687 | 73.1 | 117.0 | 28.9 | 125.5 | 134.1 | 128.3 |
| near-miss-a | baseline | 19 | 1 | 49.2 | 1.248 | 62.8 | 39.5 | 54.1 | 75.1 | 39.4 | 33.0 |
| near-miss-a | bayes | 19 | 1 | 45.2 | 1.147 | 54.0 | 39.0 | 43.0 | 64.6 | 39.4 | 33.0 |
| near-miss-a | default | 19 | 1 | 48.8 | 1.239 | 58.7 | 41.9 | 48.7 | 70.2 | 39.4 | 33.0 |
| near-miss-b | baseline | 19 | 1 | 50.9 | 1.291 | 64.0 | 41.6 | 55.3 | 78.8 | 39.4 | 33.0 |
| near-miss-b | bayes | 19 | 1 | 46.3 | 1.175 | 55.0 | 40.3 | 43.7 | 66.8 | 39.4 | 33.0 |
| near-miss-b | default | 19 | 1 | 42.6 | 1.080 | 51.9 | 36.2 | 40.8 | 57.8 | 39.4 | 33.0 |
| near-miss-c | baseline | 18 | 1 | 50.3 | 1.347 | 63.7 | 40.8 | 55.4 | 75.9 | 37.4 | 33.3 |
| near-miss-c | bayes | 18 | 1 | 43.1 | 1.152 | 51.9 | 37.0 | 41.5 | 59.3 | 37.4 | 33.3 |
| near-miss-c | default | 18 | 1 | 43.2 | 1.156 | 52.1 | 37.0 | 42.1 | 58.8 | 37.4 | 33.3 |
| pfss-dhcs | baseline | 58 | 1 | 69.1 | 0.834 | 71.3 | 68.7 | 45.3 | 79.4 | 82.9 | 83.8 |
| pfss-dhcs | default | 58 | 1 | 66.1 | 0.797 | 61.4 | 71.6 | 45.5 | 83.8 | 82.9 | 83.8 |
| pfss-shadow | baseline | 60 | 1 | 84.5 | 0.926 | 77.0 | 95.0 | 37.3 | 86.6 | 91.2 | 94.1 |
| pfss-shadow | default | 60 | 1 | 75.9 | 0.832 | 71.7 | 81.1 | 52.9 | 100.2 | 91.2 | 94.1 |
| pfss-shadow-2 | baseline | 60 | 1 | 84.5 | 0.926 | 77.0 | 95.0 | 37.3 | 86.6 | 91.2 | 94.1 |
| pfss-shadow-2 | default | 60 | 1 | 75.9 | 0.832 | 71.7 | 81.1 | 52.9 | 100.3 | 91.2 | 94.1 |
| pfss-shadow-3 | baseline | 60 | 1 | 84.5 | 0.926 | 77.0 | 95.0 | 37.3 | 86.6 | 91.2 | 94.1 |
| pfss-shadow-3 | default | 60 | 1 | 75.9 | 0.832 | 71.7 | 81.1 | 52.9 | 100.2 | 91.2 | 94.1 |
| submitted | baseline | 11 | 0 | 57.3 | 1.526 | 70.9 | 47.7 | 64.2 | 82.0 | 37.5 | 39.9 |
| submitted | bayes | 11 | 0 | 43.8 | 1.166 | 53.6 | 37.3 | 44.2 | 56.3 | 37.5 | 39.9 |
| submitted | default | 11 | 0 | 43.8 | 1.166 | 53.6 | 37.3 | 44.2 | 56.3 | 37.5 | 39.9 |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 11 | 0 | 55.6 | 1.481 | 69.9 | 45.7 | 63.0 | 81.4 | 37.5 | 39.9 |
| bayes-option | bayes | 11 | 0 | 47.2 | 1.257 | 56.5 | 41.1 | 47.7 | 66.7 | 37.5 | 39.9 |
| bayes-option | default | 11 | 0 | 46.5 | 1.238 | 55.9 | 40.3 | 46.8 | 65.5 | 37.5 | 39.9 |
| benchlab | live | 11 | 0 | 43.2 | 1.152 | 53.4 | 36.7 | 43.8 | 56.9 | 37.5 | 39.9 |
| freeze | baseline | 11 | 0 | 55.6 | 1.481 | 69.9 | 45.7 | 63.0 | 81.4 | 37.5 | 39.9 |
| freeze | bayes | 11 | 0 | 47.2 | 1.257 | 56.5 | 41.1 | 47.7 | 66.7 | 37.5 | 39.9 |
| freeze | default | 11 | 0 | 46.5 | 1.238 | 55.9 | 40.3 | 46.8 | 65.5 | 37.5 | 39.9 |
| linear-v2 | baseline | 11 | 0 | 55.6 | 1.481 | 69.9 | 45.7 | 63.0 | 81.4 | 37.5 | 39.9 |
| linear-v2 | default | 11 | 0 | 60.9 | 1.622 | 72.2 | 52.8 | 65.0 | 92.9 | 37.5 | 39.9 |
| near-miss-a | baseline | 11 | 0 | 55.6 | 1.481 | 69.9 | 45.7 | 63.0 | 81.4 | 37.5 | 39.9 |
| near-miss-a | bayes | 11 | 0 | 47.2 | 1.257 | 56.5 | 41.1 | 47.7 | 66.7 | 37.5 | 39.9 |
| near-miss-a | default | 11 | 0 | 50.4 | 1.343 | 61.4 | 43.0 | 52.8 | 70.8 | 37.5 | 39.9 |
| near-miss-b | baseline | 11 | 0 | 59.1 | 1.574 | 71.6 | 50.3 | 64.8 | 88.7 | 37.5 | 39.9 |
| near-miss-b | bayes | 11 | 0 | 49.6 | 1.321 | 57.9 | 44.2 | 48.8 | 71.0 | 37.5 | 39.9 |
| near-miss-b | default | 11 | 0 | 44.0 | 1.172 | 54.1 | 37.4 | 44.5 | 57.2 | 37.5 | 39.9 |
| near-miss-c | baseline | 11 | 0 | 55.6 | 1.481 | 69.9 | 45.7 | 63.0 | 81.4 | 37.5 | 39.9 |
| near-miss-c | bayes | 11 | 0 | 43.0 | 1.145 | 53.2 | 36.4 | 43.6 | 56.5 | 37.5 | 39.9 |
| near-miss-c | default | 11 | 0 | 42.9 | 1.144 | 53.4 | 36.2 | 43.9 | 55.8 | 37.5 | 39.9 |
| pfss-dhcs | baseline | 11 | 0 | 55.6 | 1.481 | 69.9 | 45.7 | 63.0 | 81.4 | 37.5 | 39.9 |
| pfss-dhcs | default | 11 | 0 | 46.5 | 1.238 | 55.9 | 40.3 | 46.8 | 65.5 | 37.5 | 39.9 |
| pfss-shadow | baseline | 11 | 0 | 55.6 | 1.481 | 69.9 | 45.7 | 63.0 | 81.4 | 37.5 | 39.9 |
| pfss-shadow | default | 11 | 0 | 60.9 | 1.622 | 72.2 | 52.8 | 65.0 | 92.9 | 37.5 | 39.9 |
| pfss-shadow-2 | baseline | 11 | 0 | 55.6 | 1.481 | 69.9 | 45.7 | 63.0 | 81.4 | 37.5 | 39.9 |
| pfss-shadow-2 | default | 11 | 0 | 60.9 | 1.622 | 72.2 | 52.8 | 65.0 | 92.9 | 37.5 | 39.9 |
| pfss-shadow-3 | baseline | 11 | 0 | 55.6 | 1.481 | 69.9 | 45.7 | 63.0 | 81.4 | 37.5 | 39.9 |
| pfss-shadow-3 | default | 11 | 0 | 60.9 | 1.622 | 72.2 | 52.8 | 65.0 | 92.9 | 37.5 | 39.9 |
| submitted | baseline | 11 | 0 | 57.3 | 1.526 | 70.9 | 47.7 | 64.2 | 82.0 | 37.5 | 39.9 |
| submitted | bayes | 11 | 0 | 43.8 | 1.166 | 53.6 | 37.3 | 44.2 | 56.3 | 37.5 | 39.9 |
| submitted | default | 11 | 0 | 43.8 | 1.166 | 53.6 | 37.3 | 44.2 | 56.3 | 37.5 | 39.9 |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 28 | 1 | 0 | 8.0 | 9.1 | 5.5 | — | — | — |
| bayes-option | bayes | 28 | 1 | 0 | 7.0 | 21.0 | 5.5 | 100.0% | — | — |
| bayes-option | default | 28 | 1 | 0 | 9.7 | 40.2 | 5.5 | 100.0% | — | — |
| benchlab | dryrun | 1 | 0 | 0 | 43.0 | 43.0 | 5.6 | 100.0% | — | — |
| benchlab | live | 24 | 0 | 0 | 42.0 | 50.0 | 2.7 | 100.0% | — | — |
| freeze | baseline | 28 | 1 | 0 | 7.1 | 8.5 | 5.5 | — | — | — |
| freeze | bayes | 28 | 1 | 0 | 7.1 | 10.3 | 5.5 | 100.0% | — | — |
| freeze | default | 28 | 1 | 0 | 9.8 | 11.6 | 5.5 | 100.0% | — | — |
| linear-v2 | baseline | 28 | 1 | 0 | 6.4 | 246.7 | 5.5 | — | — | — |
| linear-v2 | default | 28 | 1 | 0 | 6.4 | 247.2 | 5.5 | 94.0% | — | — |
| near-miss-a | baseline | 28 | 1 | 0 | 6.1 | 9.5 | 5.5 | — | — | — |
| near-miss-a | bayes | 28 | 1 | 0 | 7.0 | 10.7 | 5.5 | 100.0% | — | — |
| near-miss-a | default | 28 | 1 | 0 | 9.7 | 11.7 | 5.5 | 100.0% | — | — |
| near-miss-b | baseline | 28 | 1 | 0 | 7.2 | 246.7 | 5.5 | — | — | — |
| near-miss-b | bayes | 28 | 1 | 0 | 7.0 | 165.0 | 5.5 | 98.3% | — | — |
| near-miss-b | default | 28 | 1 | 0 | 9.5 | 247.4 | 5.5 | 98.9% | — | — |
| near-miss-c | baseline | 28 | 1 | 0 | 4.2 | 23.9 | 5.5 | — | — | — |
| near-miss-c | bayes | 28 | 1 | 0 | 6.9 | 10.2 | 5.5 | 100.0% | — | — |
| near-miss-c | default | 28 | 1 | 0 | 9.0 | 11.0 | 5.5 | 100.0% | — | — |
| pfss-dhcs | baseline | 28 | 1 | 0 | 7.6 | 9.9 | 5.5 | — | — | — |
| pfss-dhcs | default | 28 | 1 | 0 | 9.8 | 14.9 | 5.5 | 100.0% | — | — |
| pfss-shadow | baseline | 28 | 1 | 0 | 8.0 | 9.9 | 5.6 | — | — | — |
| pfss-shadow | default | 28 | 1 | 0 | 42.7 | 43.1 | 5.6 | 100.0% | pfss ×27 | — |
| pfss-shadow-2 | baseline | 28 | 1 | 0 | 6.5 | 9.3 | 5.5 | — | — | — |
| pfss-shadow-2 | default | 28 | 1 | 0 | 47.0 | 47.5 | 5.5 | 100.0% | pfss ×27 | — |
| pfss-shadow-3 | baseline | 28 | 1 | 0 | 5.8 | 8.2 | 5.5 | — | — | — |
| pfss-shadow-3 | default | 28 | 1 | 0 | 10.4 | 11.6 | 5.5 | 100.0% | — | — |
| submitted | baseline | 24 | 0 | 0 | 7.4 | 246.7 | 5.5 | — | — | — |
| submitted | bayes | 24 | 0 | 0 | 7.1 | 164.9 | 5.5 | 98.7% | — | — |
| submitted | default | 24 | 0 | 0 | 7.7 | 165.7 | 5.5 | 98.7% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 33 | 1 | 0 | 7.8 | 9.1 | 5.5 | — | — | — |
| bayes-option | bayes | 33 | 1 | 0 | 7.0 | 22.1 | 5.5 | 100.0% | — | — |
| bayes-option | default | 33 | 1 | 0 | 9.8 | 40.2 | 5.5 | 100.0% | — | — |
| benchlab | dryrun | 1 | 0 | 0 | 43.0 | 43.0 | 5.6 | 100.0% | — | — |
| benchlab | live | 24 | 0 | 0 | 42.0 | 50.0 | 2.7 | 100.0% | — | — |
| freeze | baseline | 32 | 1 | 0 | 7.1 | 25.9 | 5.5 | — | — | — |
| freeze | bayes | 32 | 1 | 0 | 7.1 | 10.6 | 5.5 | 100.0% | — | — |
| freeze | default | 32 | 1 | 0 | 9.8 | 11.6 | 5.5 | 100.0% | — | — |
| linear-v2 | baseline | 120 | 1 | 0 | 6.7 | 246.7 | 5.5 | — | — | — |
| linear-v2 | default | 120 | 1 | 0 | 6.9 | 247.2 | 5.5 | 98.6% | — | — |
| near-miss-a | baseline | 32 | 1 | 0 | 5.5 | 9.5 | 5.5 | — | — | — |
| near-miss-a | bayes | 32 | 1 | 0 | 7.1 | 10.7 | 5.5 | 100.0% | — | — |
| near-miss-a | default | 32 | 1 | 0 | 9.7 | 11.7 | 5.5 | 100.0% | — | — |
| near-miss-b | baseline | 32 | 1 | 0 | 7.2 | 246.7 | 5.6 | — | — | — |
| near-miss-b | bayes | 32 | 1 | 0 | 7.0 | 165.0 | 5.6 | 98.6% | — | — |
| near-miss-b | default | 32 | 1 | 0 | 9.7 | 247.4 | 5.6 | 99.0% | — | — |
| near-miss-c | baseline | 31 | 1 | 0 | 5.6 | 23.9 | 5.5 | — | — | — |
| near-miss-c | bayes | 31 | 1 | 0 | 7.0 | 10.2 | 5.5 | 100.0% | — | — |
| near-miss-c | default | 31 | 1 | 0 | 9.2 | 11.1 | 5.5 | 100.0% | — | — |
| pfss-dhcs | baseline | 71 | 1 | 0 | 7.0 | 22.4 | 5.4 | — | — | — |
| pfss-dhcs | default | 71 | 1 | 0 | 9.9 | 137.2 | 5.4 | 100.0% | — | — |
| pfss-shadow | baseline | 73 | 1 | 0 | 7.0 | 23.2 | 5.4 | — | — | — |
| pfss-shadow | default | 73 | 1 | 0 | 42.8 | 95.1 | 5.4 | 100.0% | pfss ×72 | — |
| pfss-shadow-2 | baseline | 73 | 1 | 0 | 6.4 | 9.3 | 5.4 | — | — | — |
| pfss-shadow-2 | default | 73 | 1 | 0 | 47.0 | 105.6 | 5.4 | 100.0% | pfss ×72 | — |
| pfss-shadow-3 | baseline | 73 | 1 | 0 | 5.7 | 9.9 | 5.4 | — | — | — |
| pfss-shadow-3 | default | 73 | 1 | 0 | 9.8 | 81.6 | 5.4 | 100.0% | — | — |
| submitted | baseline | 24 | 0 | 0 | 7.4 | 246.7 | 5.5 | — | — | — |
| submitted | bayes | 24 | 0 | 0 | 7.1 | 164.9 | 5.5 | 98.7% | — | — |
| submitted | default | 24 | 0 | 0 | 7.7 | 165.7 | 5.5 | 98.7% | — | — |

## Schedule

Last cycle: 2026-11-25 12:00 UTC
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

source not ok: 54, vs staging: 24, no forecast in log: 4, candidate added: 1, eligibility: 1

