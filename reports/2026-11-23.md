# Staging report 2026-11-23

Generated 2026-11-23 13:35 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-11-23 12:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 12 | 1 | 40.5 | 0.806 | 51.0 | 34.0 | 43.4 | 49.1 | 50.2 | 24.0 |
| bayes-option | bayes | 12 | 1 | 43.2 | 0.860 | 53.2 | 37.1 | 38.6 | 46.8 | 50.2 | 24.0 |
| bayes-option | default | 12 | 1 | 42.9 | 0.854 | 52.6 | 37.0 | 38.4 | 45.8 | 50.2 | 24.0 |
| benchlab | dryrun | 1 | 0 | 0.0 | — | 69.9 | 0.0 | 69.7 | 0.0 | 0.0 | 0.0 |
| benchlab | live | 3 | 0 | 49.2 | 0.930 | 55.9 | 46.5 | 53.5 | 39.2 | 53.0 | 24.5 |
| freeze | baseline | 11 | 1 | 42.1 | 0.945 | 52.5 | 35.6 | 45.3 | 51.4 | 44.5 | 22.6 |
| freeze | bayes | 11 | 1 | 43.9 | 0.986 | 54.2 | 37.5 | 43.5 | 49.0 | 44.5 | 22.6 |
| freeze | default | 11 | 1 | 43.6 | 0.980 | 53.6 | 37.5 | 43.2 | 48.1 | 44.5 | 22.6 |
| linear-v2 | baseline | 99 | 1 | 109.6 | 0.795 | 84.0 | 143.3 | 26.5 | 138.9 | 137.8 | 131.1 |
| linear-v2 | default | 99 | 1 | 92.1 | 0.668 | 72.5 | 117.8 | 24.8 | 124.1 | 137.8 | 131.1 |
| near-miss-a | baseline | 11 | 1 | 42.1 | 0.945 | 52.5 | 35.6 | 45.3 | 51.4 | 44.5 | 22.6 |
| near-miss-a | bayes | 11 | 1 | 43.9 | 0.986 | 54.2 | 37.5 | 43.5 | 49.0 | 44.5 | 22.6 |
| near-miss-a | default | 11 | 1 | 46.7 | 1.050 | 56.9 | 40.5 | 48.3 | 54.1 | 44.5 | 22.6 |
| near-miss-b | baseline | 11 | 1 | 44.3 | 0.995 | 55.9 | 37.5 | 48.0 | 55.4 | 44.5 | 22.6 |
| near-miss-b | bayes | 11 | 1 | 45.3 | 1.017 | 56.4 | 38.7 | 45.1 | 51.3 | 44.5 | 22.6 |
| near-miss-b | default | 11 | 1 | 41.2 | 0.926 | 51.4 | 34.7 | 41.4 | 44.2 | 44.5 | 22.6 |
| near-miss-c | baseline | 10 | 1 | 43.6 | 1.053 | 54.0 | 37.1 | 47.3 | 51.7 | 41.4 | 22.2 |
| near-miss-c | bayes | 10 | 1 | 42.2 | 1.019 | 51.7 | 36.1 | 43.4 | 46.2 | 41.4 | 22.2 |
| near-miss-c | default | 10 | 1 | 42.4 | 1.024 | 51.9 | 36.3 | 44.1 | 45.6 | 41.4 | 22.2 |
| pfss-dhcs | baseline | 50 | 1 | 65.0 | 0.793 | 69.7 | 65.7 | 33.7 | 70.2 | 81.9 | 80.8 |
| pfss-dhcs | default | 50 | 1 | 61.4 | 0.749 | 62.2 | 66.0 | 46.2 | 77.9 | 81.9 | 80.8 |
| pfss-shadow | baseline | 52 | 1 | 86.5 | 0.914 | 77.8 | 98.1 | 24.2 | 83.8 | 94.7 | 95.7 |
| pfss-shadow | default | 52 | 1 | 73.3 | 0.774 | 70.1 | 78.3 | 49.8 | 97.2 | 94.7 | 95.7 |
| pfss-shadow-2 | baseline | 52 | 1 | 86.5 | 0.914 | 77.8 | 98.1 | 24.2 | 83.8 | 94.7 | 95.7 |
| pfss-shadow-2 | default | 52 | 1 | 73.3 | 0.774 | 70.1 | 78.3 | 49.8 | 97.2 | 94.7 | 95.7 |
| pfss-shadow-3 | baseline | 52 | 1 | 86.5 | 0.914 | 77.8 | 98.1 | 24.2 | 83.8 | 94.7 | 95.7 |
| pfss-shadow-3 | default | 52 | 1 | 73.3 | 0.774 | 70.1 | 78.3 | 49.8 | 97.2 | 94.7 | 95.7 |
| submitted | baseline | 3 | 0 | 55.4 | 1.046 | 62.8 | 52.1 | 60.9 | 49.1 | 53.0 | 24.5 |
| submitted | bayes | 3 | 0 | 48.6 | 0.917 | 55.1 | 46.0 | 52.6 | 38.5 | 53.0 | 24.5 |
| submitted | default | 3 | 0 | 48.6 | 0.917 | 55.1 | 46.0 | 52.6 | 38.5 | 53.0 | 24.5 |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 3 | 0 | 55.4 | 1.046 | 62.8 | 52.1 | 60.9 | 49.1 | 53.0 | 24.5 |
| bayes-option | bayes | 3 | 0 | 55.1 | 1.040 | 61.8 | 52.8 | 59.6 | 49.2 | 53.0 | 24.5 |
| bayes-option | default | 3 | 0 | 54.0 | 1.019 | 60.5 | 51.7 | 58.3 | 47.1 | 53.0 | 24.5 |
| benchlab | live | 3 | 0 | 49.2 | 0.930 | 55.9 | 46.5 | 53.5 | 39.2 | 53.0 | 24.5 |
| freeze | baseline | 3 | 0 | 55.4 | 1.046 | 62.8 | 52.1 | 60.9 | 49.1 | 53.0 | 24.5 |
| freeze | bayes | 3 | 0 | 55.1 | 1.040 | 61.8 | 52.8 | 59.6 | 49.2 | 53.0 | 24.5 |
| freeze | default | 3 | 0 | 54.0 | 1.019 | 60.5 | 51.7 | 58.3 | 47.1 | 53.0 | 24.5 |
| linear-v2 | baseline | 3 | 0 | 55.4 | 1.046 | 62.8 | 52.1 | 60.9 | 49.1 | 53.0 | 24.5 |
| linear-v2 | default | 3 | 0 | 65.6 | 1.238 | 73.1 | 63.1 | 71.3 | 69.1 | 53.0 | 24.5 |
| near-miss-a | baseline | 3 | 0 | 55.4 | 1.046 | 62.8 | 52.1 | 60.9 | 49.1 | 53.0 | 24.5 |
| near-miss-a | bayes | 3 | 0 | 55.1 | 1.040 | 61.8 | 52.8 | 59.6 | 49.2 | 53.0 | 24.5 |
| near-miss-a | default | 3 | 0 | 55.8 | 1.053 | 62.6 | 53.3 | 60.5 | 51.1 | 53.0 | 24.5 |
| near-miss-b | baseline | 3 | 0 | 63.2 | 1.194 | 71.1 | 60.8 | 68.3 | 61.3 | 53.0 | 24.5 |
| near-miss-b | bayes | 3 | 0 | 60.3 | 1.139 | 67.4 | 58.5 | 64.4 | 55.8 | 53.0 | 24.5 |
| near-miss-b | default | 3 | 0 | 51.0 | 0.963 | 58.1 | 48.5 | 54.8 | 39.1 | 53.0 | 24.5 |
| near-miss-c | baseline | 3 | 0 | 55.4 | 1.046 | 62.8 | 52.1 | 60.9 | 49.1 | 53.0 | 24.5 |
| near-miss-c | bayes | 3 | 0 | 48.6 | 0.917 | 55.1 | 46.0 | 52.6 | 38.5 | 53.0 | 24.5 |
| near-miss-c | default | 3 | 0 | 48.4 | 0.914 | 54.8 | 45.9 | 52.3 | 37.1 | 53.0 | 24.5 |
| pfss-dhcs | baseline | 3 | 0 | 55.4 | 1.046 | 62.8 | 52.1 | 60.9 | 49.1 | 53.0 | 24.5 |
| pfss-dhcs | default | 3 | 0 | 54.0 | 1.019 | 60.5 | 51.7 | 58.3 | 47.1 | 53.0 | 24.5 |
| pfss-shadow | baseline | 3 | 0 | 55.4 | 1.046 | 62.8 | 52.1 | 60.9 | 49.1 | 53.0 | 24.5 |
| pfss-shadow | default | 3 | 0 | 65.6 | 1.238 | 73.1 | 63.1 | 71.3 | 69.1 | 53.0 | 24.5 |
| pfss-shadow-2 | baseline | 3 | 0 | 55.4 | 1.046 | 62.8 | 52.1 | 60.9 | 49.1 | 53.0 | 24.5 |
| pfss-shadow-2 | default | 3 | 0 | 65.6 | 1.238 | 73.1 | 63.1 | 71.3 | 69.1 | 53.0 | 24.5 |
| pfss-shadow-3 | baseline | 3 | 0 | 55.4 | 1.046 | 62.8 | 52.1 | 60.9 | 49.1 | 53.0 | 24.5 |
| pfss-shadow-3 | default | 3 | 0 | 65.6 | 1.238 | 73.1 | 63.1 | 71.3 | 69.1 | 53.0 | 24.5 |
| submitted | baseline | 3 | 0 | 55.4 | 1.046 | 62.8 | 52.1 | 60.9 | 49.1 | 53.0 | 24.5 |
| submitted | bayes | 3 | 0 | 48.6 | 0.917 | 55.1 | 46.0 | 52.6 | 38.5 | 53.0 | 24.5 |
| submitted | default | 3 | 0 | 48.6 | 0.917 | 55.1 | 46.0 | 52.6 | 38.5 | 53.0 | 24.5 |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 25 | 1 | 0 | 7.5 | 8.9 | 5.5 | — | — | — |
| bayes-option | bayes | 25 | 1 | 0 | 7.1 | 22.1 | 5.5 | 100.0% | — | — |
| bayes-option | default | 25 | 1 | 0 | 9.9 | 40.2 | 5.5 | 100.0% | — | — |
| benchlab | dryrun | 1 | 0 | 0 | 43.0 | 43.0 | 5.6 | 100.0% | — | — |
| benchlab | live | 16 | 0 | 0 | 44.5 | 50.0 | 2.7 | 100.0% | — | — |
| freeze | baseline | 24 | 1 | 0 | 8.0 | 25.9 | 5.6 | — | — | — |
| freeze | bayes | 24 | 1 | 0 | 7.1 | 10.6 | 5.6 | 100.0% | — | — |
| freeze | default | 24 | 1 | 0 | 9.8 | 11.5 | 5.6 | 100.0% | — | — |
| linear-v2 | baseline | 28 | 1 | 0 | 7.3 | 246.7 | 5.5 | — | — | — |
| linear-v2 | default | 28 | 1 | 0 | 7.1 | 247.2 | 5.5 | 97.0% | — | — |
| near-miss-a | baseline | 24 | 1 | 0 | 6.1 | 9.5 | 5.5 | — | — | — |
| near-miss-a | bayes | 24 | 1 | 0 | 7.1 | 10.7 | 5.5 | 100.0% | — | — |
| near-miss-a | default | 24 | 1 | 0 | 9.6 | 11.7 | 5.5 | 100.0% | — | — |
| near-miss-b | baseline | 24 | 1 | 0 | 7.9 | 246.7 | 5.6 | — | — | — |
| near-miss-b | bayes | 24 | 1 | 0 | 7.0 | 165.0 | 5.6 | 98.1% | — | — |
| near-miss-b | default | 24 | 1 | 0 | 9.7 | 247.4 | 5.6 | 98.7% | — | — |
| near-miss-c | baseline | 23 | 1 | 0 | 7.1 | 8.8 | 5.5 | — | — | — |
| near-miss-c | bayes | 23 | 1 | 0 | 7.0 | 10.2 | 5.5 | 100.0% | — | — |
| near-miss-c | default | 23 | 1 | 0 | 9.7 | 11.1 | 5.5 | 100.0% | — | — |
| pfss-dhcs | baseline | 28 | 1 | 0 | 7.5 | 9.7 | 5.5 | — | — | — |
| pfss-dhcs | default | 28 | 1 | 0 | 9.8 | 14.9 | 5.5 | 100.0% | — | — |
| pfss-shadow | baseline | 28 | 1 | 0 | 7.9 | 23.2 | 5.6 | — | — | — |
| pfss-shadow | default | 28 | 1 | 0 | 42.7 | 43.0 | 5.6 | 100.0% | pfss ×27 | — |
| pfss-shadow-2 | baseline | 28 | 1 | 0 | 4.1 | 9.3 | 5.5 | — | — | — |
| pfss-shadow-2 | default | 28 | 1 | 0 | 47.0 | 47.4 | 5.5 | 100.0% | pfss ×27 | — |
| pfss-shadow-3 | baseline | 28 | 1 | 0 | 6.9 | 9.9 | 5.5 | — | — | — |
| pfss-shadow-3 | default | 28 | 1 | 0 | 9.8 | 11.4 | 5.5 | 100.0% | — | — |
| submitted | baseline | 16 | 0 | 0 | 7.6 | 246.7 | 5.5 | — | — | — |
| submitted | bayes | 16 | 0 | 0 | 7.2 | 164.9 | 5.5 | 98.1% | — | — |
| submitted | default | 16 | 0 | 0 | 7.7 | 165.7 | 5.5 | 98.1% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 25 | 1 | 0 | 7.5 | 8.9 | 5.5 | — | — | — |
| bayes-option | bayes | 25 | 1 | 0 | 7.1 | 22.1 | 5.5 | 100.0% | — | — |
| bayes-option | default | 25 | 1 | 0 | 9.9 | 40.2 | 5.5 | 100.0% | — | — |
| benchlab | dryrun | 1 | 0 | 0 | 43.0 | 43.0 | 5.6 | 100.0% | — | — |
| benchlab | live | 16 | 0 | 0 | 44.5 | 50.0 | 2.7 | 100.0% | — | — |
| freeze | baseline | 24 | 1 | 0 | 8.0 | 25.9 | 5.6 | — | — | — |
| freeze | bayes | 24 | 1 | 0 | 7.1 | 10.6 | 5.6 | 100.0% | — | — |
| freeze | default | 24 | 1 | 0 | 9.8 | 11.5 | 5.6 | 100.0% | — | — |
| linear-v2 | baseline | 112 | 1 | 0 | 6.7 | 246.7 | 5.4 | — | — | — |
| linear-v2 | default | 112 | 1 | 0 | 6.9 | 247.2 | 5.4 | 99.3% | — | — |
| near-miss-a | baseline | 24 | 1 | 0 | 6.1 | 9.5 | 5.5 | — | — | — |
| near-miss-a | bayes | 24 | 1 | 0 | 7.1 | 10.7 | 5.5 | 100.0% | — | — |
| near-miss-a | default | 24 | 1 | 0 | 9.6 | 11.7 | 5.5 | 100.0% | — | — |
| near-miss-b | baseline | 24 | 1 | 0 | 7.9 | 246.7 | 5.6 | — | — | — |
| near-miss-b | bayes | 24 | 1 | 0 | 7.0 | 165.0 | 5.6 | 98.1% | — | — |
| near-miss-b | default | 24 | 1 | 0 | 9.7 | 247.4 | 5.6 | 98.7% | — | — |
| near-miss-c | baseline | 23 | 1 | 0 | 7.1 | 8.8 | 5.5 | — | — | — |
| near-miss-c | bayes | 23 | 1 | 0 | 7.0 | 10.2 | 5.5 | 100.0% | — | — |
| near-miss-c | default | 23 | 1 | 0 | 9.7 | 11.1 | 5.5 | 100.0% | — | — |
| pfss-dhcs | baseline | 63 | 1 | 0 | 6.9 | 22.4 | 5.4 | — | — | — |
| pfss-dhcs | default | 63 | 1 | 0 | 10.3 | 137.2 | 5.4 | 100.0% | — | — |
| pfss-shadow | baseline | 65 | 1 | 0 | 7.0 | 23.2 | 5.4 | — | — | — |
| pfss-shadow | default | 65 | 1 | 0 | 42.8 | 95.1 | 5.4 | 100.0% | pfss ×64 | — |
| pfss-shadow-2 | baseline | 65 | 1 | 0 | 6.1 | 9.3 | 5.4 | — | — | — |
| pfss-shadow-2 | default | 65 | 1 | 0 | 47.0 | 105.6 | 5.4 | 100.0% | pfss ×64 | — |
| pfss-shadow-3 | baseline | 65 | 1 | 0 | 6.0 | 9.9 | 5.4 | — | — | — |
| pfss-shadow-3 | default | 65 | 1 | 0 | 9.8 | 81.6 | 5.4 | 100.0% | — | — |
| submitted | baseline | 16 | 0 | 0 | 7.6 | 246.7 | 5.5 | — | — | — |
| submitted | bayes | 16 | 0 | 0 | 7.2 | 164.9 | 5.5 | 98.1% | — | — |
| submitted | default | 16 | 0 | 0 | 7.7 | 165.7 | 5.5 | 98.1% | — | — |

## Schedule

Last cycle: 2026-11-23 12:00 UTC
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

## Other events, last 7 days (counts; details in events/)

source not ok: 54, vs staging: 16, no forecast in log: 12, candidate added: 6, eligibility: 1

