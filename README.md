# Staging report 2026-11-20

Generated 2026-11-20 12:06 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-11-20 10:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 87 | 0 | 111.9 | 0.791 | 87.2 | 144.0 | 24.6 | 142.4 | 141.4 | 135.1 |
| linear-v2 | default | 87 | 0 | 93.4 | 0.661 | 74.3 | 118.1 | 22.3 | 126.1 | 141.4 | 135.1 |
| pfss-dhcs | baseline | 38 | 0 | 66.6 | 0.766 | 75.0 | 65.7 | 32.9 | 70.9 | 87.0 | 86.8 |
| pfss-dhcs | default | 38 | 0 | 62.1 | 0.714 | 65.7 | 65.6 | 48.9 | 79.3 | 87.0 | 86.8 |
| pfss-shadow | baseline | 40 | 0 | 84.2 | 0.882 | 83.8 | 93.3 | 19.7 | 69.9 | 95.4 | 97.7 |
| pfss-shadow | default | 40 | 0 | 70.4 | 0.739 | 74.0 | 73.9 | 52.4 | 85.5 | 95.4 | 97.7 |
| pfss-shadow-2 | baseline | 40 | 0 | 84.2 | 0.882 | 83.8 | 93.3 | 19.7 | 69.9 | 95.4 | 97.7 |
| pfss-shadow-2 | default | 40 | 0 | 70.4 | 0.739 | 74.0 | 73.9 | 52.4 | 85.5 | 95.4 | 97.7 |
| pfss-shadow-3 | baseline | 40 | 0 | 84.2 | 0.882 | 83.8 | 93.3 | 19.7 | 69.9 | 95.4 | 97.7 |
| pfss-shadow-3 | default | 40 | 0 | 70.4 | 0.739 | 74.0 | 73.9 | 52.4 | 85.5 | 95.4 | 97.7 |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 38 | 0 | 66.6 | 0.766 | 75.0 | 65.7 | 32.9 | 70.9 | 87.0 | 86.8 |
| linear-v2 | default | 38 | 0 | 67.5 | 0.776 | 72.4 | 70.3 | 56.6 | 93.5 | 87.0 | 86.8 |
| pfss-dhcs | baseline | 38 | 0 | 66.6 | 0.766 | 75.0 | 65.7 | 32.9 | 70.9 | 87.0 | 86.8 |
| pfss-dhcs | default | 38 | 0 | 62.1 | 0.714 | 65.7 | 65.6 | 48.9 | 79.3 | 87.0 | 86.8 |
| pfss-shadow | baseline | 38 | 0 | 66.6 | 0.766 | 75.0 | 65.7 | 32.9 | 70.9 | 87.0 | 86.8 |
| pfss-shadow | default | 38 | 0 | 67.5 | 0.776 | 72.4 | 70.3 | 56.6 | 93.5 | 87.0 | 86.8 |
| pfss-shadow-2 | baseline | 38 | 0 | 66.6 | 0.766 | 75.0 | 65.7 | 32.9 | 70.9 | 87.0 | 86.8 |
| pfss-shadow-2 | default | 38 | 0 | 67.6 | 0.777 | 72.4 | 70.3 | 56.6 | 93.5 | 87.0 | 86.8 |
| pfss-shadow-3 | baseline | 38 | 0 | 66.6 | 0.766 | 75.0 | 65.7 | 32.9 | 70.9 | 87.0 | 86.8 |
| pfss-shadow-3 | default | 38 | 0 | 67.5 | 0.776 | 72.4 | 70.3 | 56.6 | 93.5 | 87.0 | 86.8 |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 12 | 1 | 0 | 7.4 | 8.1 | 5.6 | — | — | — |
| bayes-option | bayes | 12 | 1 | 0 | 7.1 | 22.1 | 5.6 | 100.0% | — | — |
| bayes-option | default | 12 | 1 | 0 | 10.1 | 11.8 | 5.6 | 100.0% | — | — |
| benchlab | dryrun | 1 | 0 | 0 | 43.0 | 43.0 | 5.6 | 100.0% | — | — |
| benchlab | live | 3 | 0 | 0 | 42.0 | 50.0 | 5.2 | 100.0% | — | — |
| freeze | baseline | 11 | 1 | 0 | 7.8 | 25.9 | 5.7 | — | — | — |
| freeze | bayes | 11 | 1 | 0 | 7.1 | 10.6 | 5.7 | 100.0% | — | — |
| freeze | default | 11 | 1 | 0 | 9.6 | 11.5 | 5.7 | 100.0% | — | — |
| linear-v2 | baseline | 27 | 1 | 0 | 7.3 | 9.4 | 5.5 | — | — | — |
| linear-v2 | default | 27 | 1 | 0 | 6.9 | 101.8 | 5.5 | 100.0% | — | — |
| near-miss-a | baseline | 11 | 1 | 0 | 3.9 | 9.3 | 5.6 | — | — | — |
| near-miss-a | bayes | 11 | 1 | 0 | 7.2 | 10.7 | 5.6 | 100.0% | — | — |
| near-miss-a | default | 11 | 1 | 0 | 9.2 | 11.7 | 5.6 | 100.0% | — | — |
| near-miss-b | baseline | 11 | 1 | 0 | 8.0 | 246.7 | 5.6 | — | — | — |
| near-miss-b | bayes | 11 | 1 | 0 | 7.0 | 165.0 | 5.6 | 95.5% | — | — |
| near-miss-b | default | 11 | 1 | 0 | 10.1 | 247.4 | 5.6 | 96.9% | — | — |
| near-miss-c | baseline | 10 | 1 | 0 | 6.9 | 8.8 | 5.6 | — | — | — |
| near-miss-c | bayes | 10 | 1 | 0 | 8.1 | 10.2 | 5.6 | 100.0% | — | — |
| near-miss-c | default | 10 | 1 | 0 | 10.1 | 11.1 | 5.6 | 100.0% | — | — |
| pfss-dhcs | baseline | 27 | 1 | 0 | 7.1 | 9.6 | 5.5 | — | — | — |
| pfss-dhcs | default | 27 | 1 | 0 | 9.9 | 137.2 | 5.5 | 100.0% | — | — |
| pfss-shadow | baseline | 27 | 1 | 0 | 5.9 | 23.2 | 5.5 | — | — | — |
| pfss-shadow | default | 27 | 1 | 0 | 42.8 | 95.1 | 5.5 | 100.0% | pfss ×26 | — |
| pfss-shadow-2 | baseline | 27 | 1 | 0 | 3.8 | 9.3 | 5.5 | — | — | — |
| pfss-shadow-2 | default | 27 | 1 | 0 | 46.9 | 105.6 | 5.5 | 100.0% | pfss ×26 | — |
| pfss-shadow-3 | baseline | 27 | 1 | 0 | 6.6 | 9.9 | 5.5 | — | — | — |
| pfss-shadow-3 | default | 27 | 1 | 0 | 9.1 | 81.6 | 5.5 | 100.0% | — | — |
| submitted | baseline | 3 | 0 | 0 | 8.2 | 8.8 | 15.3 | — | — | — |
| submitted | bayes | 3 | 0 | 0 | 8.0 | 10.6 | 15.3 | 100.0% | — | — |
| submitted | default | 3 | 0 | 0 | 7.9 | 10.7 | 15.3 | 100.0% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 12 | 1 | 0 | 7.4 | 8.1 | 5.6 | — | — | — |
| bayes-option | bayes | 12 | 1 | 0 | 7.1 | 22.1 | 5.6 | 100.0% | — | — |
| bayes-option | default | 12 | 1 | 0 | 10.1 | 11.8 | 5.6 | 100.0% | — | — |
| benchlab | dryrun | 1 | 0 | 0 | 43.0 | 43.0 | 5.6 | 100.0% | — | — |
| benchlab | live | 3 | 0 | 0 | 42.0 | 50.0 | 5.2 | 100.0% | — | — |
| freeze | baseline | 11 | 1 | 0 | 7.8 | 25.9 | 5.7 | — | — | — |
| freeze | bayes | 11 | 1 | 0 | 7.1 | 10.6 | 5.7 | 100.0% | — | — |
| freeze | default | 11 | 1 | 0 | 9.6 | 11.5 | 5.7 | 100.0% | — | — |
| linear-v2 | baseline | 99 | 1 | 0 | 6.7 | 33.3 | 5.4 | — | — | — |
| linear-v2 | default | 99 | 1 | 0 | 6.9 | 101.8 | 5.4 | 100.0% | — | — |
| near-miss-a | baseline | 11 | 1 | 0 | 3.9 | 9.3 | 5.6 | — | — | — |
| near-miss-a | bayes | 11 | 1 | 0 | 7.2 | 10.7 | 5.6 | 100.0% | — | — |
| near-miss-a | default | 11 | 1 | 0 | 9.2 | 11.7 | 5.6 | 100.0% | — | — |
| near-miss-b | baseline | 11 | 1 | 0 | 8.0 | 246.7 | 5.6 | — | — | — |
| near-miss-b | bayes | 11 | 1 | 0 | 7.0 | 165.0 | 5.6 | 95.5% | — | — |
| near-miss-b | default | 11 | 1 | 0 | 10.1 | 247.4 | 5.6 | 96.9% | — | — |
| near-miss-c | baseline | 10 | 1 | 0 | 6.9 | 8.8 | 5.6 | — | — | — |
| near-miss-c | bayes | 10 | 1 | 0 | 8.1 | 10.2 | 5.6 | 100.0% | — | — |
| near-miss-c | default | 10 | 1 | 0 | 10.1 | 11.1 | 5.6 | 100.0% | — | — |
| pfss-dhcs | baseline | 50 | 1 | 0 | 6.8 | 22.4 | 5.4 | — | — | — |
| pfss-dhcs | default | 50 | 1 | 0 | 10.4 | 137.2 | 5.4 | 100.0% | — | — |
| pfss-shadow | baseline | 52 | 1 | 0 | 6.9 | 23.2 | 5.4 | — | — | — |
| pfss-shadow | default | 52 | 1 | 0 | 42.8 | 95.1 | 5.4 | 100.0% | pfss ×51 | — |
| pfss-shadow-2 | baseline | 52 | 1 | 0 | 6.1 | 9.3 | 5.4 | — | — | — |
| pfss-shadow-2 | default | 52 | 1 | 0 | 47.0 | 105.6 | 5.4 | 100.0% | pfss ×51 | — |
| pfss-shadow-3 | baseline | 52 | 1 | 0 | 5.9 | 9.9 | 5.4 | — | — | — |
| pfss-shadow-3 | default | 52 | 1 | 0 | 9.5 | 81.6 | 5.4 | 100.0% | — | — |
| submitted | baseline | 3 | 0 | 0 | 8.2 | 8.8 | 15.3 | — | — | — |
| submitted | bayes | 3 | 0 | 0 | 8.0 | 10.6 | 15.3 | 100.0% | — | — |
| submitted | default | 3 | 0 | 0 | 7.9 | 10.7 | 15.3 | 100.0% | — | — |

## Schedule

Last cycle: 2026-11-20 06:00 UTC
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

## Other events, last 7 days (counts; details in events/)

source not ok: 52, no forecast in log: 19, candidate added: 6, vs staging: 3, eligibility: 1

