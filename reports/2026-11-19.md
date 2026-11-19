# Staging report 2026-11-19

Generated 2026-11-19 12:04 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-11-19 10:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 83 | 0 | 112.9 | 0.794 | 88.5 | 144.5 | 24.3 | 143.6 | 142.2 | 136.0 |
| linear-v2 | default | 83 | 0 | 94.3 | 0.663 | 75.4 | 118.5 | 24.0 | 127.2 | 142.2 | 136.0 |
| pfss-dhcs | baseline | 34 | 0 | 68.1 | 0.777 | 77.2 | 66.4 | 34.2 | 73.0 | 87.6 | 87.8 |
| pfss-dhcs | default | 34 | 0 | 63.1 | 0.721 | 67.2 | 66.1 | 55.6 | 81.0 | 87.6 | 87.8 |
| pfss-shadow | baseline | 36 | 0 | 85.7 | 0.892 | 86.2 | 93.7 | 20.2 | 71.6 | 96.1 | 99.0 |
| pfss-shadow | default | 36 | 0 | 71.9 | 0.748 | 76.1 | 74.6 | 58.4 | 87.5 | 96.1 | 99.0 |
| pfss-shadow-2 | baseline | 36 | 0 | 85.7 | 0.892 | 86.2 | 93.7 | 20.2 | 71.6 | 96.1 | 99.0 |
| pfss-shadow-2 | default | 36 | 0 | 71.9 | 0.748 | 76.2 | 74.6 | 58.5 | 87.6 | 96.1 | 99.0 |
| pfss-shadow-3 | baseline | 36 | 0 | 85.7 | 0.892 | 86.2 | 93.7 | 20.2 | 71.6 | 96.1 | 99.0 |
| pfss-shadow-3 | default | 36 | 0 | 71.9 | 0.748 | 76.1 | 74.6 | 58.4 | 87.5 | 96.1 | 99.0 |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 34 | 0 | 68.1 | 0.777 | 77.2 | 66.4 | 34.2 | 73.0 | 87.6 | 87.8 |
| linear-v2 | default | 34 | 0 | 69.1 | 0.788 | 74.5 | 71.1 | 63.5 | 95.9 | 87.6 | 87.8 |
| pfss-dhcs | baseline | 34 | 0 | 68.1 | 0.777 | 77.2 | 66.4 | 34.2 | 73.0 | 87.6 | 87.8 |
| pfss-dhcs | default | 34 | 0 | 63.1 | 0.721 | 67.2 | 66.1 | 55.6 | 81.0 | 87.6 | 87.8 |
| pfss-shadow | baseline | 34 | 0 | 68.1 | 0.777 | 77.2 | 66.4 | 34.2 | 73.0 | 87.6 | 87.8 |
| pfss-shadow | default | 34 | 0 | 69.1 | 0.788 | 74.5 | 71.1 | 63.5 | 95.9 | 87.6 | 87.8 |
| pfss-shadow-2 | baseline | 34 | 0 | 68.1 | 0.777 | 77.2 | 66.4 | 34.2 | 73.0 | 87.6 | 87.8 |
| pfss-shadow-2 | default | 34 | 0 | 69.1 | 0.789 | 74.6 | 71.1 | 63.5 | 95.9 | 87.6 | 87.8 |
| pfss-shadow-3 | baseline | 34 | 0 | 68.1 | 0.777 | 77.2 | 66.4 | 34.2 | 73.0 | 87.6 | 87.8 |
| pfss-shadow-3 | default | 34 | 0 | 69.1 | 0.788 | 74.5 | 71.1 | 63.5 | 95.9 | 87.6 | 87.8 |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 8 | 0 | 0 | 6.7 | 8.1 | 5.6 | — | — | — |
| bayes-option | bayes | 8 | 0 | 0 | 7.3 | 22.1 | 5.6 | 100.0% | — | — |
| bayes-option | default | 8 | 0 | 0 | 10.2 | 11.8 | 5.6 | 100.0% | — | — |
| freeze | baseline | 7 | 0 | 0 | 8.0 | 25.9 | 5.7 | — | — | — |
| freeze | bayes | 7 | 0 | 0 | 6.9 | 10.6 | 5.7 | 100.0% | — | — |
| freeze | default | 7 | 0 | 0 | 9.8 | 11.5 | 5.7 | 100.0% | — | — |
| linear-v2 | baseline | 27 | 0 | 0 | 7.6 | 26.6 | 5.5 | — | — | — |
| linear-v2 | default | 27 | 0 | 0 | 7.1 | 101.8 | 5.5 | 100.0% | — | — |
| near-miss-a | baseline | 7 | 0 | 0 | 3.5 | 8.0 | 5.6 | — | — | — |
| near-miss-a | bayes | 7 | 0 | 0 | 7.2 | 10.4 | 5.6 | 100.0% | — | — |
| near-miss-a | default | 7 | 0 | 0 | 8.5 | 11.7 | 5.6 | 100.0% | — | — |
| near-miss-b | baseline | 7 | 0 | 0 | 7.9 | 9.7 | 5.6 | — | — | — |
| near-miss-b | bayes | 7 | 0 | 0 | 6.8 | 26.6 | 5.6 | 100.0% | — | — |
| near-miss-b | default | 7 | 0 | 0 | 9.7 | 12.2 | 5.6 | 100.0% | — | — |
| near-miss-c | baseline | 6 | 0 | 0 | 5.2 | 8.1 | 5.6 | — | — | — |
| near-miss-c | bayes | 6 | 0 | 0 | 7.5 | 9.8 | 5.6 | 100.0% | — | — |
| near-miss-c | default | 6 | 0 | 0 | 9.1 | 11.1 | 5.6 | 100.0% | — | — |
| pfss-dhcs | baseline | 27 | 0 | 0 | 7.0 | 9.6 | 5.5 | — | — | — |
| pfss-dhcs | default | 27 | 0 | 0 | 10.0 | 137.2 | 5.5 | 100.0% | — | — |
| pfss-shadow | baseline | 27 | 0 | 0 | 6.9 | 23.2 | 5.4 | — | — | — |
| pfss-shadow | default | 27 | 0 | 0 | 42.8 | 95.1 | 5.4 | 100.0% | pfss ×27 | — |
| pfss-shadow-2 | baseline | 27 | 0 | 0 | 5.5 | 9.3 | 5.5 | — | — | — |
| pfss-shadow-2 | default | 27 | 0 | 0 | 46.9 | 105.6 | 5.5 | 100.0% | pfss ×27 | — |
| pfss-shadow-3 | baseline | 27 | 0 | 0 | 7.0 | 9.9 | 5.5 | — | — | — |
| pfss-shadow-3 | default | 27 | 0 | 0 | 9.5 | 81.6 | 5.5 | 100.0% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 8 | 0 | 0 | 6.7 | 8.1 | 5.6 | — | — | — |
| bayes-option | bayes | 8 | 0 | 0 | 7.3 | 22.1 | 5.6 | 100.0% | — | — |
| bayes-option | default | 8 | 0 | 0 | 10.2 | 11.8 | 5.6 | 100.0% | — | — |
| freeze | baseline | 7 | 0 | 0 | 8.0 | 25.9 | 5.7 | — | — | — |
| freeze | bayes | 7 | 0 | 0 | 6.9 | 10.6 | 5.7 | 100.0% | — | — |
| freeze | default | 7 | 0 | 0 | 9.8 | 11.5 | 5.7 | 100.0% | — | — |
| linear-v2 | baseline | 95 | 0 | 0 | 6.7 | 33.3 | 5.4 | — | — | — |
| linear-v2 | default | 95 | 0 | 0 | 6.9 | 101.8 | 5.4 | 100.0% | — | — |
| near-miss-a | baseline | 7 | 0 | 0 | 3.5 | 8.0 | 5.6 | — | — | — |
| near-miss-a | bayes | 7 | 0 | 0 | 7.2 | 10.4 | 5.6 | 100.0% | — | — |
| near-miss-a | default | 7 | 0 | 0 | 8.5 | 11.7 | 5.6 | 100.0% | — | — |
| near-miss-b | baseline | 7 | 0 | 0 | 7.9 | 9.7 | 5.6 | — | — | — |
| near-miss-b | bayes | 7 | 0 | 0 | 6.8 | 26.6 | 5.6 | 100.0% | — | — |
| near-miss-b | default | 7 | 0 | 0 | 9.7 | 12.2 | 5.6 | 100.0% | — | — |
| near-miss-c | baseline | 6 | 0 | 0 | 5.2 | 8.1 | 5.6 | — | — | — |
| near-miss-c | bayes | 6 | 0 | 0 | 7.5 | 9.8 | 5.6 | 100.0% | — | — |
| near-miss-c | default | 6 | 0 | 0 | 9.1 | 11.1 | 5.6 | 100.0% | — | — |
| pfss-dhcs | baseline | 46 | 0 | 0 | 6.8 | 22.4 | 5.4 | — | — | — |
| pfss-dhcs | default | 46 | 0 | 0 | 10.4 | 137.2 | 5.4 | 100.0% | — | — |
| pfss-shadow | baseline | 48 | 0 | 0 | 7.0 | 23.2 | 5.4 | — | — | — |
| pfss-shadow | default | 48 | 0 | 0 | 42.8 | 95.1 | 5.4 | 100.0% | pfss ×48 | — |
| pfss-shadow-2 | baseline | 48 | 0 | 0 | 6.4 | 9.3 | 5.4 | — | — | — |
| pfss-shadow-2 | default | 48 | 0 | 0 | 47.0 | 105.6 | 5.4 | 100.0% | pfss ×48 | — |
| pfss-shadow-3 | baseline | 48 | 0 | 0 | 6.0 | 9.9 | 5.4 | — | — | — |
| pfss-shadow-3 | default | 48 | 0 | 0 | 9.5 | 81.6 | 5.4 | 100.0% | — | — |

## Schedule

Last cycle: 2026-11-19 06:00 UTC
Missed cycles: 2 (10-26 12Z, 10-27 06Z)

## Alerts, last 7 days

None.

## Other events, last 7 days (counts; details in events/)

source not ok: 54, candidate added: 5

