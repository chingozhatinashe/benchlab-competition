# Staging report 2026-11-18

Generated 2026-11-18 12:11 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-11-18 11:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 79 | 0 | 113.2 | 0.787 | 88.6 | 145.0 | 21.9 | 144.3 | 143.7 | 137.2 |
| linear-v2 | default | 79 | 0 | 95.0 | 0.661 | 76.4 | 118.9 | 23.4 | 128.5 | 143.7 | 137.2 |
| pfss-dhcs | baseline | 30 | 0 | 67.6 | 0.751 | 76.0 | 66.9 | 30.9 | 73.8 | 90.1 | 89.3 |
| pfss-dhcs | default | 30 | 0 | 64.4 | 0.715 | 69.1 | 66.8 | 58.5 | 83.8 | 90.1 | 89.3 |
| pfss-shadow | baseline | 32 | 0 | 85.8 | 0.868 | 86.0 | 94.3 | 16.2 | 72.2 | 98.8 | 100.9 |
| pfss-shadow | default | 32 | 0 | 73.5 | 0.743 | 78.3 | 75.5 | 60.7 | 90.5 | 98.8 | 100.9 |
| pfss-shadow-2 | baseline | 32 | 0 | 85.8 | 0.868 | 86.0 | 94.3 | 16.2 | 72.2 | 98.8 | 100.9 |
| pfss-shadow-2 | default | 32 | 0 | 73.5 | 0.744 | 78.3 | 75.5 | 60.7 | 90.5 | 98.8 | 100.9 |
| pfss-shadow-3 | baseline | 32 | 0 | 85.8 | 0.868 | 86.0 | 94.3 | 16.2 | 72.2 | 98.8 | 100.9 |
| pfss-shadow-3 | default | 32 | 0 | 73.5 | 0.743 | 78.3 | 75.5 | 60.7 | 90.5 | 98.8 | 100.9 |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 30 | 0 | 67.6 | 0.751 | 76.0 | 66.9 | 30.9 | 73.8 | 90.1 | 89.3 |
| linear-v2 | default | 30 | 0 | 70.5 | 0.783 | 76.6 | 71.9 | 66.3 | 99.3 | 90.1 | 89.3 |
| pfss-dhcs | baseline | 30 | 0 | 67.6 | 0.751 | 76.0 | 66.9 | 30.9 | 73.8 | 90.1 | 89.3 |
| pfss-dhcs | default | 30 | 0 | 64.4 | 0.715 | 69.1 | 66.8 | 58.5 | 83.8 | 90.1 | 89.3 |
| pfss-shadow | baseline | 30 | 0 | 67.6 | 0.751 | 76.0 | 66.9 | 30.9 | 73.8 | 90.1 | 89.3 |
| pfss-shadow | default | 30 | 0 | 70.5 | 0.783 | 76.6 | 71.9 | 66.3 | 99.3 | 90.1 | 89.3 |
| pfss-shadow-2 | baseline | 30 | 0 | 67.6 | 0.751 | 76.0 | 66.9 | 30.9 | 73.8 | 90.1 | 89.3 |
| pfss-shadow-2 | default | 30 | 0 | 70.5 | 0.783 | 76.6 | 71.9 | 66.4 | 99.3 | 90.1 | 89.3 |
| pfss-shadow-3 | baseline | 30 | 0 | 67.6 | 0.751 | 76.0 | 66.9 | 30.9 | 73.8 | 90.1 | 89.3 |
| pfss-shadow-3 | default | 30 | 0 | 70.5 | 0.783 | 76.6 | 71.9 | 66.3 | 99.3 | 90.1 | 89.3 |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 5 | 0 | 0 | 7.6 | 8.1 | 5.6 | — | — | — |
| bayes-option | bayes | 5 | 0 | 0 | 7.7 | 22.1 | 5.6 | 100.0% | — | — |
| bayes-option | default | 5 | 0 | 0 | 10.9 | 11.8 | 5.6 | 100.0% | — | — |
| freeze | baseline | 4 | 0 | 0 | 6.2 | 25.9 | 5.7 | — | — | — |
| freeze | bayes | 4 | 0 | 0 | 7.7 | 10.6 | 5.7 | 100.0% | — | — |
| freeze | default | 4 | 0 | 0 | 9.6 | 11.5 | 5.7 | 100.0% | — | — |
| linear-v2 | baseline | 28 | 0 | 0 | 7.8 | 26.6 | 5.4 | — | — | — |
| linear-v2 | default | 28 | 0 | 0 | 7.5 | 101.8 | 5.4 | 100.0% | — | — |
| near-miss-a | baseline | 4 | 0 | 0 | 3.6 | 7.0 | 5.6 | — | — | — |
| near-miss-a | bayes | 4 | 0 | 0 | 8.7 | 10.4 | 5.6 | 100.0% | — | — |
| near-miss-a | default | 4 | 0 | 0 | 9.9 | 11.7 | 5.6 | 100.0% | — | — |
| near-miss-b | baseline | 4 | 0 | 0 | 6.1 | 8.3 | 5.7 | — | — | — |
| near-miss-b | bayes | 4 | 0 | 0 | 6.8 | 10.3 | 5.7 | 100.0% | — | — |
| near-miss-b | default | 4 | 0 | 0 | 9.7 | 12.2 | 5.7 | 100.0% | — | — |
| near-miss-c | baseline | 3 | 0 | 0 | 6.9 | 7.7 | 5.5 | — | — | — |
| near-miss-c | bayes | 3 | 0 | 0 | 9.4 | 9.8 | 5.5 | 100.0% | — | — |
| near-miss-c | default | 3 | 0 | 0 | 10.9 | 11.1 | 5.5 | 100.0% | — | — |
| pfss-dhcs | baseline | 28 | 0 | 0 | 6.9 | 9.6 | 5.4 | — | — | — |
| pfss-dhcs | default | 28 | 0 | 0 | 10.0 | 137.2 | 5.4 | 100.0% | — | — |
| pfss-shadow | baseline | 28 | 0 | 0 | 6.4 | 23.2 | 5.4 | — | — | — |
| pfss-shadow | default | 28 | 0 | 0 | 42.8 | 95.1 | 5.4 | 100.0% | pfss ×28 | — |
| pfss-shadow-2 | baseline | 28 | 0 | 0 | 5.5 | 9.3 | 5.4 | — | — | — |
| pfss-shadow-2 | default | 28 | 0 | 0 | 47.0 | 105.6 | 5.4 | 100.0% | pfss ×28 | — |
| pfss-shadow-3 | baseline | 28 | 0 | 0 | 6.3 | 9.9 | 5.4 | — | — | — |
| pfss-shadow-3 | default | 28 | 0 | 0 | 9.4 | 81.6 | 5.4 | 100.0% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 5 | 0 | 0 | 7.6 | 8.1 | 5.6 | — | — | — |
| bayes-option | bayes | 5 | 0 | 0 | 7.7 | 22.1 | 5.6 | 100.0% | — | — |
| bayes-option | default | 5 | 0 | 0 | 10.9 | 11.8 | 5.6 | 100.0% | — | — |
| freeze | baseline | 4 | 0 | 0 | 6.2 | 25.9 | 5.7 | — | — | — |
| freeze | bayes | 4 | 0 | 0 | 7.7 | 10.6 | 5.7 | 100.0% | — | — |
| freeze | default | 4 | 0 | 0 | 9.6 | 11.5 | 5.7 | 100.0% | — | — |
| linear-v2 | baseline | 92 | 0 | 0 | 6.8 | 33.3 | 5.4 | — | — | — |
| linear-v2 | default | 92 | 0 | 0 | 7.0 | 101.8 | 5.4 | 100.0% | — | — |
| near-miss-a | baseline | 4 | 0 | 0 | 3.6 | 7.0 | 5.6 | — | — | — |
| near-miss-a | bayes | 4 | 0 | 0 | 8.7 | 10.4 | 5.6 | 100.0% | — | — |
| near-miss-a | default | 4 | 0 | 0 | 9.9 | 11.7 | 5.6 | 100.0% | — | — |
| near-miss-b | baseline | 4 | 0 | 0 | 6.1 | 8.3 | 5.7 | — | — | — |
| near-miss-b | bayes | 4 | 0 | 0 | 6.8 | 10.3 | 5.7 | 100.0% | — | — |
| near-miss-b | default | 4 | 0 | 0 | 9.7 | 12.2 | 5.7 | 100.0% | — | — |
| near-miss-c | baseline | 3 | 0 | 0 | 6.9 | 7.7 | 5.5 | — | — | — |
| near-miss-c | bayes | 3 | 0 | 0 | 9.4 | 9.8 | 5.5 | 100.0% | — | — |
| near-miss-c | default | 3 | 0 | 0 | 10.9 | 11.1 | 5.5 | 100.0% | — | — |
| pfss-dhcs | baseline | 43 | 0 | 0 | 6.8 | 22.4 | 5.4 | — | — | — |
| pfss-dhcs | default | 43 | 0 | 0 | 10.4 | 137.2 | 5.4 | 100.0% | — | — |
| pfss-shadow | baseline | 45 | 0 | 0 | 6.8 | 23.2 | 5.4 | — | — | — |
| pfss-shadow | default | 45 | 0 | 0 | 42.8 | 95.1 | 5.4 | 100.0% | pfss ×45 | — |
| pfss-shadow-2 | baseline | 45 | 0 | 0 | 6.3 | 9.3 | 5.4 | — | — | — |
| pfss-shadow-2 | default | 45 | 0 | 0 | 47.0 | 105.6 | 5.4 | 100.0% | pfss ×45 | — |
| pfss-shadow-3 | baseline | 45 | 0 | 0 | 5.6 | 9.9 | 5.4 | — | — | — |
| pfss-shadow-3 | default | 45 | 0 | 0 | 9.3 | 81.6 | 5.4 | 100.0% | — | — |

## Schedule

Last cycle: 2026-11-18 12:00 UTC
Missed cycles: 2 (10-26 12Z, 10-27 06Z)

## Alerts, last 7 days

None.

## Other events, last 7 days (counts; details in events/)

source not ok: 56, candidate added: 5

