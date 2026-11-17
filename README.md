# Staging report 2026-11-17

Generated 2026-11-17 12:24 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-11-17 10:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 75 | 0 | 113.2 | 0.778 | 87.9 | 146.4 | 18.3 | 143.5 | 145.5 | 138.8 |
| linear-v2 | default | 75 | 0 | 94.9 | 0.652 | 75.4 | 120.2 | 20.0 | 127.6 | 145.5 | 138.8 |
| pfss-dhcs | baseline | 26 | 0 | 69.4 | 0.714 | 71.5 | 70.8 | 24.2 | 76.5 | 97.2 | 97.0 |
| pfss-dhcs | default | 26 | 0 | 67.1 | 0.690 | 66.2 | 71.6 | 55.1 | 89.8 | 97.2 | 97.0 |
| pfss-shadow | baseline | 28 | 0 | 90.0 | 0.842 | 83.1 | 101.6 | 8.7 | 82.3 | 106.9 | 109.7 |
| pfss-shadow | default | 28 | 0 | 77.0 | 0.720 | 76.1 | 80.8 | 57.1 | 105.9 | 106.9 | 109.7 |
| pfss-shadow-2 | baseline | 28 | 0 | 90.0 | 0.842 | 83.1 | 101.6 | 8.7 | 82.3 | 106.9 | 109.7 |
| pfss-shadow-2 | default | 28 | 0 | 77.0 | 0.720 | 76.1 | 80.8 | 57.1 | 105.9 | 106.9 | 109.7 |
| pfss-shadow-3 | baseline | 28 | 0 | 90.0 | 0.842 | 83.1 | 101.6 | 8.7 | 82.3 | 106.9 | 109.7 |
| pfss-shadow-3 | default | 28 | 0 | 77.0 | 0.720 | 76.1 | 80.8 | 57.1 | 105.9 | 106.9 | 109.7 |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 26 | 0 | 69.4 | 0.714 | 71.5 | 70.8 | 24.2 | 76.5 | 97.2 | 97.0 |
| linear-v2 | default | 26 | 0 | 73.6 | 0.757 | 73.9 | 76.9 | 63.1 | 106.4 | 97.2 | 97.0 |
| pfss-dhcs | baseline | 26 | 0 | 69.4 | 0.714 | 71.5 | 70.8 | 24.2 | 76.5 | 97.2 | 97.0 |
| pfss-dhcs | default | 26 | 0 | 67.1 | 0.690 | 66.2 | 71.6 | 55.1 | 89.8 | 97.2 | 97.0 |
| pfss-shadow | baseline | 26 | 0 | 69.4 | 0.714 | 71.5 | 70.8 | 24.2 | 76.5 | 97.2 | 97.0 |
| pfss-shadow | default | 26 | 0 | 73.6 | 0.757 | 73.9 | 76.9 | 63.1 | 106.4 | 97.2 | 97.0 |
| pfss-shadow-2 | baseline | 26 | 0 | 69.4 | 0.714 | 71.5 | 70.8 | 24.2 | 76.5 | 97.2 | 97.0 |
| pfss-shadow-2 | default | 26 | 0 | 73.6 | 0.757 | 73.9 | 76.9 | 63.1 | 106.4 | 97.2 | 97.0 |
| pfss-shadow-3 | baseline | 26 | 0 | 69.4 | 0.714 | 71.5 | 70.8 | 24.2 | 76.5 | 97.2 | 97.0 |
| pfss-shadow-3 | default | 26 | 0 | 73.6 | 0.757 | 73.9 | 76.9 | 63.1 | 106.4 | 97.2 | 97.0 |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 1 | 0 | 0 | 7.9 | 7.9 | 5.6 | — | — | — |
| bayes-option | bayes | 1 | 0 | 0 | 7.0 | 7.0 | 5.6 | 100.0% | — | — |
| bayes-option | default | 1 | 0 | 0 | 10.9 | 10.9 | 5.6 | 100.0% | — | — |
| linear-v2 | baseline | 28 | 0 | 0 | 7.3 | 26.6 | 5.4 | — | — | — |
| linear-v2 | default | 28 | 0 | 0 | 7.2 | 101.8 | 5.4 | 100.0% | — | — |
| pfss-dhcs | baseline | 28 | 0 | 0 | 6.8 | 9.6 | 5.4 | — | — | — |
| pfss-dhcs | default | 28 | 0 | 0 | 10.2 | 137.2 | 5.4 | 100.0% | — | — |
| pfss-shadow | baseline | 28 | 0 | 0 | 6.0 | 8.9 | 5.4 | — | — | — |
| pfss-shadow | default | 28 | 0 | 0 | 42.8 | 95.1 | 5.4 | 100.0% | pfss ×28 | — |
| pfss-shadow-2 | baseline | 28 | 0 | 0 | 6.2 | 9.3 | 5.4 | — | — | — |
| pfss-shadow-2 | default | 28 | 0 | 0 | 47.0 | 105.6 | 5.4 | 100.0% | pfss ×28 | — |
| pfss-shadow-3 | baseline | 28 | 0 | 0 | 6.5 | 9.9 | 5.4 | — | — | — |
| pfss-shadow-3 | default | 28 | 0 | 0 | 9.6 | 81.6 | 5.4 | 100.0% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| bayes-option | baseline | 1 | 0 | 0 | 7.9 | 7.9 | 5.6 | — | — | — |
| bayes-option | bayes | 1 | 0 | 0 | 7.0 | 7.0 | 5.6 | 100.0% | — | — |
| bayes-option | default | 1 | 0 | 0 | 10.9 | 10.9 | 5.6 | 100.0% | — | — |
| linear-v2 | baseline | 88 | 0 | 0 | 6.7 | 33.3 | 5.4 | — | — | — |
| linear-v2 | default | 88 | 0 | 0 | 6.9 | 101.8 | 5.4 | 100.0% | — | — |
| pfss-dhcs | baseline | 39 | 0 | 0 | 6.8 | 22.4 | 5.4 | — | — | — |
| pfss-dhcs | default | 39 | 0 | 0 | 10.4 | 137.2 | 5.4 | 100.0% | — | — |
| pfss-shadow | baseline | 41 | 0 | 0 | 6.7 | 9.8 | 5.4 | — | — | — |
| pfss-shadow | default | 41 | 0 | 0 | 42.8 | 95.1 | 5.4 | 100.0% | pfss ×41 | — |
| pfss-shadow-2 | baseline | 41 | 0 | 0 | 6.5 | 9.3 | 5.4 | — | — | — |
| pfss-shadow-2 | default | 41 | 0 | 0 | 47.0 | 105.6 | 5.4 | 100.0% | pfss ×41 | — |
| pfss-shadow-3 | baseline | 41 | 0 | 0 | 5.9 | 9.9 | 5.4 | — | — | — |
| pfss-shadow-3 | default | 41 | 0 | 0 | 9.6 | 81.6 | 5.4 | 100.0% | — | — |

## Schedule

Last cycle: 2026-11-17 12:00 UTC
Missed cycles: 2 (10-26 12Z, 10-27 06Z)

## Alerts, last 7 days

None.

## Other events, last 7 days (counts; details in events/)

source not ok: 56, candidate added: 1

