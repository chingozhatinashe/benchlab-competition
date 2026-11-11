# Staging report 2026-11-11

Generated 2026-11-11 11:25 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-11-11 08:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 51 | 0 | 125.6 | 0.804 | 96.1 | 164.6 | 6.6 | 152.9 | 156.1 | 151.3 |
| linear-v2 | default | 51 | 0 | 99.6 | 0.638 | 77.1 | 129.9 | -1.3 | 130.3 | 156.1 | 151.3 |
| pfss-dhcs | baseline | 2 | 0 | 99.6 | 1.943 | 94.0 | 103.9 | -89.1 | 74.8 | 51.2 | 134.7 |
| pfss-dhcs | default | 2 | 0 | 70.6 | 1.379 | 77.2 | 68.8 | 51.4 | 92.8 | 51.2 | 134.7 |
| pfss-shadow | baseline | 4 | 0 | 133.0 | 1.288 | 133.1 | 137.9 | -126.4 | 90.2 | 103.2 | 162.6 |
| pfss-shadow | default | 4 | 0 | 88.7 | 0.859 | 95.8 | 81.4 | 18.5 | 105.9 | 103.2 | 162.6 |
| pfss-shadow-2 | baseline | 4 | 0 | 133.0 | 1.288 | 133.1 | 137.9 | -126.4 | 90.2 | 103.2 | 162.6 |
| pfss-shadow-2 | default | 4 | 0 | 88.7 | 0.859 | 95.8 | 81.4 | 18.5 | 105.9 | 103.2 | 162.6 |
| pfss-shadow-3 | baseline | 4 | 0 | 133.0 | 1.288 | 133.1 | 137.9 | -126.4 | 90.2 | 103.2 | 162.6 |
| pfss-shadow-3 | default | 4 | 0 | 88.7 | 0.859 | 95.8 | 81.4 | 18.5 | 105.9 | 103.2 | 162.6 |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 2 | 0 | 99.5 | 1.943 | 94.0 | 103.9 | -89.1 | 74.8 | 51.2 | 134.7 |
| linear-v2 | default | 2 | 0 | 75.4 | 1.471 | 81.6 | 73.8 | 53.4 | 105.3 | 51.2 | 134.7 |
| pfss-dhcs | baseline | 2 | 0 | 99.6 | 1.943 | 94.0 | 103.9 | -89.1 | 74.8 | 51.2 | 134.7 |
| pfss-dhcs | default | 2 | 0 | 70.6 | 1.379 | 77.2 | 68.8 | 51.4 | 92.8 | 51.2 | 134.7 |
| pfss-shadow | baseline | 2 | 0 | 99.6 | 1.943 | 94.0 | 103.9 | -89.1 | 74.8 | 51.2 | 134.7 |
| pfss-shadow | default | 2 | 0 | 75.4 | 1.471 | 81.6 | 73.8 | 53.4 | 105.3 | 51.2 | 134.7 |
| pfss-shadow-2 | baseline | 2 | 0 | 99.6 | 1.943 | 94.0 | 103.9 | -89.1 | 74.8 | 51.2 | 134.7 |
| pfss-shadow-2 | default | 2 | 0 | 75.4 | 1.471 | 81.6 | 73.8 | 53.4 | 105.3 | 51.2 | 134.7 |
| pfss-shadow-3 | baseline | 2 | 0 | 99.5 | 1.943 | 94.0 | 103.9 | -89.1 | 74.8 | 51.2 | 134.7 |
| pfss-shadow-3 | default | 2 | 0 | 75.4 | 1.471 | 81.6 | 73.8 | 53.4 | 105.3 | 51.2 | 134.7 |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 28 | 0 | 0 | 6.95 | 24.1 | 5.408333333333333 | — | — | — |
| linear-v2 | default | 28 | 0 | 0 | 7.1 | 30.0 | 5.408333333333333 | 100.0% | — | — |
| pfss-dhcs | baseline | 14 | 0 | 0 | 6.85 | 22.4 | 5.375 | — | — | — |
| pfss-dhcs | default | 14 | 0 | 0 | 10.9 | 13.7 | 5.375 | 100.0% | — | — |
| pfss-shadow | baseline | 16 | 0 | 0 | 7.1 | 9.8 | 5.375 | — | — | — |
| pfss-shadow | default | 16 | 0 | 0 | 42.849999999999994 | 62.3 | 5.375 | 100.0% | pfss ×16 | — |
| pfss-shadow-2 | baseline | 16 | 0 | 0 | 6.7 | 9.0 | 5.383333333333334 | — | — | — |
| pfss-shadow-2 | default | 16 | 0 | 0 | 47.0 | 47.7 | 5.383333333333334 | 100.0% | pfss ×16 | — |
| pfss-shadow-3 | baseline | 16 | 0 | 0 | 5.4 | 9.5 | 5.375 | — | — | — |
| pfss-shadow-3 | default | 16 | 0 | 0 | 9.65 | 12.4 | 5.375 | 100.0% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 63 | 0 | 0 | 6.5 | 33.3 | 5.416666666666667 | — | — | — |
| linear-v2 | default | 63 | 0 | 0 | 6.8 | 30.0 | 5.416666666666667 | 100.0% | — | — |
| pfss-dhcs | baseline | 14 | 0 | 0 | 6.85 | 22.4 | 5.375 | — | — | — |
| pfss-dhcs | default | 14 | 0 | 0 | 10.9 | 13.7 | 5.375 | 100.0% | — | — |
| pfss-shadow | baseline | 16 | 0 | 0 | 7.1 | 9.8 | 5.375 | — | — | — |
| pfss-shadow | default | 16 | 0 | 0 | 42.849999999999994 | 62.3 | 5.375 | 100.0% | pfss ×16 | — |
| pfss-shadow-2 | baseline | 16 | 0 | 0 | 6.7 | 9.0 | 5.383333333333334 | — | — | — |
| pfss-shadow-2 | default | 16 | 0 | 0 | 47.0 | 47.7 | 5.383333333333334 | 100.0% | pfss ×16 | — |
| pfss-shadow-3 | baseline | 16 | 0 | 0 | 5.4 | 9.5 | 5.375 | — | — | — |
| pfss-shadow-3 | default | 16 | 0 | 0 | 9.65 | 12.4 | 5.375 | 100.0% | — | — |

## Schedule

Last cycle: 2026-11-11 06:00 UTC
Missed cycles: 2 (10-26 12Z, 10-27 06Z)

## Alerts, last 7 days

None.

## Other events, last 7 days (counts; details in events/)

source not ok: 32, candidate added: 4

