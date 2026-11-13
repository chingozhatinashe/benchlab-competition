# Staging report 2026-11-13

Generated 2026-11-13 11:28 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-11-13 10:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 59 | 0 | 116.5 | 0.768 | 91.1 | 149.2 | 4.9 | 146.0 | 151.7 | 147.1 |
| linear-v2 | default | 59 | 0 | 97.7 | 0.644 | 78.3 | 122.5 | 9.5 | 129.9 | 151.7 | 147.1 |
| pfss-dhcs | baseline | 10 | 0 | 65.4 | 0.641 | 60.4 | 75.2 | -22.8 | 55.3 | 102.0 | 120.3 |
| pfss-dhcs | default | 10 | 0 | 75.5 | 0.740 | 78.6 | 75.1 | 67.0 | 96.2 | 102.0 | 120.3 |
| pfss-shadow | baseline | 12 | 0 | 96.4 | 0.821 | 85.7 | 111.9 | -46.0 | 68.1 | 117.4 | 136.1 |
| pfss-shadow | default | 12 | 0 | 84.9 | 0.723 | 88.5 | 83.2 | 57.2 | 110.9 | 117.4 | 136.1 |
| pfss-shadow-2 | baseline | 12 | 0 | 96.4 | 0.821 | 85.7 | 111.9 | -46.0 | 68.1 | 117.4 | 136.1 |
| pfss-shadow-2 | default | 12 | 0 | 85.0 | 0.723 | 88.5 | 83.2 | 57.2 | 110.9 | 117.4 | 136.1 |
| pfss-shadow-3 | baseline | 12 | 0 | 96.4 | 0.821 | 85.7 | 111.9 | -46.0 | 68.1 | 117.4 | 136.1 |
| pfss-shadow-3 | default | 12 | 0 | 84.9 | 0.723 | 88.5 | 83.2 | 57.2 | 110.9 | 117.4 | 136.1 |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 10 | 0 | 65.4 | 0.641 | 60.4 | 75.2 | -22.8 | 55.3 | 102.0 | 120.3 |
| linear-v2 | default | 10 | 0 | 80.6 | 0.790 | 84.7 | 79.3 | 72.4 | 112.4 | 102.0 | 120.3 |
| pfss-dhcs | baseline | 10 | 0 | 65.4 | 0.641 | 60.4 | 75.2 | -22.8 | 55.3 | 102.0 | 120.3 |
| pfss-dhcs | default | 10 | 0 | 75.5 | 0.740 | 78.6 | 75.1 | 67.0 | 96.2 | 102.0 | 120.3 |
| pfss-shadow | baseline | 10 | 0 | 65.4 | 0.641 | 60.4 | 75.2 | -22.8 | 55.3 | 102.0 | 120.3 |
| pfss-shadow | default | 10 | 0 | 80.6 | 0.790 | 84.7 | 79.3 | 72.4 | 112.4 | 102.0 | 120.3 |
| pfss-shadow-2 | baseline | 10 | 0 | 65.4 | 0.641 | 60.4 | 75.2 | -22.8 | 55.3 | 102.0 | 120.3 |
| pfss-shadow-2 | default | 10 | 0 | 80.6 | 0.790 | 84.8 | 79.3 | 72.4 | 112.4 | 102.0 | 120.3 |
| pfss-shadow-3 | baseline | 10 | 0 | 65.4 | 0.641 | 60.4 | 75.2 | -22.8 | 55.3 | 102.0 | 120.3 |
| pfss-shadow-3 | default | 10 | 0 | 80.6 | 0.790 | 84.7 | 79.3 | 72.4 | 112.4 | 102.0 | 120.3 |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 28 | 0 | 0 | 6.75 | 26.6 | 5.391666666666667 | — | — | — |
| linear-v2 | default | 28 | 0 | 0 | 7.1 | 23.1 | 5.391666666666667 | 100.0% | — | — |
| pfss-dhcs | baseline | 22 | 0 | 0 | 6.5 | 22.4 | 5.391666666666667 | — | — | — |
| pfss-dhcs | default | 22 | 0 | 0 | 10.75 | 13.7 | 5.391666666666667 | 100.0% | — | — |
| pfss-shadow | baseline | 24 | 0 | 0 | 7.0 | 9.8 | 5.383333333333334 | — | — | — |
| pfss-shadow | default | 24 | 0 | 0 | 42.9 | 62.3 | 5.383333333333334 | 100.0% | pfss ×24 | — |
| pfss-shadow-2 | baseline | 24 | 0 | 0 | 6.7 | 9.3 | 5.383333333333334 | — | — | — |
| pfss-shadow-2 | default | 24 | 0 | 0 | 47.0 | 47.7 | 5.383333333333334 | 100.0% | pfss ×24 | — |
| pfss-shadow-3 | baseline | 24 | 0 | 0 | 5.1 | 9.6 | 5.375 | — | — | — |
| pfss-shadow-3 | default | 24 | 0 | 0 | 9.75 | 13.4 | 5.375 | 100.0% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 71 | 0 | 0 | 6.6 | 33.3 | 5.4 | — | — | — |
| linear-v2 | default | 71 | 0 | 0 | 6.9 | 30.0 | 5.4 | 100.0% | — | — |
| pfss-dhcs | baseline | 22 | 0 | 0 | 6.5 | 22.4 | 5.391666666666667 | — | — | — |
| pfss-dhcs | default | 22 | 0 | 0 | 10.75 | 13.7 | 5.391666666666667 | 100.0% | — | — |
| pfss-shadow | baseline | 24 | 0 | 0 | 7.0 | 9.8 | 5.383333333333334 | — | — | — |
| pfss-shadow | default | 24 | 0 | 0 | 42.9 | 62.3 | 5.383333333333334 | 100.0% | pfss ×24 | — |
| pfss-shadow-2 | baseline | 24 | 0 | 0 | 6.7 | 9.3 | 5.383333333333334 | — | — | — |
| pfss-shadow-2 | default | 24 | 0 | 0 | 47.0 | 47.7 | 5.383333333333334 | 100.0% | pfss ×24 | — |
| pfss-shadow-3 | baseline | 24 | 0 | 0 | 5.1 | 9.6 | 5.375 | — | — | — |
| pfss-shadow-3 | default | 24 | 0 | 0 | 9.75 | 13.4 | 5.375 | 100.0% | — | — |

## Schedule

Last cycle: 2026-11-13 06:00 UTC
Missed cycles: 2 (10-26 12Z, 10-27 06Z)

## Alerts, last 7 days

None.

## Other events, last 7 days (counts; details in events/)

source not ok: 48, candidate added: 4

