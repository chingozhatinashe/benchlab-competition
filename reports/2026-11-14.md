# Staging report 2026-11-14

Generated 2026-11-14 10:52 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-11-14 08:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 63 | 0 | 115.2 | 0.761 | 90.0 | 147.8 | 8.8 | 145.6 | 151.4 | 144.6 |
| linear-v2 | default | 63 | 0 | 97.3 | 0.643 | 78.3 | 121.6 | 14.1 | 130.4 | 151.4 | 144.6 |
| pfss-dhcs | baseline | 14 | 0 | 66.3 | 0.614 | 65.7 | 71.4 | -0.5 | 65.9 | 107.9 | 112.5 |
| pfss-dhcs | default | 14 | 0 | 73.4 | 0.680 | 75.1 | 74.4 | 65.6 | 94.5 | 107.9 | 112.5 |
| pfss-shadow | baseline | 16 | 0 | 93.3 | 0.781 | 84.9 | 106.5 | -21.3 | 75.4 | 119.4 | 126.6 |
| pfss-shadow | default | 16 | 0 | 83.7 | 0.701 | 86.2 | 83.2 | 62.1 | 112.3 | 119.4 | 126.6 |
| pfss-shadow-2 | baseline | 16 | 0 | 93.3 | 0.781 | 84.9 | 106.5 | -21.3 | 75.4 | 119.4 | 126.6 |
| pfss-shadow-2 | default | 16 | 0 | 83.7 | 0.701 | 86.2 | 83.2 | 62.1 | 112.3 | 119.4 | 126.6 |
| pfss-shadow-3 | baseline | 16 | 0 | 93.3 | 0.781 | 84.9 | 106.5 | -21.3 | 75.4 | 119.4 | 126.6 |
| pfss-shadow-3 | default | 16 | 0 | 83.7 | 0.701 | 86.2 | 83.2 | 62.1 | 112.3 | 119.4 | 126.6 |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 14 | 0 | 66.3 | 0.614 | 65.7 | 71.4 | -0.5 | 65.9 | 107.9 | 112.5 |
| linear-v2 | default | 14 | 0 | 79.9 | 0.740 | 83.4 | 79.2 | 73.6 | 113.8 | 107.9 | 112.5 |
| pfss-dhcs | baseline | 14 | 0 | 66.3 | 0.614 | 65.7 | 71.4 | -0.5 | 65.9 | 107.9 | 112.5 |
| pfss-dhcs | default | 14 | 0 | 73.4 | 0.680 | 75.1 | 74.4 | 65.6 | 94.5 | 107.9 | 112.5 |
| pfss-shadow | baseline | 14 | 0 | 66.3 | 0.614 | 65.7 | 71.4 | -0.5 | 65.9 | 107.9 | 112.5 |
| pfss-shadow | default | 14 | 0 | 79.9 | 0.740 | 83.4 | 79.2 | 73.6 | 113.8 | 107.9 | 112.5 |
| pfss-shadow-2 | baseline | 14 | 0 | 66.3 | 0.614 | 65.7 | 71.4 | -0.5 | 65.9 | 107.9 | 112.5 |
| pfss-shadow-2 | default | 14 | 0 | 79.9 | 0.740 | 83.4 | 79.2 | 73.7 | 113.8 | 107.9 | 112.5 |
| pfss-shadow-3 | baseline | 14 | 0 | 66.3 | 0.614 | 65.7 | 71.4 | -0.5 | 65.9 | 107.9 | 112.5 |
| pfss-shadow-3 | default | 14 | 0 | 79.9 | 0.740 | 83.4 | 79.2 | 73.6 | 113.8 | 107.9 | 112.5 |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 28 | 0 | 0 | 6.65 | 26.6 | 5.391666666666667 | — | — | — |
| linear-v2 | default | 28 | 0 | 0 | 7.1 | 23.1 | 5.391666666666667 | 100.0% | — | — |
| pfss-dhcs | baseline | 26 | 0 | 0 | 6.5 | 22.4 | 5.391666666666667 | — | — | — |
| pfss-dhcs | default | 26 | 0 | 0 | 10.7 | 13.7 | 5.391666666666667 | 100.0% | — | — |
| pfss-shadow | baseline | 28 | 0 | 0 | 7.0 | 9.8 | 5.383333333333334 | — | — | — |
| pfss-shadow | default | 28 | 0 | 0 | 42.849999999999994 | 62.3 | 5.383333333333334 | 100.0% | pfss ×28 | — |
| pfss-shadow-2 | baseline | 28 | 0 | 0 | 6.7 | 9.3 | 5.4 | — | — | — |
| pfss-shadow-2 | default | 28 | 0 | 0 | 47.0 | 47.7 | 5.4 | 100.0% | pfss ×28 | — |
| pfss-shadow-3 | baseline | 28 | 0 | 0 | 5.1 | 9.6 | 5.375 | — | — | — |
| pfss-shadow-3 | default | 28 | 0 | 0 | 9.9 | 13.4 | 5.375 | 100.0% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 75 | 0 | 0 | 6.6 | 33.3 | 5.416666666666667 | — | — | — |
| linear-v2 | default | 75 | 0 | 0 | 6.9 | 30.0 | 5.416666666666667 | 100.0% | — | — |
| pfss-dhcs | baseline | 26 | 0 | 0 | 6.5 | 22.4 | 5.391666666666667 | — | — | — |
| pfss-dhcs | default | 26 | 0 | 0 | 10.7 | 13.7 | 5.391666666666667 | 100.0% | — | — |
| pfss-shadow | baseline | 28 | 0 | 0 | 7.0 | 9.8 | 5.383333333333334 | — | — | — |
| pfss-shadow | default | 28 | 0 | 0 | 42.849999999999994 | 62.3 | 5.383333333333334 | 100.0% | pfss ×28 | — |
| pfss-shadow-2 | baseline | 28 | 0 | 0 | 6.7 | 9.3 | 5.4 | — | — | — |
| pfss-shadow-2 | default | 28 | 0 | 0 | 47.0 | 47.7 | 5.4 | 100.0% | pfss ×28 | — |
| pfss-shadow-3 | baseline | 28 | 0 | 0 | 5.1 | 9.6 | 5.375 | — | — | — |
| pfss-shadow-3 | default | 28 | 0 | 0 | 9.9 | 13.4 | 5.375 | 100.0% | — | — |

## Schedule

Last cycle: 2026-11-14 06:00 UTC
Missed cycles: 2 (10-26 12Z, 10-27 06Z)

## Alerts, last 7 days

None.

## Other events, last 7 days (counts; details in events/)

source not ok: 56, candidate added: 4

