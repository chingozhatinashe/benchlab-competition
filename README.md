# Staging report 2026-11-15

Generated 2026-11-15 11:26 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-11-15 10:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 67 | 0 | 114.7 | 0.765 | 89.4 | 147.5 | 12.8 | 145.1 | 149.8 | 142.8 |
| linear-v2 | default | 67 | 0 | 96.6 | 0.644 | 77.3 | 121.3 | 17.0 | 129.2 | 149.8 | 142.8 |
| pfss-dhcs | baseline | 18 | 0 | 67.8 | 0.646 | 70.0 | 69.6 | 13.2 | 69.3 | 105.1 | 106.1 |
| pfss-dhcs | default | 18 | 0 | 70.0 | 0.667 | 69.8 | 73.5 | 60.9 | 89.8 | 105.1 | 106.1 |
| pfss-shadow | baseline | 20 | 0 | 91.9 | 0.799 | 85.0 | 103.0 | -5.5 | 78.1 | 115.1 | 119.4 |
| pfss-shadow | default | 20 | 0 | 80.8 | 0.702 | 81.8 | 82.0 | 61.5 | 108.0 | 115.1 | 119.4 |
| pfss-shadow-2 | baseline | 20 | 0 | 91.9 | 0.799 | 85.0 | 103.0 | -5.5 | 78.1 | 115.1 | 119.4 |
| pfss-shadow-2 | default | 20 | 0 | 80.8 | 0.702 | 81.8 | 82.0 | 61.5 | 108.0 | 115.1 | 119.4 |
| pfss-shadow-3 | baseline | 20 | 0 | 91.9 | 0.799 | 85.0 | 103.0 | -5.5 | 78.1 | 115.1 | 119.4 |
| pfss-shadow-3 | default | 20 | 0 | 80.8 | 0.702 | 81.8 | 82.0 | 61.5 | 108.0 | 115.1 | 119.4 |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 18 | 0 | 67.8 | 0.646 | 70.0 | 69.6 | 13.2 | 69.3 | 105.1 | 106.1 |
| linear-v2 | default | 18 | 0 | 77.3 | 0.735 | 79.2 | 78.4 | 70.5 | 108.9 | 105.1 | 106.1 |
| pfss-dhcs | baseline | 18 | 0 | 67.8 | 0.646 | 70.0 | 69.6 | 13.2 | 69.3 | 105.1 | 106.1 |
| pfss-dhcs | default | 18 | 0 | 70.0 | 0.667 | 69.8 | 73.5 | 60.9 | 89.8 | 105.1 | 106.1 |
| pfss-shadow | baseline | 18 | 0 | 67.8 | 0.646 | 70.0 | 69.6 | 13.2 | 69.3 | 105.1 | 106.1 |
| pfss-shadow | default | 18 | 0 | 77.3 | 0.735 | 79.2 | 78.4 | 70.5 | 108.9 | 105.1 | 106.1 |
| pfss-shadow-2 | baseline | 18 | 0 | 67.8 | 0.646 | 70.0 | 69.6 | 13.2 | 69.3 | 105.1 | 106.1 |
| pfss-shadow-2 | default | 18 | 0 | 77.3 | 0.735 | 79.2 | 78.4 | 70.5 | 108.9 | 105.1 | 106.1 |
| pfss-shadow-3 | baseline | 18 | 0 | 67.8 | 0.646 | 70.0 | 69.6 | 13.2 | 69.3 | 105.1 | 106.1 |
| pfss-shadow-3 | default | 18 | 0 | 77.3 | 0.735 | 79.2 | 78.4 | 70.5 | 108.9 | 105.1 | 106.1 |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 28 | 0 | 0 | 6.9 | 26.6 | 5.4 | — | — | — |
| linear-v2 | default | 28 | 0 | 0 | 7.2 | 101.8 | 5.4 | 100.0% | — | — |
| pfss-dhcs | baseline | 28 | 0 | 0 | 6.65 | 22.4 | 5.383333333333334 | — | — | — |
| pfss-dhcs | default | 28 | 0 | 0 | 10.5 | 137.2 | 5.383333333333334 | 100.0% | — | — |
| pfss-shadow | baseline | 28 | 0 | 0 | 6.800000000000001 | 9.8 | 5.366666666666666 | — | — | — |
| pfss-shadow | default | 28 | 0 | 0 | 42.8 | 95.1 | 5.366666666666666 | 100.0% | pfss ×28 | — |
| pfss-shadow-2 | baseline | 28 | 0 | 0 | 6.1 | 9.3 | 5.383333333333334 | — | — | — |
| pfss-shadow-2 | default | 28 | 0 | 0 | 47.0 | 105.6 | 5.383333333333334 | 100.0% | pfss ×28 | — |
| pfss-shadow-3 | baseline | 28 | 0 | 0 | 5.449999999999999 | 9.6 | 5.383333333333334 | — | — | — |
| pfss-shadow-3 | default | 28 | 0 | 0 | 9.25 | 81.6 | 5.383333333333334 | 100.0% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 79 | 0 | 0 | 6.6 | 33.3 | 5.4 | — | — | — |
| linear-v2 | default | 79 | 0 | 0 | 6.9 | 101.8 | 5.4 | 100.0% | — | — |
| pfss-dhcs | baseline | 30 | 0 | 0 | 6.8 | 22.4 | 5.391666666666667 | — | — | — |
| pfss-dhcs | default | 30 | 0 | 0 | 10.649999999999999 | 137.2 | 5.391666666666667 | 100.0% | — | — |
| pfss-shadow | baseline | 32 | 0 | 0 | 6.95 | 9.8 | 5.375 | — | — | — |
| pfss-shadow | default | 32 | 0 | 0 | 42.849999999999994 | 95.1 | 5.375 | 100.0% | pfss ×32 | — |
| pfss-shadow-2 | baseline | 32 | 0 | 0 | 6.4 | 9.3 | 5.391666666666667 | — | — | — |
| pfss-shadow-2 | default | 32 | 0 | 0 | 47.05 | 105.6 | 5.391666666666667 | 100.0% | pfss ×32 | — |
| pfss-shadow-3 | baseline | 32 | 0 | 0 | 5.1 | 9.6 | 5.383333333333334 | — | — | — |
| pfss-shadow-3 | default | 32 | 0 | 0 | 9.65 | 81.6 | 5.383333333333334 | 100.0% | — | — |

## Schedule

Last cycle: 2026-11-15 06:00 UTC
Missed cycles: 2 (10-26 12Z, 10-27 06Z)

## Alerts, last 7 days

None.

## Other events, last 7 days (counts; details in events/)

source not ok: 56

