# Staging report 2026-11-12

Generated 2026-11-12 11:46 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-11-12 10:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 55 | 0 | 119.9 | 0.785 | 93.7 | 153.8 | 3.7 | 148.6 | 152.8 | 150.8 |
| linear-v2 | default | 55 | 0 | 98.4 | 0.644 | 78.1 | 124.7 | 4.4 | 129.6 | 152.8 | 150.8 |
| pfss-dhcs | baseline | 6 | 0 | 75.4 | 0.793 | 68.1 | 86.2 | -51.0 | 53.3 | 95.1 | 136.2 |
| pfss-dhcs | default | 6 | 0 | 80.2 | 0.844 | 81.7 | 77.2 | 65.3 | 97.7 | 95.1 | 136.2 |
| pfss-shadow | baseline | 8 | 0 | 112.5 | 0.954 | 98.6 | 126.9 | -77.7 | 71.1 | 117.9 | 155.9 |
| pfss-shadow | default | 8 | 0 | 90.4 | 0.766 | 91.5 | 85.9 | 46.6 | 108.9 | 117.9 | 155.9 |
| pfss-shadow-2 | baseline | 8 | 0 | 112.5 | 0.954 | 98.6 | 126.9 | -77.7 | 71.1 | 117.9 | 155.9 |
| pfss-shadow-2 | default | 8 | 0 | 90.4 | 0.766 | 91.5 | 85.9 | 46.6 | 108.9 | 117.9 | 155.9 |
| pfss-shadow-3 | baseline | 8 | 0 | 112.5 | 0.954 | 98.6 | 126.9 | -77.7 | 71.1 | 117.9 | 155.9 |
| pfss-shadow-3 | default | 8 | 0 | 90.4 | 0.766 | 91.5 | 85.9 | 46.6 | 108.9 | 117.9 | 155.9 |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 6 | 0 | 75.4 | 0.793 | 68.1 | 86.2 | -51.0 | 53.3 | 95.1 | 136.2 |
| linear-v2 | default | 6 | 0 | 84.8 | 0.891 | 86.3 | 81.4 | 68.6 | 110.6 | 95.1 | 136.2 |
| pfss-dhcs | baseline | 6 | 0 | 75.4 | 0.793 | 68.1 | 86.2 | -51.0 | 53.3 | 95.1 | 136.2 |
| pfss-dhcs | default | 6 | 0 | 80.2 | 0.844 | 81.7 | 77.2 | 65.3 | 97.7 | 95.1 | 136.2 |
| pfss-shadow | baseline | 6 | 0 | 75.4 | 0.793 | 68.1 | 86.2 | -51.0 | 53.3 | 95.1 | 136.2 |
| pfss-shadow | default | 6 | 0 | 84.8 | 0.891 | 86.3 | 81.4 | 68.6 | 110.6 | 95.1 | 136.2 |
| pfss-shadow-2 | baseline | 6 | 0 | 75.4 | 0.793 | 68.1 | 86.2 | -51.0 | 53.3 | 95.1 | 136.2 |
| pfss-shadow-2 | default | 6 | 0 | 84.8 | 0.891 | 86.3 | 81.4 | 68.6 | 110.6 | 95.1 | 136.2 |
| pfss-shadow-3 | baseline | 6 | 0 | 75.4 | 0.793 | 68.1 | 86.2 | -51.0 | 53.3 | 95.1 | 136.2 |
| pfss-shadow-3 | default | 6 | 0 | 84.8 | 0.891 | 86.3 | 81.4 | 68.6 | 110.6 | 95.1 | 136.2 |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 28 | 0 | 0 | 6.55 | 24.1 | 5.383333333333334 | — | — | — |
| linear-v2 | default | 28 | 0 | 0 | 6.9 | 23.1 | 5.383333333333334 | 100.0% | — | — |
| pfss-dhcs | baseline | 18 | 0 | 0 | 6.45 | 22.4 | 5.383333333333334 | — | — | — |
| pfss-dhcs | default | 18 | 0 | 0 | 10.9 | 13.7 | 5.383333333333334 | 100.0% | — | — |
| pfss-shadow | baseline | 20 | 0 | 0 | 6.85 | 9.8 | 5.375 | — | — | — |
| pfss-shadow | default | 20 | 0 | 0 | 42.9 | 62.3 | 5.375 | 100.0% | pfss ×20 | — |
| pfss-shadow-2 | baseline | 20 | 0 | 0 | 6.9 | 9.3 | 5.391666666666667 | — | — | — |
| pfss-shadow-2 | default | 20 | 0 | 0 | 47.05 | 47.7 | 5.391666666666667 | 100.0% | pfss ×20 | — |
| pfss-shadow-3 | baseline | 20 | 0 | 0 | 4.9 | 9.5 | 5.375 | — | — | — |
| pfss-shadow-3 | default | 20 | 0 | 0 | 10.4 | 12.4 | 5.375 | 100.0% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 67 | 0 | 0 | 6.5 | 33.3 | 5.4 | — | — | — |
| linear-v2 | default | 67 | 0 | 0 | 6.8 | 30.0 | 5.4 | 100.0% | — | — |
| pfss-dhcs | baseline | 18 | 0 | 0 | 6.45 | 22.4 | 5.383333333333334 | — | — | — |
| pfss-dhcs | default | 18 | 0 | 0 | 10.9 | 13.7 | 5.383333333333334 | 100.0% | — | — |
| pfss-shadow | baseline | 20 | 0 | 0 | 6.85 | 9.8 | 5.375 | — | — | — |
| pfss-shadow | default | 20 | 0 | 0 | 42.9 | 62.3 | 5.375 | 100.0% | pfss ×20 | — |
| pfss-shadow-2 | baseline | 20 | 0 | 0 | 6.9 | 9.3 | 5.391666666666667 | — | — | — |
| pfss-shadow-2 | default | 20 | 0 | 0 | 47.05 | 47.7 | 5.391666666666667 | 100.0% | pfss ×20 | — |
| pfss-shadow-3 | baseline | 20 | 0 | 0 | 4.9 | 9.5 | 5.375 | — | — | — |
| pfss-shadow-3 | default | 20 | 0 | 0 | 10.4 | 12.4 | 5.375 | 100.0% | — | — |

## Schedule

Last cycle: 2026-11-12 06:00 UTC
Missed cycles: 2 (10-26 12Z, 10-27 06Z)

## Alerts, last 7 days

None.

## Other events, last 7 days (counts; details in events/)

source not ok: 40, candidate added: 4

