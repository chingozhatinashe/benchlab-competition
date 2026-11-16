# Staging report 2026-11-16

Generated 2026-11-16 11:59 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-11-16 10:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 71 | 0 | 114.1 | 0.772 | 88.6 | 147.3 | 16.0 | 144.2 | 147.8 | 140.8 |
| linear-v2 | default | 71 | 0 | 95.6 | 0.647 | 76.1 | 121.0 | 18.7 | 128.0 | 147.8 | 140.8 |
| pfss-dhcs | baseline | 22 | 0 | 68.8 | 0.683 | 71.1 | 69.9 | 20.6 | 71.5 | 100.8 | 100.8 |
| pfss-dhcs | default | 22 | 0 | 67.8 | 0.672 | 66.5 | 72.8 | 57.3 | 87.9 | 100.8 | 100.8 |
| pfss-shadow | baseline | 24 | 0 | 90.9 | 0.822 | 83.8 | 102.2 | 3.6 | 78.8 | 110.6 | 113.8 |
| pfss-shadow | default | 24 | 0 | 78.3 | 0.708 | 77.7 | 81.4 | 59.1 | 104.9 | 110.6 | 113.8 |
| pfss-shadow-2 | baseline | 24 | 0 | 90.9 | 0.822 | 83.8 | 102.2 | 3.6 | 78.8 | 110.6 | 113.8 |
| pfss-shadow-2 | default | 24 | 0 | 78.3 | 0.708 | 77.8 | 81.4 | 59.1 | 104.9 | 110.6 | 113.8 |
| pfss-shadow-3 | baseline | 24 | 0 | 90.9 | 0.822 | 83.8 | 102.2 | 3.6 | 78.8 | 110.6 | 113.8 |
| pfss-shadow-3 | default | 24 | 0 | 78.3 | 0.708 | 77.7 | 81.4 | 59.1 | 104.9 | 110.6 | 113.8 |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 22 | 0 | 68.8 | 0.683 | 71.1 | 69.9 | 20.6 | 71.5 | 100.8 | 100.8 |
| linear-v2 | default | 22 | 0 | 74.9 | 0.742 | 75.3 | 77.8 | 66.3 | 105.4 | 100.8 | 100.8 |
| pfss-dhcs | baseline | 22 | 0 | 68.8 | 0.683 | 71.1 | 69.9 | 20.6 | 71.5 | 100.8 | 100.8 |
| pfss-dhcs | default | 22 | 0 | 67.8 | 0.672 | 66.5 | 72.8 | 57.3 | 87.9 | 100.8 | 100.8 |
| pfss-shadow | baseline | 22 | 0 | 68.8 | 0.683 | 71.1 | 69.9 | 20.6 | 71.5 | 100.8 | 100.8 |
| pfss-shadow | default | 22 | 0 | 74.9 | 0.742 | 75.3 | 77.8 | 66.3 | 105.4 | 100.8 | 100.8 |
| pfss-shadow-2 | baseline | 22 | 0 | 68.8 | 0.683 | 71.1 | 69.9 | 20.6 | 71.5 | 100.8 | 100.8 |
| pfss-shadow-2 | default | 22 | 0 | 74.9 | 0.743 | 75.4 | 77.9 | 66.3 | 105.4 | 100.8 | 100.8 |
| pfss-shadow-3 | baseline | 22 | 0 | 68.8 | 0.683 | 71.1 | 69.9 | 20.6 | 71.5 | 100.8 | 100.8 |
| pfss-shadow-3 | default | 22 | 0 | 74.9 | 0.742 | 75.3 | 77.8 | 66.3 | 105.4 | 100.8 | 100.8 |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 28 | 0 | 0 | 7.2 | 26.6 | 5.4 | — | — | — |
| linear-v2 | default | 28 | 0 | 0 | 7.2 | 101.8 | 5.4 | 100.0% | — | — |
| pfss-dhcs | baseline | 28 | 0 | 0 | 6.7 | 22.4 | 5.4 | — | — | — |
| pfss-dhcs | default | 28 | 0 | 0 | 10.5 | 137.2 | 5.4 | 100.0% | — | — |
| pfss-shadow | baseline | 28 | 0 | 0 | 6.4 | 9.8 | 5.4 | — | — | — |
| pfss-shadow | default | 28 | 0 | 0 | 42.8 | 95.1 | 5.4 | 100.0% | pfss ×28 | — |
| pfss-shadow-2 | baseline | 28 | 0 | 0 | 5.8 | 9.3 | 5.4 | — | — | — |
| pfss-shadow-2 | default | 28 | 0 | 0 | 47.0 | 105.6 | 5.4 | 100.0% | pfss ×28 | — |
| pfss-shadow-3 | baseline | 28 | 0 | 0 | 5.8 | 9.6 | 5.4 | — | — | — |
| pfss-shadow-3 | default | 28 | 0 | 0 | 9.2 | 81.6 | 5.4 | 100.0% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 83 | 0 | 0 | 6.6 | 33.3 | 5.4 | — | — | — |
| linear-v2 | default | 83 | 0 | 0 | 6.9 | 101.8 | 5.4 | 100.0% | — | — |
| pfss-dhcs | baseline | 34 | 0 | 0 | 6.7 | 22.4 | 5.4 | — | — | — |
| pfss-dhcs | default | 34 | 0 | 0 | 10.6 | 137.2 | 5.4 | 100.0% | — | — |
| pfss-shadow | baseline | 36 | 0 | 0 | 7.0 | 9.8 | 5.4 | — | — | — |
| pfss-shadow | default | 36 | 0 | 0 | 42.8 | 95.1 | 5.4 | 100.0% | pfss ×36 | — |
| pfss-shadow-2 | baseline | 36 | 0 | 0 | 6.2 | 9.3 | 5.4 | — | — | — |
| pfss-shadow-2 | default | 36 | 0 | 0 | 47.0 | 105.6 | 5.4 | 100.0% | pfss ×36 | — |
| pfss-shadow-3 | baseline | 36 | 0 | 0 | 5.4 | 9.6 | 5.4 | — | — | — |
| pfss-shadow-3 | default | 36 | 0 | 0 | 9.4 | 81.6 | 5.4 | 100.0% | — | — |

## Schedule

Last cycle: 2026-11-16 06:00 UTC
Missed cycles: 2 (10-26 12Z, 10-27 06Z)

## Alerts, last 7 days

None.

## Other events, last 7 days (counts; details in events/)

source not ok: 56

