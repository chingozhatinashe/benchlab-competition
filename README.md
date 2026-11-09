# Staging report 2026-11-09

Generated 2026-11-09 12:02 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-11-09 10:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 43 | 0 | 89.2 | 0.647 | 73.8 | 111.2 | 35.1 | 134.8 | 137.8 | 116.1 |
| linear-v2 | default | 43 | 0 | 86.8 | 0.630 | 64.4 | 120.1 | 7.4 | 130.0 | 137.8 | 116.1 |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 43 | 0 | 89.2 | 0.647 | 73.8 | 111.2 | 35.1 | 134.8 | 137.8 | 116.1 |
| linear-v2 | default | 43 | 0 | 86.8 | 0.630 | 64.4 | 120.1 | 7.4 | 130.0 | 137.8 | 116.1 |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 27 | 0 | 0 | 6.8 | 9.8 | 5.4 | — | — | — |
| linear-v2 | default | 27 | 0 | 0 | 7.1 | 30.0 | 5.4 | 100.0% | — | — |
| pfss-dhcs | baseline | 6 | 0 | 0 | 6.6 | 7.6 | 5.4 | — | — | — |
| pfss-dhcs | default | 6 | 0 | 0 | 10.9 | 11.1 | 5.4 | 100.0% | — | — |
| pfss-shadow | baseline | 8 | 0 | 0 | 7.2 | 9.3 | 5.391666666666667 | — | — | — |
| pfss-shadow | default | 8 | 0 | 0 | 42.849999999999994 | 62.3 | 5.391666666666667 | 100.0% | pfss ×8 | — |
| pfss-shadow-2 | baseline | 8 | 0 | 0 | 7.1 | 9.0 | 5.383333333333334 | — | — | — |
| pfss-shadow-2 | default | 8 | 0 | 0 | 47.0 | 47.7 | 5.383333333333334 | 100.0% | pfss ×8 | — |
| pfss-shadow-3 | baseline | 8 | 0 | 0 | 4.449999999999999 | 6.9 | 5.366666666666666 | — | — | — |
| pfss-shadow-3 | default | 8 | 0 | 0 | 10.9 | 12.4 | 5.366666666666666 | 100.0% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 55 | 0 | 0 | 5.8 | 33.3 | 5.4 | — | — | — |
| linear-v2 | default | 55 | 0 | 0 | 6.6 | 30.0 | 5.4 | 100.0% | — | — |
| pfss-dhcs | baseline | 6 | 0 | 0 | 6.6 | 7.6 | 5.4 | — | — | — |
| pfss-dhcs | default | 6 | 0 | 0 | 10.9 | 11.1 | 5.4 | 100.0% | — | — |
| pfss-shadow | baseline | 8 | 0 | 0 | 7.2 | 9.3 | 5.391666666666667 | — | — | — |
| pfss-shadow | default | 8 | 0 | 0 | 42.849999999999994 | 62.3 | 5.391666666666667 | 100.0% | pfss ×8 | — |
| pfss-shadow-2 | baseline | 8 | 0 | 0 | 7.1 | 9.0 | 5.383333333333334 | — | — | — |
| pfss-shadow-2 | default | 8 | 0 | 0 | 47.0 | 47.7 | 5.383333333333334 | 100.0% | pfss ×8 | — |
| pfss-shadow-3 | baseline | 8 | 0 | 0 | 4.449999999999999 | 6.9 | 5.366666666666666 | — | — | — |
| pfss-shadow-3 | default | 8 | 0 | 0 | 10.9 | 12.4 | 5.366666666666666 | 100.0% | — | — |

## Schedule

Last cycle: 2026-11-09 06:00 UTC
Missed cycles: 2 (10-26 12Z, 10-27 06Z)

## Alerts, last 7 days

None.

## Other events, last 7 days (counts; details in events/)

source not ok: 16, candidate added: 4

