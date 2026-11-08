# Staging report 2026-11-08

Generated 2026-11-08 10:43 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-11-08 08:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 39 | 0 | 64.9 | 0.603 | 64.2 | 67.9 | 43.7 | 85.5 | 107.6 | 90.4 |
| linear-v2 | default | 39 | 0 | 62.9 | 0.585 | 52.3 | 78.4 | 17.2 | 80.2 | 107.6 | 90.4 |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 39 | 0 | 64.9 | 0.603 | 64.2 | 67.9 | 43.7 | 85.5 | 107.6 | 90.4 |
| linear-v2 | default | 39 | 0 | 62.9 | 0.585 | 52.3 | 78.4 | 17.2 | 80.2 | 107.6 | 90.4 |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 28 | 0 | 0 | 6.75 | 33.3 | 5.4 | — | — | — |
| linear-v2 | default | 28 | 0 | 0 | 6.9 | 30.0 | 5.4 | 100.0% | — | — |
| pfss-dhcs | baseline | 2 | 0 | 0 | 7.199999999999999 | 7.6 | 5.441666666666666 | — | — | — |
| pfss-dhcs | default | 2 | 0 | 0 | 11.05 | 11.1 | 5.441666666666666 | 100.0% | — | — |
| pfss-shadow | baseline | 4 | 0 | 0 | 7.2 | 7.6 | 5.4 | — | — | — |
| pfss-shadow | default | 4 | 0 | 0 | 42.95 | 62.3 | 5.4 | 100.0% | pfss ×4 | — |
| pfss-shadow-2 | baseline | 4 | 0 | 0 | 7.1 | 7.5 | 5.408333333333333 | — | — | — |
| pfss-shadow-2 | default | 4 | 0 | 0 | 47.1 | 47.7 | 5.408333333333333 | 100.0% | pfss ×4 | — |
| pfss-shadow-3 | baseline | 4 | 0 | 0 | 3.9 | 6.9 | 5.375 | — | — | — |
| pfss-shadow-3 | default | 4 | 0 | 0 | 11.0 | 11.9 | 5.375 | 100.0% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 51 | 0 | 0 | 6.5 | 33.3 | 5.433333333333334 | — | — | — |
| linear-v2 | default | 51 | 0 | 0 | 6.6 | 30.0 | 5.433333333333334 | 100.0% | — | — |
| pfss-dhcs | baseline | 2 | 0 | 0 | 7.199999999999999 | 7.6 | 5.441666666666666 | — | — | — |
| pfss-dhcs | default | 2 | 0 | 0 | 11.05 | 11.1 | 5.441666666666666 | 100.0% | — | — |
| pfss-shadow | baseline | 4 | 0 | 0 | 7.2 | 7.6 | 5.4 | — | — | — |
| pfss-shadow | default | 4 | 0 | 0 | 42.95 | 62.3 | 5.4 | 100.0% | pfss ×4 | — |
| pfss-shadow-2 | baseline | 4 | 0 | 0 | 7.1 | 7.5 | 5.408333333333333 | — | — | — |
| pfss-shadow-2 | default | 4 | 0 | 0 | 47.1 | 47.7 | 5.408333333333333 | 100.0% | pfss ×4 | — |
| pfss-shadow-3 | baseline | 4 | 0 | 0 | 3.9 | 6.9 | 5.375 | — | — | — |
| pfss-shadow-3 | default | 4 | 0 | 0 | 11.0 | 11.9 | 5.375 | 100.0% | — | — |

## Schedule

Last cycle: 2026-11-08 06:00 UTC
Missed cycles: 2 (10-26 12Z, 10-27 06Z)

## Alerts, last 7 days

None.

## Other events, last 7 days (counts; details in events/)

source not ok: 8, candidate added: 4

