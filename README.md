# Staging report 2026-11-10

Generated 2026-11-10 11:41 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-11-10 09:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 47 | 0 | 120.9 | 0.728 | 90.3 | 164.0 | 18.1 | 160.0 | 166.1 | 144.0 |
| linear-v2 | default | 47 | 0 | 103.1 | 0.621 | 75.1 | 143.2 | -2.7 | 132.6 | 166.1 | 144.0 |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 47 | 0 | 120.9 | 0.728 | 90.3 | 164.0 | 18.1 | 160.0 | 166.1 | 144.0 |
| linear-v2 | default | 47 | 0 | 103.1 | 0.621 | 75.1 | 143.2 | -2.7 | 132.6 | 166.1 | 144.0 |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 28 | 0 | 0 | 6.95 | 24.1 | 5.408333333333333 | — | — | — |
| linear-v2 | default | 28 | 0 | 0 | 7.1 | 30.0 | 5.408333333333333 | 100.0% | — | — |
| pfss-dhcs | baseline | 10 | 0 | 0 | 6.6 | 22.4 | 5.383333333333334 | — | — | — |
| pfss-dhcs | default | 10 | 0 | 0 | 10.75 | 13.7 | 5.383333333333334 | 100.0% | — | — |
| pfss-shadow | baseline | 12 | 0 | 0 | 7.2 | 9.8 | 5.391666666666667 | — | — | — |
| pfss-shadow | default | 12 | 0 | 0 | 42.849999999999994 | 62.3 | 5.391666666666667 | 100.0% | pfss ×12 | — |
| pfss-shadow-2 | baseline | 12 | 0 | 0 | 6.7 | 9.0 | 5.383333333333334 | — | — | — |
| pfss-shadow-2 | default | 12 | 0 | 0 | 46.95 | 47.7 | 5.383333333333334 | 100.0% | pfss ×12 | — |
| pfss-shadow-3 | baseline | 12 | 0 | 0 | 5.45 | 9.5 | 5.375 | — | — | — |
| pfss-shadow-3 | default | 12 | 0 | 0 | 10.4 | 12.4 | 5.375 | 100.0% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 59 | 0 | 0 | 6.5 | 33.3 | 5.416666666666667 | — | — | — |
| linear-v2 | default | 59 | 0 | 0 | 6.7 | 30.0 | 5.416666666666667 | 100.0% | — | — |
| pfss-dhcs | baseline | 10 | 0 | 0 | 6.6 | 22.4 | 5.383333333333334 | — | — | — |
| pfss-dhcs | default | 10 | 0 | 0 | 10.75 | 13.7 | 5.383333333333334 | 100.0% | — | — |
| pfss-shadow | baseline | 12 | 0 | 0 | 7.2 | 9.8 | 5.391666666666667 | — | — | — |
| pfss-shadow | default | 12 | 0 | 0 | 42.849999999999994 | 62.3 | 5.391666666666667 | 100.0% | pfss ×12 | — |
| pfss-shadow-2 | baseline | 12 | 0 | 0 | 6.7 | 9.0 | 5.383333333333334 | — | — | — |
| pfss-shadow-2 | default | 12 | 0 | 0 | 46.95 | 47.7 | 5.383333333333334 | 100.0% | pfss ×12 | — |
| pfss-shadow-3 | baseline | 12 | 0 | 0 | 5.45 | 9.5 | 5.375 | — | — | — |
| pfss-shadow-3 | default | 12 | 0 | 0 | 10.4 | 12.4 | 5.375 | 100.0% | — | — |

## Schedule

Last cycle: 2026-11-10 06:00 UTC
Missed cycles: 2 (10-26 12Z, 10-27 06Z)

## Alerts, last 7 days

None.

## Other events, last 7 days (counts; details in events/)

source not ok: 24, candidate added: 4

