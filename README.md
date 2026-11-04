# Staging report 2026-11-04

Generated 2026-11-04 11:30 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-11-04 08:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 23 | 0 | 48.3 | 0.450 | 48.6 | 49.5 | 29.6 | 50.2 | 107.3 | 84.0 |
| linear-v2 | default | 23 | 0 | 55.9 | 0.521 | 53.2 | 60.9 | 15.3 | 54.0 | 107.3 | 84.0 |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 23 | 0 | 48.3 | 0.450 | 48.6 | 49.5 | 29.6 | 50.2 | 107.3 | 84.0 |
| linear-v2 | default | 23 | 0 | 55.9 | 0.521 | 53.2 | 60.9 | 15.3 | 54.0 | 107.3 | 84.0 |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 28 | 0 | 0 | 6.15 | 33.3 | 5.4 | — | — | — |
| linear-v2 | default | 28 | 0 | 0 | 6.699999999999999 | 27.2 | 5.4 | 100.0% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 35 | 0 | 0 | 5.2 | 33.3 | 5.433333333333334 | — | — | — |
| linear-v2 | default | 35 | 0 | 0 | 6.4 | 27.2 | 5.433333333333334 | 100.0% | — | — |

## Schedule

Last cycle: 2026-11-04 06:00 UTC
Missed cycles: 2 (10-26 12Z, 10-27 06Z)

## Alerts, last 7 days

None.

## Other events, last 7 days (counts; details in events/)

None.

