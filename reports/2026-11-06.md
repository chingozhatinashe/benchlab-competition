# Staging report 2026-11-06

Generated 2026-11-06 11:26 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-11-06 10:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 31 | 0 | 51.1 | 0.494 | 53.6 | 50.3 | 37.5 | 52.7 | 103.4 | 78.8 |
| linear-v2 | default | 31 | 0 | 53.4 | 0.516 | 49.9 | 59.2 | 17.8 | 51.2 | 103.4 | 78.8 |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 31 | 0 | 51.1 | 0.494 | 53.6 | 50.3 | 37.5 | 52.7 | 103.4 | 78.8 |
| linear-v2 | default | 31 | 0 | 53.4 | 0.516 | 49.9 | 59.2 | 17.8 | 51.2 | 103.4 | 78.8 |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 28 | 0 | 0 | 5.5 | 33.3 | 5.4 | — | — | — |
| linear-v2 | default | 28 | 0 | 0 | 6.5 | 30.0 | 5.4 | 100.0% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 43 | 0 | 0 | 5.8 | 33.3 | 5.433333333333334 | — | — | — |
| linear-v2 | default | 43 | 0 | 0 | 6.6 | 30.0 | 5.433333333333334 | 100.0% | — | — |

## Schedule

Last cycle: 2026-11-06 06:00 UTC
Missed cycles: 2 (10-26 12Z, 10-27 06Z)

## Alerts, last 7 days

None.

## Other events, last 7 days (counts; details in events/)

None.

