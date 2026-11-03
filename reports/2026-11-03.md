# Staging report 2026-11-03

Generated 2026-11-03 11:17 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-11-03 10:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 19 | 0 | 40.0 | 0.389 | 40.2 | 41.9 | 21.2 | 44.6 | 103.0 | 87.9 |
| linear-v2 | default | 19 | 0 | 50.1 | 0.486 | 47.3 | 55.5 | 4.5 | 49.4 | 103.0 | 87.9 |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 19 | 0 | 40.0 | 0.389 | 40.2 | 41.9 | 21.2 | 44.6 | 103.0 | 87.9 |
| linear-v2 | default | 19 | 0 | 50.1 | 0.486 | 47.3 | 55.5 | 4.5 | 49.4 | 103.0 | 87.9 |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 28 | 0 | 0 | 5.0 | 33.3 | 5.4 | — | — | — |
| linear-v2 | default | 28 | 0 | 0 | 6.1 | 27.2 | 5.4 | 100.0% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 31 | 0 | 0 | 4.9 | 33.3 | 5.433333333333334 | — | — | — |
| linear-v2 | default | 31 | 0 | 0 | 6.0 | 27.2 | 5.433333333333334 | 100.0% | — | — |

## Schedule

Last cycle: 2026-11-03 06:00 UTC
Missed cycles: 2 (10-26 12Z, 10-27 06Z)

## Alerts, last 7 days

| t0 | candidate | variant | kind | detail |
|---|---|---|---|---|
| 20261028T060000Z |  |  | missed cycle | no staging cycle ran for this t0 |

## Other events, last 7 days (counts; details in events/)

None.

