# Staging report 2026-11-01

Generated 2026-11-01 10:46 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-11-01 09:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 11 | 0 | 33.7 | 0.609 | 41.6 | 29.7 | 28.1 | 40.1 | 55.3 | 77.8 |
| linear-v2 | default | 11 | 0 | 41.3 | 0.747 | 49.1 | 37.6 | 28.7 | 52.5 | 55.3 | 77.8 |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 11 | 0 | 33.7 | 0.609 | 41.6 | 29.7 | 28.1 | 40.1 | 55.3 | 77.8 |
| linear-v2 | default | 11 | 0 | 41.3 | 0.747 | 49.1 | 37.6 | 28.7 | 52.5 | 55.3 | 77.8 |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 23 | 0 | 0 | 5.2 | 11.2 | 5.433333333333334 | — | — | — |
| linear-v2 | default | 23 | 0 | 0 | 6.4 | 8.7 | 5.433333333333334 | 100.0% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 23 | 0 | 0 | 5.2 | 11.2 | 5.433333333333334 | — | — | — |
| linear-v2 | default | 23 | 0 | 0 | 6.4 | 8.7 | 5.433333333333334 | 100.0% | — | — |

## Schedule

Last cycle: 2026-11-01 06:00 UTC
Missed cycles: 2 (10-26 12Z, 10-27 06Z)

## Alerts, last 7 days

| t0 | candidate | variant | kind | detail |
|---|---|---|---|---|
| 20261026T120000Z |  |  | missed cycle | no staging cycle ran for this t0 |
| 20261027T060000Z |  |  | missed cycle | no staging cycle ran for this t0 |
| 20261028T060000Z |  |  | missed cycle | no staging cycle ran for this t0 |

## Other events, last 7 days (counts; details in events/)

candidate added: 1

