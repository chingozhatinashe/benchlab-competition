# Staging report 2026-11-05

Generated 2026-11-05 11:27 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-11-05 08:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 27 | 0 | 50.7 | 0.476 | 52.3 | 50.5 | 35.0 | 50.7 | 106.5 | 81.4 |
| linear-v2 | default | 27 | 0 | 54.8 | 0.514 | 52.0 | 59.7 | 18.4 | 52.5 | 106.5 | 81.4 |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 27 | 0 | 50.7 | 0.476 | 52.3 | 50.5 | 35.0 | 50.7 | 106.5 | 81.4 |
| linear-v2 | default | 27 | 0 | 54.8 | 0.514 | 52.0 | 59.7 | 18.4 | 52.5 | 106.5 | 81.4 |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 28 | 0 | 0 | 6.6 | 33.3 | 5.4 | — | — | — |
| linear-v2 | default | 28 | 0 | 0 | 6.85 | 30.0 | 5.4 | 100.0% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 39 | 0 | 0 | 6.5 | 33.3 | 5.433333333333334 | — | — | — |
| linear-v2 | default | 39 | 0 | 0 | 6.8 | 30.0 | 5.433333333333334 | 100.0% | — | — |

## Schedule

Last cycle: 2026-11-05 06:00 UTC
Missed cycles: 2 (10-26 12Z, 10-27 06Z)

## Alerts, last 7 days

None.

## Other events, last 7 days (counts; details in events/)

None.

