# Staging report 2026-10-27

Generated 2026-10-27 09:19 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-10-27 08:00 UTC

## Scores (runs whose 72 h have passed)

No completed runs yet.

## Scores on the t0s every current candidate ran

—

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 3 | 0 | 0 | 4.9 | 8.3 | 294.0833333333333 | — | — | — |
| linear-v2 | default | 3 | 0 | 0 | 5.3 | 8.4 | 294.0833333333333 | 100.0% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 3 | 0 | 0 | 4.9 | 8.3 | 294.0833333333333 | — | — | — |
| linear-v2 | default | 3 | 0 | 0 | 5.3 | 8.4 | 294.0833333333333 | 100.0% | — | — |

## Schedule

Last cycle: 2026-10-27 00:00 UTC
Missed cycles: 2 (10-26 12Z, 10-27 06Z)

## Alerts, last 7 days

| t0 | candidate | variant | kind | detail |
|---|---|---|---|---|
| 20261026T120000Z |  |  | missed cycle | no staging cycle ran for this t0 |
| 20261027T060000Z |  |  | missed cycle | no staging cycle ran for this t0 |

## Other events, last 7 days (counts; details in events/)

candidate added: 1

