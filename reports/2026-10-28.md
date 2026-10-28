# Staging report 2026-10-28

Generated 2026-10-28 09:59 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-10-28 06:00 UTC

## Scores (runs whose 72 h have passed)

No completed runs yet.

## Scores on the t0s every current candidate ran

—

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 6 | 0 | 0 | 4.75 | 8.3 | 311.55 | — | — | — |
| linear-v2 | default | 6 | 0 | 0 | 5.35 | 8.4 | 311.55 | 100.0% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 6 | 0 | 0 | 4.75 | 8.3 | 311.55 | — | — | — |
| linear-v2 | default | 6 | 0 | 0 | 5.35 | 8.4 | 311.55 | 100.0% | — | — |

## Schedule

Last cycle: 2026-10-28 00:00 UTC
Missed cycles: 3 (10-26 12Z, 10-27 06Z, 10-28 06Z)

## Alerts, last 7 days

| t0 | candidate | variant | kind | detail |
|---|---|---|---|---|
| 20261026T120000Z |  |  | missed cycle | no staging cycle ran for this t0 |
| 20261027T060000Z |  |  | missed cycle | no staging cycle ran for this t0 |
| 20261028T060000Z |  |  | missed cycle | no staging cycle ran for this t0 |

## Other events, last 7 days (counts; details in events/)

candidate added: 1

