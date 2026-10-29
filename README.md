# Staging report 2026-10-29

Generated 2026-10-29 10:22 UTC. Every `upload_candidate/*` tag runs every 6 h as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official metric (`training.competition_score`; S in km/s, lower is better). This report only shows scores and health: what ships is chosen by hand and uploaded by the "Upload to Benchlab" workflow on main.

Official truth up to 2026-10-29 08:00 UTC

## Scores (runs whose 72 h have passed)

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 1 | 0 | — | — | 23.1 | — | -11.4 | — | — | — |
| linear-v2 | default | 1 | 0 | — | — | 44.8 | — | -44.5 | — | — | — |

## Scores on the t0s every current candidate ran

| candidate | variant | runs | failed (→ persistence) | S_final | / persistence | RMSE | TRMSE | bias | S_49:72 | persistence S | recurrence S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 1 | 0 | — | — | 23.1 | — | -11.4 | — | — | — |
| linear-v2 | default | 1 | 0 | — | — | 44.8 | — | -44.5 | — | — | — |

## Health, last 7 days

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 11 | 0 | 0 | 4.9 | 11.2 | 247.6 | — | — | — |
| linear-v2 | default | 11 | 0 | 0 | 5.4 | 8.7 | 247.6 | 100.0% | — | — |

## Health, all time

| candidate | variant | runs | failed | fell back | runtime p50 (s) | runtime max (s) | start delay p50 (min) | regressors observed | sources not ok | undeclared hosts |
|---|---|---|---|---|---|---|---|---|---|---|
| linear-v2 | baseline | 11 | 0 | 0 | 4.9 | 11.2 | 247.6 | — | — | — |
| linear-v2 | default | 11 | 0 | 0 | 5.4 | 8.7 | 247.6 | 100.0% | — | — |

## Schedule

Last cycle: 2026-10-29 06:00 UTC
Missed cycles: 2 (10-26 12Z, 10-27 06Z)

## Alerts, last 7 days

| t0 | candidate | variant | kind | detail |
|---|---|---|---|---|
| 20261026T120000Z |  |  | missed cycle | no staging cycle ran for this t0 |
| 20261027T060000Z |  |  | missed cycle | no staging cycle ran for this t0 |
| 20261028T060000Z |  |  | missed cycle | no staging cycle ran for this t0 |

## Other events, last 7 days (counts; details in events/)

candidate added: 1

