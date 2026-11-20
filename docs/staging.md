# Staging: upload candidates run like the live evaluation

**What it is.** A staging environment in front of Benchlab.
- **Runs:** every commit tagged `upload_candidate/<name>` runs every 6 hours, the way Benchlab runs the qualified submission.
- **Scoring:** it is scored daily with the official metric.
- **Logging:** every problem is logged.

The goal is to find out, before and during the live evaluation, whether the forecast infrastructure holds up: builds, data sources, fallbacks, time and declared domains. When there are several candidates, it also shows how each would have scored.

**Staging only scores and records.** It never ranks candidates, picks one or uploads anything. What ships is your decision. It goes to Benchlab only through the "Upload to Benchlab" workflow on main ([CLAUDE.md](../CLAUDE.md), "Uploading"), which sends main's CI-tested zip.

## Candidates

```bash
git tag -a upload_candidate/linear-v2 <sha> -m "re-tuned linear model, official weights b=0.25"
git push origin upload_candidate/linear-v2
git push origin :refs/tags/upload_candidate/linear-v2   # stop running it (its records stay)
```

- **Names:** letters, digits, `.`, `_` and `-`.
- **Moving a tag:** re-tagging the same name on a new commit is logged as "candidate moved".

## What runs (`.github/workflows/staging.yml`)

| Job | When (UTC) | What |
|---|---|---|
| `cycle` | t0 + 5 min from the on-time trigger; GitHub's cron at 00:07, 06:07, 12:07, 18:07 as the backup | Each candidate's image is built from its own commit and run at the t0 in fresh containers. A t0 already recorded is skipped. |
| `evaluate` | 03:40 | Scores, health, missed cycles and the report. |

Both can be started by hand: Actions → Staging → Run workflow, `job = cycle | evaluate`, on main. A cycle takes an optional `t0` (a 6-hourly mark in the last 24 h; empty = the current one).

**The on-time trigger.** GitHub's cron starts runs hours late and drops some (see Limits), so a scheduled task on the developer's Windows laptop dispatches each cycle, and the Benchlab monitor, instead.
- `ci/staging/trigger_cycle.ps1` runs every 10 min, at :05, :15, …. Once t0 + 5 min has passed, it runs `gh workflow run staging.yml -f job=cycle -f t0=<t0>`, once per t0. Once t0 + 30 min has passed (Benchlab's live run is done by about t0 + 7 min), it runs `gh workflow run monitor.yml -f mode=record`, once per t0. After a sleep it catches up on wake, for the current t0 only.
- After changing `trigger_cycle.ps1`, run the installer again: the task runs its own copy.
- **Install / remove:** `powershell -ExecutionPolicy Bypass -File ci\staging\install_trigger.ps1 [-Uninstall]`.
  - The task copies the script to `%LOCALAPPDATA%\benchlab-staging`, where its state (`last_t0.txt`, `last_monitor_t0.txt`) and log (`trigger.log`, dispatches and failures only) are kept.
  - It runs `gh` with your login, which needs the `workflow` scope, and only while you are logged on.
- **When the laptop is off or asleep at a slot,** GitHub's cron runs that cycle later, as before. Whichever fires second finds the cycle recorded and skips it. If both start before either has recorded, the cycle runs twice, and the later record replaces the earlier one.

**A cycle, per candidate** (`ci/staging/run_candidate.sh`):
1. **Build:** `docker build` of the candidate's `src/`, with the base image from the Actions cache (`.github/actions/base-image`, shared with CI). Build time and image size are recorded. A failed build is retried once after 60 s (usually a package-archive outage, as on 2026-11-19 12Z when `apt-get update` timed out for every candidate); the recorded build time then includes the retry.
2. **Up to three variants, each in a fresh container, exactly as `local_tests/test_local.sh` runs it:**
   - `default`, the submitted model;
   - `baseline` (`SWP_MODEL=baseline`), its fallback;
   - `bayes` (`SWP_MODEL=bayes`), the [Bayes model](bayes_model.md) option (since 2026-11-17). It runs only for candidates whose `inference/pipeline.py` has it; older candidates skip it.

   Each run gets:
   - argv t0, `INFERENCE_TIMESTAMP`, `SUBMISSION_ID`;
   - `OUTPUT_POST` pointing at `mock_platform.py`;
   - the CPU and memory declared in the candidate's `submission.json`;
   - a 900 s `timeout`.
3. **Declared domains:** `tcpdump` captures every DNS query from the container, and each host is checked against the candidate's `declared_domains` (a subdomain of a declared domain counts).
4. **A record per run** (`ci/staging/record_run.py`):
   - start delay after t0, build and run times, image size and estimated pull time;
   - exit code, whether the forecast is valid (72 values in 0–10000 km/s, exit code 0);
   - the model actually used and whether it fell back;
   - regressors observed and missing;
   - data sources' status, MB and seconds;
   - hosts queried;
   - the forecast and the log.

**The record job** (`ci/staging/events.py`):
- files the records on the orphan branch `staging-records`;
- appends the events;
- posts the alerts.

**The evaluate job** (`ci/staging/evaluate.py`):
- **Truth:** refreshes the official truth (SWPC's hourly ACE files, kept in `truth/`; the current and previous month are fetched again).
- **Scores:** scores every run whose 72 h have passed with `training.competition_score`, exactly as the organisers' spec does:
  - per lead, RMSE and percentile TIMSE, combined and averaged over the leads;
  - **a run that failed, timed out or gave invalid output counts as persistence**, as Benchlab scores it;
  - a cycle the staging schedule itself missed is left out for everyone;
  - persistence and 27-day recurrence are scored on the same t0s.
- **Tables:** all runs, and the t0s every current candidate ran, so candidates are comparable.
- **Health** (last 7 days and all time):
  - runs, failures, fallbacks;
  - runtime p50/max and start delay;
  - regressors observed, sources failing, undeclared hosts.
- **Missed cycles:** expected 6-hourly cycles with no `cycle` event.

## The records branch (`staging-records`)

```
README.md                       the latest daily report
reports/YYYY-MM-DD.md           every daily report
scores/YYYY-MM-DD.json          every score with its lead curves
runs/YYYY-MM/<t0>/<candidate>__<variant>.json   one record per run
events/YYYY-MM.csv              time, t0, candidate, sha, variant, kind, severity, detail
candidates.json                 the candidates of the last cycle
truth/YYYYMM_ace_swepam_1h.txt  the official truth
live/status.json                the submission's status on Benchlab (monitor.yml)
live/runs/YYYY-MM.jsonl         Benchlab's live runs, as the portal lists them
live/logs/YYYY-MM/*.log         Benchlab's run logs, scrubbed
live/state.json                 what the monitor has seen
```

The repository is public, so the branch is too. Records hold forecasts, logs and host names, but no secrets.

## Events and alerts

Every event goes to `events/`. Issues are off on this repository, so alerts are reported in the workflow run instead: one error annotation per alert (titled `Staging: <candidate>` or `Staging: schedule`) and a "Staging alerts" section in the job summary. The run then fails, after the records are pushed, and GitHub emails you the failure. A failed Staging run therefore means "look at the alerts", not lost data.

| Severity | Events |
|---|---|
| alert | build failed, failed (exit code), timeout, invalid output, over budget (run + estimated pull > 900 s), slow (> 600 s), image too large (> 15 GiB), fell back, few regressors (< 75% observed), undeclared host, no record, missed cycle, schedule stale (no cycle for 18 h) |
| info | run ok, missing regressors, source not ok, dns not captured, candidate added / moved / removed, cycle |

The report lists the last 7 days of alerts and counts the info events.

## Benchlab's live runs (`.github/workflows/monitor.yml`)

The "Monitor Benchlab" workflow reads the portal at t0 + 30 min from the on-time trigger, every 3 h from GitHub's cron as the backup (which in November 2026 dropped most passes), and on demand and files the submitted model's live runs on the same branch (`ci/monitor/live.py`).

- **Run statuses:** each finished run's status fields (status, exit code, failure reason, output bytes, pull and run seconds, start and stop times, retries) go to `live/runs/`. Task ARNs, network ids, IPs and object keys are not stored.
- **Logs:** each new run log is downloaded and stored in `live/logs/` with links and IP addresses removed. Since 942a451, `run.sh` puts the `[inference]` log and the forecast (`[run] forecast {...}`) in the run log, the only way to see the live forecast: the portal gives no download for it.
- **Run records:** each finished run whose log carries the forecast, or that failed, becomes a run record in the staging format, candidate `benchlab`, variant `live`. The daily evaluation scores it with the candidates (a failed run counts as persistence) and shows its health. Its runtime and pull time are Benchlab's own. Runs of versions before 942a451 log no forecast and are kept as logs only. A run off the 6-hourly marks (the qualification dry run after an upload) is recorded as variant `dryrun`, scored apart from the cycles, and ignored by the missed-cycle check. Some logs are served gzipped; they're unpacked before scrubbing, and a log that isn't text is not stored.
- **Alerts** (same reporting as staging): the staging rules on each record, plus not ELIGIBLE, missed cycle (a 6-hourly t0 with no run), no log 3 h after a run, no forecast in a successful run's log (info), and **differs from staging**: a different model, or a forecast more than 25 km/s from staging's run of `upload_candidate/submitted` at the same t0 (small differences are expected from data arriving between the two runs; the difference is in each record and as a `vs staging` info event). Staging's record of a t0 can come hours after the live one (a cycle run by the cron backup), so a live record without it is compared on a later pass; after 24 h without one it's marked `"staging": null` with a `no staging run` info event.
- **Secrets:** the portal's response holds presigned links and upload credentials. It stays in a file on the runner and is deleted; links are used in memory only and never printed or stored. `mode=shape` prints only the response's field names and types (`ci/monitor/shape.jq`).

## Limits

- **GitHub's cron** starts runs late, in October 2026 typically ~5 h and up to ~6 h, and sometimes drops one. The start delay is recorded. It's the backup for the on-time trigger, which needs the laptop on.
  - Each 6-hourly slot has its own cron line, and a cycle takes its t0 from the slot that fired, not from the clock. A late run therefore still runs its own t0, rather than repeating the previous one and leaving its own t0 missed.
  - A cycle counts as missed only 8 h after its t0, and the schedule as stale after 18 h without a cycle.
  - Inputs are filtered by availability at t0, so forecasts stay valid. But a run hours after t0 (a backup run) finds upstream data that was late at t0 already in place, so it understates the data-latency risk of Benchlab's live run. Compare start delays in the records.
  - GitHub disables scheduled workflows after 60 days without repository activity; re-enable them in the Actions tab.
- **The runner isn't Fargate.** CPU speed, network path and pull time differ, so runtimes are indicative. The pull is estimated as size / 100 MB/s, as `test_local.sh` does.
- **Unreleased spec details:** the persistence convention (labelled ≤ t0 − 1 h; see [official_score.md](official_score.md)) and the quantile interpolation are fixed constants in `training.competition_score`.
- **Small samples:** a few weeks of runs are needed before candidate differences of a few percent mean much.
- **Staging uses main's scripts.** A candidate's own `ci/` is not used, so an old candidate is still recorded and scored the current way.

## Code

- **Workflow and shared step:** `.github/workflows/staging.yml`, `.github/actions/base-image/action.yml`.
- **Scripts:** `ci/staging/run_candidate.sh`, `record_run.py`, `events.py`, `evaluate.py`, `records.sh`, `report_alerts.sh`.
- **Live monitor:** `.github/workflows/monitor.yml`, `ci/monitor/live.py`, `ci/monitor/shape.jq`; tests in `tests/test_monitor.py` (records, alerts, repeated passes, and no link, credential, ARN or IP on the branch).
- **Tests:** `tests/test_staging.py` covers log parsing, DNS and declared domains, record validity, event rules, and a whole cycle-to-report run on synthetic truth.
