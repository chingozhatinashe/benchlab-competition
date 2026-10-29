# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A submission for the 2026 Benchlab Solar Wind Prediction Tournament. Benchlab builds the Docker image from `src/` and runs a fresh container every 6 hours. The container receives an inference timestamp `t0` (e.g. `20261111T060000Z`, as argv[1] and `$INFERENCE_TIMESTAMP`) and must upload exactly 72 hourly solar-wind speeds (km/s, `t0+1h … t0+72h`) within 15 minutes, image pull included. `PARTICIPANT_CONTRACT.md` and `src/README.md` are the authoritative rules.

## Commands

The dev environment is a Python 3.10 venv (matching `python3` in the ubuntu:22.04 image), managed with uv:

```bash
uv venv --python 3.10 .venv && uv pip install --python .venv -r requirements-dev.txt
```

```bash
.venv/Scripts/python src/model.py 20261111T060000Z      # run the forecast, JSON on stdout
.venv/Scripts/python -m pytest                          # all tests (testpaths = tests/)
.venv/Scripts/python -m pytest tests/test_model_contract.py::test_forecast_contract -k 20261231
.venv/Scripts/ruff check src tests ci                   # ruff is pinned; rules in pyproject.toml
.venv/Scripts/python ci/check_submission.py             # static contract checks (same as CI)
(cd conformance_pack && bash verify.sh)                 # must print "OK — kit is unmodified"
(cd src/local_tests && ./test_local.sh 20261111T060000Z) # full Docker build + run + upload + validate
bash package_submission.sh                              # builds submission.zip from src/
.venv/Scripts/python -m training.pull_omni              # pulls 2 years of OMNI hourly data into src/data/omni_hourly/
.venv/Scripts/python -m training.pull_ace_rtsw          # pulls the pinned real-time ACE speed series (2016-10 -> 2026-10) into src/data/ace_rtsw/
.venv/Scripts/python -m training.pull_lasco_cme --start 2021-10  # pulls the SOHO/LASCO CME catalogue into src/data/lasco_cme/ (analysis only: published months late)
.venv/Scripts/python -m training.pull_features all       # regressor history (ACE plasma/field/EPAM, AIA images, GOES flares, F10.7) -> src/data/features/table.csv.gz + assets/feature_climatology.json; resumable
.venv/Scripts/python -m inference.check_features 20261021T180000Z  # live dry run of the regressors: sources, timing, MB, values
.venv/Scripts/python -m training.linear_model select --sample 500 --seed 0  # timed pilot of the linear model's variant search + forward selection; run before a full selection
.venv/Scripts/python -m training.linear_model select|train|evaluate  # full selection (resumable, --max-seconds; scored with the official metric, --objective run_timse for the old one; writes *_official* files), fit -> src/assets/linear_model.json, eval year once
.venv/Scripts/python -m training.xgboost_model select --base linear --sample-frac 0.1  # timed pilot of the XGBoost hybrid's selection
.venv/Scripts/python -m training.xgboost_model select|tame|train|evaluate --base linear  # full selection (resumable; --sets, XGB_NTHREAD=2 caps threads/memory on this laptop), fit -> src/assets/xgboost_model.json
.venv/Scripts/python -m training.compare_history       # refits the shipped configs on history since 2016 vs 2021 over the yearly folds (training.folds)
.venv/Scripts/python -m training.regime_models         # regime-model feasibility study (oracle, classifiers, routing) on the yearly folds; not built, see src/experiments/regime_models.ipynb
.venv/Scripts/python -m training.regime_hmm           # HMM regime features (models.regime_hmm) added to the shipped models on the yearly folds; resumable; no gain, see src/experiments/regime_hmm.ipynb
.venv/Scripts/python -m training.regime_switching      # standalone HMM-gated experts (models.regime_switching) on the yearly folds; resumable; not built, see docs/regime_switching.md
.venv/Scripts/python -m training.regressor_cap         # chooses the linear model's regressor cap (Config.cap_quantile) on the yearly folds; resumable; see src/experiments/regressor_cap.ipynb
.venv/Scripts/python -m training.quantile_mapping      # quantile-mapping post-step for the submitted linear model, official metric (--metric run_timse: the old study with the hybrid) on the yearly folds; resumable; not built, see docs/quantile_mapping.md
.venv/Scripts/python -m training.recency_weights       # recency sample weights for the linear model and hybrid on the yearly folds; resumable; not built, see docs/recency_weights.md
.venv/Scripts/python -m training.lead_aligned_ch       # lead-aligned coronal-hole features (image D h before each forecast hour) on the yearly folds; resumable; --hybrid "at 72"; see docs/lead_aligned_ch.md
.venv/Scripts/python -m training.coronal_grid images|folds  # coronal-hole latitude bands, B0 and darkness from the cached AIA images, lead-aligned, on the yearly folds; not built, see docs/coronal_grid.md
.venv/Scripts/python -m training.hybrid_forward_add    # adds a second window per regressor family to the hybrid on the yearly folds; resumable; none kept, see docs/hybrid_forward_add.md
.venv/Scripts/python -m training.pull_lasco_images      # LASCO C2/C3 browse-image history (2016-10 -> now) reduced to sector cells -> src/data/features/lasco_cells.csv; resumable, ~6 h
.venv/Scripts/python -m training.coronagraph_features  # LASCO brightening features in the hybrid and linear model on the yearly folds; resumable; not built, see docs/coronagraph.md
.venv/Scripts/python -m training.cme_detector          # two-stage CME detector (models.cme_detector) on the LASCO history, outputs added to the hybrid on the yearly folds; resumable; not built, see docs/cme_detector.md
.venv/Scripts/python -m training.pull_hmi_images      # SDO/HMI quick-look magnetogram history (2016-10 -> now) reduced to a 10-deg lat/lon grid -> src/data/features/hmi_grid/; resumable, ~1.3 h
.venv/Scripts/python -m training.magnetogram_features # HMI magnetogram features in the hybrid and linear model on the yearly folds; resumable; not built, see docs/magnetogram.md
.venv/Scripts/python -m training.gaussian_process_folds  # offset-GP correction of the submitted linear model, official metric (--metric run_timse: the old study with ml2 and the hybrid), on the yearly folds; resumable; not built, see docs/gaussian_process.md
.venv/Scripts/python -m training.pull_protons          # GOES >=10 MeV proton history (OMNI to 2020-03, NCEI SGPS from 2020-11) -> src/data/features/protons_hourly.csv; resumable, ~25 min; --check compares with SWPC's live feed
.venv/Scripts/python -m training.new_features          # new regressors (Bt/temperature 26-28 d earlier, F10.7 81-day level/trend, protons) added to the submitted linear model, official metric, on the yearly folds; --sample 500 for the pilot; resumable; none kept, see docs/new_features.md
.venv/Scripts/python -m training.pull_ace_1h            # official ground truth: SWPC hourly ACE files (2016-10 -> now) -> src/data/ace_1h/speed_hourly.csv (~40 s)
.venv/Scripts/python -m training.official_rescore       # shipped models + spec baselines scored with the official metric on the yearly folds; resumable; see docs/official_score.md
.venv/Scripts/python -m training.official_retune        # loss weights re-tuned for the official metric (linear, hybrid, final baseline) on the yearly folds; resumable; --eval-year scores the assets once; see docs/official_retune.md
.venv/Scripts/python -m training.final_baseline tune     # grid-searches the reference baseline on the training years only
.venv/Scripts/python -m training.final_baseline train    # fits the chosen config, writes src/assets/final_baseline.json
.venv/Scripts/python -m training.final_baseline evaluate # scores it on the eval year (once, after choosing)
```

**Official metric** (`docs/official_score.md`, organisers' spec `docs/timse_implementation.pdf`, released 2026-10-25, kept locally): `training.competition_score.score(forecasts, observations, persistence)`. Per lead, S_τ = √(RMSE_τ·√TIMSE_τ), with a 100-percentile, range-normalised TIMSE, averaged over the 72 leads, against SWPC's hourly ACE file (`training.pull_ace_1h`, status ≠ 0 missing). A missing or failed forecast is scored as persistence. Score model choices with it. The models were tuned on the old per-run TIMSE (`docs/timse.md`), which upward-biases them.

**Evaluation** (`docs/evaluation.md`): score every model with `training.evaluate.evaluate(model)`, which computes TIMSE on the held-out final year. A forecast at `t0` may use, and refit on, only hours before `t0` (e.g. the last 30 days, `training.baselines.Rolling`). Choose hyper-parameters, features and models on the training years (`training.datasets.load_train()`), never by looking at eval scores.

`test_local.sh` takes `CPU`/`MEM` env vars. CI sets them from `submission.json`.

## Architecture

**`src/` is the submission root and the Docker build context.** Its contents become the ZIP root, and Benchlab requires `Dockerfile`, `submission.json` and `README.md` there. `src/README.md` must describe the actual submitted model.

**What ships and what doesn't.**
- The image gets only `model.py`, `run.sh`, `upload.sh`, `requirements.txt`, `assets/` and the packages `connectors`, `feature_engineering`, `models` and `inference`. These are explicit `COPY` lines in the EDIT section of `src/Dockerfile`, so a new runtime package needs a new `COPY` line there.
- `data/`, `experiments/`, `training/` and `local_tests/` are dev-only. They're excluded by `src/.dockerignore` and by `package_submission.sh`, and CI fails if any of them appear in `/app`.
- `src/data/` is gitignored. `src/assets/` (weights) is tracked.

**Runtime call chain:** `ENTRYPOINT /app/run.sh <t0>` → `python3 /app/model.py <t0> > /tmp/forecast.json` → `/app/upload.sh` POSTs to the presigned `$OUTPUT_POST`.
- `model.py` owns only the CLI and the output format (`t0`, `valid_from`, `forecast_speed_kms`). It calls `inference.run_selected_forecast(t0)`, which runs the **linear model by default** (the submitted model since 2026-10-26, re-tuned for the official metric, `docs/official_retune.md`; the hybrid was the default 2026-10-24 → 26). `SWP_MODEL=baseline` selects the final baseline and `SWP_MODEL=hybrid` the hybrid; anything unexpected ends in the final baseline. The fallback chain is linear model → final baseline → persistence → training mean. `inference.run_forecast(t0)` is the final baseline's path; it orchestrates connectors → feature_engineering → models: it pulls ~3 months of 1-minute ACE speed (`ACEArchiveConnector`, the training-data source, topped up from the SWPC real-time feed), averages it to hourly exactly as the training data, and runs `models.final_baseline.FinalBaseline` loaded from `assets/final_baseline.json`. Every step falls back (persistence, then the training mean) rather than failing.
- **Regressors** (`docs/features.md`) are not used by the forecast yet.
  - Pipeline: `inference.fetch_inputs` (all sources in parallel, 300 s budget) → `feature_engineering.build.build_table` → `feature_engineering.impute`.
  - Each source has an availability rule (in-situ t0 − 10 min, AIA t0 − 1 h, flares by peak time, F10.7 up to the previous day).
  - `tests/test_features.py` checks that scrambling everything not available at t0 leaves the features unchanged. Keep that test passing when adding a regressor.
- **Linear model** (`models.linear_model`, `docs/linear_model.md`): a per-lead GLM on the speed terms plus selected windowed regressors (`feature_engineering.windows`). It is the submitted model (`inference.run_linear_forecast`, falling back to the final baseline). Its loss weights are `weights="official"`, `blend=0.25`: per lead, 75% plain squared error and 25% the official percentile-TIMSE weights (`models.final_baseline.official_weights`). Training drops isolated one-hour speed glitches (`recurrence.despike`) from targets and recurrence terms. The coronal-hole regressor is lead-aligned (`ch_cm_lead_72h`: lead k uses the image about 72 h before t0 + k; `Config.input_features` expands it into 13 lagged columns, so callers pass `input_features`, not `features`). It falls back to older images, and if it is still missing, a submodel fitted without it is used (`Config.optional`).
- **XGBoost hybrid** (`models.xgboost_model`, `docs/xgboost_model.md`): gradient-boosted trees, pooled over leads, that correct the linear model's forecast. It is trained with `xgboost-cpu` (dev only) and evaluated in numpy from `assets/xgboost_model.json`, with no xgboost in the image. Since 2026-10-26 it is not the default, because on the official metric it ties with the re-tuned linear model. `SWP_MODEL=hybrid` runs it (`inference.run_xgboost_forecast`, falling back to the linear model). Its asset keeps the old weights and embeds its own copy of the old linear anchor.
- **The reference baseline** is `models.final_baseline` (`docs/final_baseline.md`), now the last model fallback (`tail_power=0.5` since 2026-10-26). New models are compared against it and against the submitted linear model, with the official metric. `SWP_OFFLINE=1` makes `run_forecast` skip the network (used by `tests/test_model_contract.py`).
- The platform layer and the `ENTRYPOINT ["/app/run.sh"]` contract in the Dockerfile must stay intact.

**Imports.** The packages are top-level, with no `src.` prefix.
- In the container they resolve because `WORKDIR /app`.
- Locally they resolve through the editable install from the root `pyproject.toml` (`package-dir = src`), so notebooks and `python -m training.x` work from any directory.
- `pyproject.toml` and `requirements-dev.txt` are dev-only. Image dependencies come solely from `src/requirements.txt`.

**Competition rules the code must respect:**
- **No future data.** Every input for a forecast at `t0` must have been *available* at or before `t0`, even if it's fetched a few minutes later. Connectors must filter by availability time and handle latency, gaps and upstream outages.
- **Stateless runs.** Nothing carries over between runs. Bake weights into `assets/` rather than downloading them at runtime.
- **Declared domains.** Every external host contacted at runtime must be listed in `declared_domains` in `submission.json`.
- **Resource pairs.** `cpu_units`/`memory_mib` must be a valid Fargate pair; the table is in `src/README.md` and enforced by `ci/check_submission.py`.
- **CPU-only ML wheels.** torch must stay `+cpu` from the PyTorch CPU index, and TensorFlow must be `tensorflow-cpu`.
- **Base image.** Keep it on `public.ecr.aws/...`, which avoids Docker Hub rate limits on Benchlab's shared build.

**Upstream reference folders.** `submission_pack/` and `conformance_pack/` come from the upstream starter kit (`upstream` remote, Trillium-Technologies/benchlab-starter-kit). Don't edit them. `conformance_pack/` is checked byte-for-byte by `verify.sh` in CI.

## CI and workflow

`.github/workflows/ci.yml` runs on PRs into `main` and on pushes to `main`. `main` is branch-protected: the `checks` and `container` jobs must pass, branches must be up to date, and the rules apply to admins. Work on a branch and open a PR against `chingozhatinashe/benchlab-competition`, not upstream.
- **`checks` job:** conformance digest, `ci/check_submission.py`, ruff, pytest.
- **`container` job:** loads the base image from the Actions cache, refreshed weekly, because public.ecr.aws returns 429s to shared runners. It then runs `test_local.sh` at the latest 6-hourly `t0` with the declared CPU/memory and uploads `submission-<sha>` (containing `submission.zip`) as an artifact.
- **Uploading:** a manual trigger only. A replacement upload must re-qualify, and an earlier ELIGIBLE version is not kept as a fallback.
  - `.github/workflows/upload.yml` ("Upload to Benchlab", `workflow_dispatch` from `main`, confirm with `UPLOAD`):
    - `action=upload` sends the exact `submission.zip` that CI tested for that commit and waits for the qualification result. It is refused before pre-deployment opens (`PREDEPLOY_START`, 2026-11-01), because until then the portal treats every upload as a conformance attempt and a model zip fails it as "Test kit modified".
    - `action=conformance` re-uploads the unchanged `conformance_pack/`. It is refused once the submission is ELIGIBLE.
    - `action=status` only reads the portal status.
  - It needs the repository secret `BENCHLAB_PORTAL`, the portal link from the Benchlab email.
  - The repo is public, so the workflow never prints the portal link, the presigned upload form or the presigned build-log links.
- **Staging** (`docs/staging.md`, `.github/workflows/staging.yml`): every commit tagged `upload_candidate/<name>` (tagged by hand) is built and run every 6 h like Benchlab's live run. That means the submitted model and the `baseline` fallback, in fresh containers with the declared CPU/memory, a 900 s limit, a mock upload, and a DNS check against `declared_domains`. A daily job scores the candidates with the official metric (failed runs count as persistence) and flags missed cycles.
  - **Output:** records, events, truth and the report (README) live on the orphan branch `staging-records`. Alerts are error annotations plus a job-summary section, and they fail the run so GitHub emails you (issues are off on this repo).
  - **Scope:** staging only scores and records. It never picks or uploads; that stays with the user and the upload workflow.
  - **On-time trigger:** GitHub's cron runs hours late, so a Windows scheduled task on the laptop (`ci/staging/install_trigger.ps1`) dispatches each cycle at t0 + 5 min with its `t0`. The cron remains as the backup, and a t0 already recorded is skipped.
  - **Shared step:** `.github/actions/base-image` (the base-image cache) is shared by `ci.yml` and `staging.yml`.
- **`STRICT_TEAM_INFO`:** setting it to `"1"` in `ci.yml` makes template team details a CI failure.

## Windows notes

This repo is developed on Windows.
- **Line endings:** `.gitattributes` forces LF everywhere. CRLF breaks `verify.sh` and the `.sh` scripts inside the container. Tools like `jq` on Windows write CRLF, so check `git diff` after generating files.
- **Bash:** use Git Bash. In PowerShell, `bash` resolves to a broken WSL install.
- **Docker paths:** in Git Bash, prefix `docker run` commands that take Linux paths with `MSYS_NO_PATHCONV=1`, e.g. `MSYS_NO_PATHCONV=1 docker run --rm --entrypoint ls swp-local /app`.
- **Docker Desktop** must be running for `test_local.sh`.
