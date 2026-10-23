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
.venv/Scripts/python -m training.linear_model select|train|evaluate  # full selection (resumable, --max-seconds), fit -> src/assets/linear_model.json, eval year once
.venv/Scripts/python -m training.xgboost_model select --base linear --sample-frac 0.1  # timed pilot of the XGBoost hybrid's selection
.venv/Scripts/python -m training.xgboost_model select|tame|train|evaluate --base linear  # full selection (resumable; --sets, XGB_NTHREAD=2 caps threads/memory on this laptop), fit -> src/assets/xgboost_model.json
.venv/Scripts/python -m training.compare_history       # refits the shipped configs on history since 2016 vs 2021 over the yearly folds (training.folds)
.venv/Scripts/python -m training.regime_models         # regime-model feasibility study (oracle, classifiers, routing) on the yearly folds; not built, see src/experiments/regime_models.ipynb
.venv/Scripts/python -m training.final_baseline tune     # grid-searches the reference baseline on the training years only
.venv/Scripts/python -m training.final_baseline train    # fits the chosen config, writes src/assets/final_baseline.json
.venv/Scripts/python -m training.final_baseline evaluate # scores it on the eval year (once, after choosing)
```

**Evaluation** (`docs/evaluation.md`): score every model with `training.evaluate.evaluate(model)`, which computes TIMSE on the held-out final year. A forecast at `t0` may use, and refit on, only hours before `t0` (e.g. the last 30 days, `training.baselines.Rolling`). Choose hyper-parameters, features and models on the training years (`training.datasets.load_train()`), never by looking at eval scores.

`test_local.sh` takes `CPU`/`MEM` env vars. CI sets them from `submission.json`.

## Architecture

**`src/` is the submission root and the Docker build context.** Its contents become the ZIP root, and Benchlab requires `Dockerfile`, `submission.json` and `README.md` there. `src/README.md` must describe the actual submitted model.

**What ships and what doesn't.**
- The image gets only `model.py`, `run.sh`, `upload.sh`, `requirements.txt`, `assets/` and the packages `connectors`, `feature_engineering`, `models` and `inference`. These are explicit `COPY` lines in the EDIT section of `src/Dockerfile`, so a new runtime package needs a new `COPY` line there.
- `data/`, `experiments/`, `training/` and `local_tests/` are dev-only. They're excluded by `src/.dockerignore` and by `package_submission.sh`, and CI fails if any of them appear in `/app`.
- `src/data/` is gitignored. `src/assets/` (weights) is tracked.

**Runtime call chain:** `ENTRYPOINT /app/run.sh <t0>` → `python3 /app/model.py <t0> > /tmp/forecast.json` → `/app/upload.sh` POSTs to the presigned `$OUTPUT_POST`.
- `model.py` owns only the CLI and the output format (`t0`, `valid_from`, `forecast_speed_kms`). Forecasting logic belongs behind `inference.run_forecast(t0)`, which orchestrates connectors → feature_engineering → models: it pulls ~3 months of 1-minute ACE speed (`ACEArchiveConnector`, the training-data source, topped up from the SWPC real-time feed), averages it to hourly exactly as the training data, and runs `models.final_baseline.FinalBaseline` loaded from `assets/final_baseline.json`. Every step falls back (persistence, then the training mean) rather than failing.
- **Regressors** (`docs/features.md`) are not used by the forecast yet.
  - Pipeline: `inference.fetch_inputs` (all sources in parallel, 300 s budget) → `feature_engineering.build.build_table` → `feature_engineering.impute`.
  - Each source has an availability rule (in-situ t0 − 10 min, AIA t0 − 1 h, flares by peak time, F10.7 up to the previous day).
  - `tests/test_features.py` checks that scrambling everything not available at t0 leaves the features unchanged. Keep that test passing when adding a regressor.
- **Linear model** (`models.linear_model`, `docs/linear_model.md`): a per-lead GLM on the speed terms plus selected windowed regressors (`feature_engineering.windows`). It is opt-in live with `SWP_MODEL=linear` (`inference.run_linear_forecast`, falling back to the final baseline). Training drops isolated one-hour speed glitches (`recurrence.despike`) from targets and recurrence terms. The coronal-hole regressor falls back to older images, and if it is still missing, a submodel fitted without it is used (`Config.optional`). `model.py` calls `inference.run_selected_forecast`, which runs the final baseline by default.
- **XGBoost hybrid** (`models.xgboost_model`, `docs/xgboost_model.md`): gradient-boosted trees, pooled over leads, that correct the linear model's forecast. It is trained with `xgboost-cpu` (dev only) and evaluated in numpy from `assets/xgboost_model.json`, with no xgboost in the image. It is opt-in live with `SWP_MODEL=xgboost` (`inference.run_xgboost_forecast`, falling back to the linear model).
- **The reference baseline** is `models.final_baseline` (`docs/final_baseline.md`). Every new model is compared against it. `SWP_OFFLINE=1` makes `run_forecast` skip the network (used by `tests/test_model_contract.py`).
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
- **Uploading:** you upload to Benchlab manually. A replacement upload must re-qualify, and an earlier ELIGIBLE version is not kept as a fallback.
- **`STRICT_TEAM_INFO`:** setting it to `"1"` in `ci.yml` makes template team details a CI failure.

## Windows notes

This repo is developed on Windows.
- **Line endings:** `.gitattributes` forces LF everywhere. CRLF breaks `verify.sh` and the `.sh` scripts inside the container. Tools like `jq` on Windows write CRLF, so check `git diff` after generating files.
- **Bash:** use Git Bash. In PowerShell, `bash` resolves to a broken WSL install.
- **Docker paths:** in Git Bash, prefix `docker run` commands that take Linux paths with `MSYS_NO_PATHCONV=1`, e.g. `MSYS_NO_PATHCONV=1 docker run --rm --entrypoint ls swp-local /app`.
- **Docker Desktop** must be running for `test_local.sh`.
