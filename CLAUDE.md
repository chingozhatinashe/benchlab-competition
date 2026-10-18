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
```

`test_local.sh` takes `CPU`/`MEM` env vars. CI sets them from `submission.json`.

## Architecture

**`src/` is the submission root and the Docker build context.** Its contents become the ZIP root, and Benchlab requires `Dockerfile`, `submission.json` and `README.md` there. `src/README.md` must describe the actual submitted model.

**What ships and what doesn't.**
- The image gets only `model.py`, `run.sh`, `upload.sh`, `requirements.txt`, `assets/` and the packages `connectors`, `feature_engineering`, `models` and `inference`. These are explicit `COPY` lines in the EDIT section of `src/Dockerfile`, so a new runtime package needs a new `COPY` line there.
- `data/`, `experiments/`, `training/` and `local_tests/` are dev-only. They're excluded by `src/.dockerignore` and by `package_submission.sh`, and CI fails if any of them appear in `/app`.
- `src/data/` is gitignored. `src/assets/` (weights) is tracked.

**Runtime call chain:** `ENTRYPOINT /app/run.sh <t0>` → `python3 /app/model.py <t0> > /tmp/forecast.json` → `/app/upload.sh` POSTs to the presigned `$OUTPUT_POST`.
- `model.py` owns only the CLI and the output format (`t0`, `valid_from`, `forecast_speed_kms`). Forecasting logic belongs behind `inference.run_forecast(t0)`. That function is currently a 420 km/s placeholder; it should orchestrate connectors → feature_engineering → models.
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
- **`container` job:** runs `test_local.sh` at the latest 6-hourly `t0` with the declared CPU/memory, then uploads `submission-<sha>` (containing `submission.zip`) as an artifact.
- **Uploading:** you upload to Benchlab manually. A replacement upload must re-qualify, and an earlier ELIGIBLE version is not kept as a fallback.
- **`STRICT_TEAM_INFO`:** setting it to `"1"` in `ci.yml` makes template team details a CI failure.

## Windows notes

This repo is developed on Windows.
- **Line endings:** `.gitattributes` forces LF everywhere. CRLF breaks `verify.sh` and the `.sh` scripts inside the container. Tools like `jq` on Windows write CRLF, so check `git diff` after generating files.
- **Bash:** use Git Bash. In PowerShell, `bash` resolves to a broken WSL install.
- **Docker paths:** in Git Bash, prefix `docker run` commands that take Linux paths with `MSYS_NO_PATHCONV=1`, e.g. `MSYS_NO_PATHCONV=1 docker run --rm --entrypoint ls swp-local /app`.
- **Docker Desktop** must be running for `test_local.sh`.
