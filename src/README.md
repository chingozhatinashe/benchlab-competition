# Solar wind speed forecast — reference baseline (team TINASHE)

A 72-hour hourly forecast of solar wind speed at L1 from the speed history alone.
This is the team's reference baseline: every later model is compared against it.

## The submitted model

**Input.** The real-time ACE SWEPAM 1-minute proton speed that NOAA SWPC
publishes, read from the daily lists archived at `soho.nascom.nasa.gov` (the
same source and processing as the training data) and topped up with the latest
minutes from the SWPC real-time feed (`services.swpc.noaa.gov`). Only minutes
stamped before `t0` are used. They are averaged to hourly speeds, with no
imputation.

**Model** (`models/final_baseline.py`). For each lead k = 1…72 h:

    R_k = a_k + b_k·(last − μ) + Σ_j c_kj·(r_j(k) − μ)
    F_k = w_k·last + (1 − w_k)·R_k

- `last` is the last observed hourly speed before `t0`, and μ the training mean.
- `r_j(k)` is the mean speed within ±3 h of `t0 + k − lag_j`. The lags are one
  solar rotation (27 d) plus sub-rotation lags, for several streams per rotation.
- **Frozen + rolling coefficients.** (a, b, c) blend coefficients fitted once on the
  training years (baked into `assets/final_baseline.json`) with coefficients
  refitted at every run on a recent window of history.
- **Tail-weighted loss.** Both fits weight each hour's squared error by its share of
  the run's TIMSE, so the loss is the competition metric.
- **Persistence blend.** w_k ∈ [0, 1] blends in persistence; it is fitted out of
  sample on the last training year.

The hyper-parameters (lags, window, rolling weight, loss strength) were chosen on
the training years only: lags of 27 d and 13.5 d, a 60-day rolling window with
weight 0.25, and the full per-run TIMSE loss. Weights, choices and scores are in
`docs/final_baseline.md` in the repository.

**Robustness.** A run never fails for lack of data. If the archive or the feed is
down, the model works with whatever history arrives (missing inputs become μ).
With no data at all it forecasts the training mean, and if the model itself fails
it falls back to persistence.

**Runtime.** numpy and pandas only. A run downloads about 3 months of daily files
(a few MB, typically under a minute) and computes the forecast in about a second.

## The contract

| | |
|---|---|
| **Base** | Linux. Anything you can `docker build`. Python is an example, not a requirement. |
| **Architecture** | `linux/amd64`. The build forces this. |
| **Entrypoint** | The platform runs `<your ENTRYPOINT> <timestamp>`. The timestamp is **argv[1]**. |
| **Timestamp** | e.g. `20261111T060000Z` — the cycle time, `t0`. Also available as `$INFERENCE_TIMESTAMP`. |
| **Forecast** | 72 hourly values, km/s, for `t0+1h … t0+72h`. |
| **Output** | JSON, POSTed to the one-time URL in `$OUTPUT_POST`. Max 1 MiB. |
| **Time limit** | 15 minutes wall clock from launch, **including image pull**. |
| **Resources** | As declared in `submission.json`, up to 8 vCPU / 32 GB / 20 GB usable disk. |
| **Network** | Any protocol, any public source. 20 GB per run fair use. |
| **State** | None. Every run starts fresh from the image. |
| **Success** | Exit code 0 **and** a valid forecast at the upload URL. Either alone is a failure. |

Every one of these is enforced by the platform's infrastructure, not by code
in this template.  Nothing you edit here can escape any of them, which is exactly why the template is otherwise yours to gut and rebuild.

## Layout

```
submission/
├── Dockerfile           # edit the marked region
├── run.sh               # edit the marked line
├── upload.sh            # provided — or reimplement in your language
├── submission.json      # required
├── README.md            # required — this file
├── requirements.txt     # runtime Python dependencies (numpy, pandas)
├── model.py             # CLI and output format; calls inference.run_forecast
├── connectors/          # data clients (ACE archive, SWPC real time, CDAWeb)
├── feature_engineering/ # minutes -> hourly speeds, causal features
├── models/              # the forecast model (models/final_baseline.py)
├── inference/           # fetch -> features -> model, with fallbacks
├── assets/              # fitted weights (final_baseline.json)
└── local_tests/         # run test_local.sh from here before you submit
    ├── test_local.sh
    └── mock_platform.py
```

Only `Dockerfile`, `submission.json`, and `README.md` are structurally
required — the platform's upload check looks for exactly those three.
Everything else is here to make the contract easy to meet:

- **`Dockerfile`** — builds your image. The top and bottom (image pull helper,
  user setup, entrypoint) are load-bearing; the middle is yours to replace
  with any language or runtime.
- **`run.sh`** — the entrypoint. One line to edit: whatever command produces
  your forecast. Everything around it (argument handling, upload call) is
  plumbing you shouldn't need to touch.
- **`upload.sh`** — sends your forecast to the platform. Reads `$OUTPUT_POST`
  (a presigned, one-time S3 upload URL the platform injects per run) and
  POSTs your output file to it. Reimplement it in your own language if you
  like; the 1 MiB cap and one-key scope are enforced by S3 regardless of
  what does the POSTing.
- **`submission.json`** — declares your team and your resource request (see
  below). Read once, at build time.
- **`requirements.txt`** — the image's Python dependencies.
- **`model.py`** — parses `t0`, calls `inference.run_forecast(t0)` and writes
  the output JSON. The model itself is described at the top of this file.
- **`local_tests/`** — not required at submission time, but the fastest way
  to catch a packaging mistake before it costs you a build cycle.
  `test_local.sh` builds and runs your image exactly as the platform will;
  `mock_platform.py` stands in for the real presigned-upload endpoint so the
  upload path actually gets exercised, not skipped. Run it before every
  submission:
  ```
  cd local_tests && bash test_local.sh
  ```

## `submission.json`

```json
{
  "team_name": "Team Corona",
  "contact_email": "lead@example.edu",
  "cpu_units": 4096,
  "memory_mib": 16384,
  "declared_domains": ["services.swpc.noaa.gov", "cdaweb.gsfc.nasa.gov"]
}
```

`cpu_units` / `memory_mib` must be one of the valid Fargate combinations below. Unsupported combinations are rounded up to the next supported Fargate configuration. 

| `cpu_units` | vCPU | `memory_mib` range | Step |
|---|---|---|---|
| 1024 | 1 | 2048 – 8192 | 1024 |
| 2048 | 2 | 4096 – 16384 | 1024 |
| 4096 | 4 | 8192 – 30720 | 1024 |
| 8192 | 8 | 16384 – 32768 | 4096 |

Note the 4-vCPU tier tops out at **30720 MiB (30 GB), not 32 GB** — "4 cores
and 32 GB" isn't a real Fargate shape. Requesting it gets you promoted to
8 vCPU automatically. Your portal page always shows what you were actually
granted.

## Output schema

The platform only reads `forecast_speed_kms`. Everything else in the object
is yours to use or ignore.

```json
{
  "t0": "2026-11-11T06:00:00Z",
  "valid_from": "2026-11-11T07:00:00Z",
  "forecast_speed_kms": [418.2, 419.9, "... 72 values total ..."]
}
```

## Two mistakes worth avoiding up front

**A plain `FROM ubuntu:22.04` or `FROM python:3.11-slim` will eventually hit
a Docker Hub rate limit** during the build (shared build infrastructure,
shared outbound IPs). Use the ECR Public Gallery mirrors instead —
`public.ecr.aws/ubuntu/ubuntu:22.04`, `public.ecr.aws/docker/library/python:3.11-slim` —
already what this template uses. Browse [gallery.ecr.aws](https://gallery.ecr.aws)
for others.

**If you use PyTorch, keep the CPU wheel.** Deleting `--extra-index-url
https://download.pytorch.org/whl/cpu` and `+cpu` from `requirements.txt`
pulls the CUDA build — about 2.5 GB of NVIDIA libraries onto hardware with
no GPU. That's 2.3 GB you store in the registry for six months and re-pull
inside your 15-minute budget on every run. `tensorflow-cpu` has the same
trap and the same fix.
