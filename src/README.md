# Solar wind speed forecast — linear + gradient-boosted hybrid (team TINASHE)

A 72-hour hourly forecast of the solar wind speed at L1 (ACE SWEPAM), from the speed
history and 40 regressors: solar-wind plasma and field at L1, energetic particles,
coronal holes in SDO/AIA images, flares and F10.7.

## The submitted model

The forecast is a linear model with gradient-boosted trees on top
(`models/xgboost_model.py`, base `models/linear_model.py`). Both were fitted on
2016-10 → 2025-10, and all choices were made on six yearly walk-forward folds. On those
folds its per-run mean TIMSE is 0.42 of persistence's, against 0.48 for the speed-only
reference baseline. Choices and scores are in `docs/` in the repository
(`xgboost_model.md`, `linear_model.md`, `lead_aligned_ch.md`, `evaluation.md`).

**1. Linear anchor** (`models/linear_model.py`, `assets/linear_model.json`). For each lead
k = 1…72 h, a ridge regression of v(t0+k) − last on:
- the last observed hourly speed;
- the mean speed ±3 h around t0 + k − 27 d and t0 + k − 13.5 d (recurrence);
- four regressors:
  - last-hour Bt;
  - the coronal-hole area of the central-meridian strip in the SDO/AIA 193 Å image about 72 h before each forecast hour (lead k uses the image 72 − k h before t0; 13 images, 0–72 h);
  - the 72 h peak X-ray flare flux;
  - the 24 h mean density.

Rows are weighted by each run's TIMSE weights, so the loss is the competition metric.
Each regressor is clipped to its training [0.5%, 99.5%] range.

**2. Trees** (`assets/xgboost_model.json`). 100 trees of depth 4 (every leaf supported by
at least 43 training days) correct the linear forecast. Their inputs are the lead, the
speed terms, the linear forecast and 30 regressors: solar-wind plasma and field means,
extremes and slopes, ACE EPAM protons, 27-day recurrence of density, pressure and
polarity, coronal-hole strips at several lags, flares and F10.7. They are evaluated in
numpy, so the image has no xgboost.

**Robustness.** A run never fails for lack of data.
- **Missing values:** a missing regressor follows each tree's learned default branch, or takes its training median in the linear anchor. A missing coronal-hole image falls back to one 6 h, then 12 h older, then to the nearest of the other images within 24 h. If the strip is still missing, a linear submodel fitted without it is used.
- **Failures:** if the trees fail, the linear model forecasts; if it fails, the reference baseline below does; then persistence; then the training mean.
- **Other modes:** `SWP_MODEL=linear` or `SWP_MODEL=baseline` submits one of the fallbacks directly.

**Runtime.** numpy, pandas and Pillow only.
- **Inputs:** all sources are fetched in parallel within a 300 s budget: about 3 months of ACE speed, 30 days of ACE plasma, field and EPAM, about 15 AIA images, flares and F10.7, roughly 20 MB.
- **Timing:** a run at 2026-10-24 00Z took 14 s, with all 40 regressors observed.

## Reference baseline (fallback)

`models/final_baseline.py` uses the speed history alone. For each lead k:

    R_k = a_k + b_k·(last − μ) + Σ_j c_kj·(r_j(k) − μ)
    F_k = w_k·last + (1 − w_k)·R_k

- **Terms:** `last` is the last observed hourly speed and μ the training mean. `r_j(k)` is the mean speed within ±3 h of `t0 + k − lag_j` (27 d and 13.5 d).
- **Coefficients:** frozen coefficients from the training years are blended with coefficients refitted at every run on the last 60 days. The loss is the per-run TIMSE.
- **Persistence blend:** w_k is fitted out of sample.

Details are in `docs/final_baseline.md`.

## Inputs and data rules

| Source | Host (declared) | Used for |
|---|---|---|
| ACE SWEPAM/MAG/EPAM daily lists | `soho.nascom.nasa.gov` | speed history; plasma, field, particles |
| SWPC real-time feeds, flares, F10.7 | `services.swpc.noaa.gov` | latest minutes; flares; F10.7 |
| SDO/AIA 193 browse images | `sdo.gsfc.nasa.gov` | coronal holes (decoded with Pillow) |
| CDAWeb OMNI | `cdaweb.gsfc.nasa.gov` | declared for history connectors; not contacted by a run |

- **Only data published by `t0` is used:**
  - in-situ minutes before t0 − 10 min (speed minutes before t0);
  - AIA images taken before t0 − 1 h;
  - flares that peaked by t0 − 10 min;
  - F10.7 up to the previous day.
- **Speed:** averaged to hourly values exactly as in training, with no imputation.
- **Regressors:** the windowed statistics are built by `feature_engineering/windows.py`; `python3 -m inference.check_features <t0>` runs the fetch and build as a dry run.

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
├── requirements.txt     # runtime Python dependencies (numpy, pandas, Pillow)
├── model.py             # CLI and output format; calls inference.run_selected_forecast
├── connectors/          # data clients (ACE archive, SWPC, SDO/AIA, CDAWeb)
├── feature_engineering/ # hourly speeds, recurrence features, regressors, imputation
├── models/              # the hybrid (xgboost_model.py), its linear anchor, the baseline
├── inference/           # fetch -> features -> model, with fallbacks
├── assets/              # fitted weights (hybrid, linear model, baseline), regressor medians
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
