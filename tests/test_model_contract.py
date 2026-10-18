"""Output contract for src/model.py, as run by run.sh inside the container."""

import json
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

MODEL = Path(__file__).resolve().parent.parent / "src" / "model.py"
MAX_BYTES = 1_048_576


def run_model(*args):
    return subprocess.run(
        [sys.executable, str(MODEL), *args],
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.mark.parametrize(
    "t0",
    [
        "20261111T060000Z",
        "20261130T180000Z",  # month rollover within the horizon
        "20261231T120000Z",  # year rollover within the horizon
        "20270228T000000Z",
    ],
)
def test_forecast_contract(t0):
    result = run_model(t0)
    assert result.returncode == 0, result.stderr

    assert len(result.stdout.encode()) <= MAX_BYTES
    out = json.loads(result.stdout)

    t0_dt = datetime.strptime(t0, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    assert out["t0"] == t0_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert out["valid_from"] == (t0_dt + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    speeds = out["forecast_speed_kms"]
    assert len(speeds) == 72
    for v in speeds:
        assert isinstance(v, float)
        assert math.isfinite(v)
        assert 0 <= v <= 10000


@pytest.mark.parametrize("args", [[], ["not-a-timestamp"], ["20261111T060000Z", "extra"]])
def test_bad_arguments_fail(args):
    assert run_model(*args).returncode != 0
