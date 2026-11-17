"""The contract: the same t0 and inputs must give the same forecast (it may be re-run to
check). Two separate processes per model, with different hash seeds and thread counts, must
write byte-identical output (offline, so the inputs are the same by construction)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
T0 = "20261117T120000Z"


def forecast(model, run):
    env = {**os.environ, "SWP_OFFLINE": "1", "PYTHONHASHSEED": str(17 * run),
           "OMP_NUM_THREADS": str(run), "OPENBLAS_NUM_THREADS": str(run)}
    env.pop("SWP_MODEL", None)
    if model != "default":
        env["SWP_MODEL"] = model
    out = subprocess.run([sys.executable, str(ROOT / "src" / "model.py"), T0], env=env,
                         capture_output=True, check=True, timeout=600)
    return out.stdout


@pytest.mark.parametrize("model", ["default", "bayes"])
def test_repeat_runs_are_byte_identical(model):
    first, second = forecast(model, 1), forecast(model, 2)
    assert first and first == second
