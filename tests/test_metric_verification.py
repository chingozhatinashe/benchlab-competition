"""training.competition_score against the organisers' scorer (scoring_demo/swp_scoring)."""

import numpy as np
import pytest

from training import competition_score as cs
from training import metric_verification as mv

pytest.importorskip("tomli")
if not mv.DEMO.exists():
    pytest.skip("no scoring_demo (upstream starter kit)", allow_module_level=True)
mv.official()
from swp_scoring.core import metric  # noqa: E402

RNG = np.random.default_rng(3)


@pytest.mark.parametrize("step", [0.0, 0.1, 1.0, 10.0])
def test_score_matches_the_organisers_scorer(step):
    obs = 300 + 500 * RNG.beta(2, 5, (400, cs.HORIZON))
    if step:
        obs = np.round(obs / step) * step
    pred = obs + RNG.normal(0, 60, obs.shape)
    obs[RNG.random(obs.shape) < 0.1] = np.nan
    ours = cs.score(pred, obs, pred[:, 0])
    theirs = metric.score(obs, pred)
    assert ours["s_final"] == pytest.approx(theirs.s_final, rel=1e-12)
    np.testing.assert_allclose(ours["curves"]["s"], theirs.s_curve, rtol=1e-12)


def test_constant_lead_matches_the_organisers_scorer():
    obs = 300 + 500 * RNG.beta(2, 5, (30, cs.HORIZON))
    obs[:, 0] = 400.0
    pred = obs + RNG.normal(0, 60, obs.shape)
    assert cs.score(pred, obs, pred[:, 0])["s_final"] == pytest.approx(
        metric.score(obs, pred).s_final, rel=1e-12)


def test_the_demo_reproduces_the_expected_report():
    demo, _, _ = mv.check_demo(mv.official())
    for name in ("climatology", "persistence", "recurrence_27d"):
        assert demo[name]["rerun reproduces expected"]
        assert demo[name]["ours"] == pytest.approx(demo[name]["expected"], abs=1e-6)
    assert demo["obs matrices identical"]
