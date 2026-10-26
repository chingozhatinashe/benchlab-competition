"""The staging environment's scripts (ci/staging): run records, events, the daily report."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ci" / "staging"))
import evaluate as st_eval  # noqa: E402
import events  # noqa: E402
import record_run  # noqa: E402

LINEAR_LOG = """\
[run] t0=20261026T060000Z submission=sub_local
[fetch_inputs] inputs in 3.5 s, 18.3 MB: {'swepam': 'ok', 'mag': 'ok', 'sdo': 'timeout'}
[inference] minutes: 105392 from the archive, 10 added from SWPC real time
[inference] linear model: 15/16 regressors observed; imputed: ['bt_mean_1h']
[inference] linear model done in 6.5 s; forecast 333..375 km/s
[run] done
"""
HYBRID_FALLBACK_LOG = """\
[inference] xgboost: 40/40 regressors observed; missing: none
[inference] xgboost failed (ValueError('boom'))
[inference] falling back to the linear model
[inference] linear model: 16/16 regressors observed; imputed: none
[inference] linear model done in 6.5 s; forecast 333..375 km/s
"""
BASELINE_LOG = """\
[inference] t0 2026-10-26 06:00: 2159/2160 hours observed, latest 2026-10-26 05:00
[inference] done in 2.0 s; forecast 340..380 km/s
"""
PERSISTENCE_LOG = """\
[inference] model failed (RuntimeError('x')); falling back to persistence
[inference] using the fallback forecast
[inference] done in 0.1 s; forecast 400..400 km/s
"""
DNS = """\
12:00:01.1 veth1 In  IP 172.17.0.2.40000 > 8.8.8.8.53: 1234+ A? soho.nascom.nasa.gov. (38)
12:00:01.2 veth1 In  IP 172.17.0.2.40001 > 8.8.8.8.53: 1235+ AAAA? services.swpc.noaa.gov. (40)
12:00:01.3 veth1 In  IP 172.17.0.2.40002 > 8.8.8.8.53: 1236+ A? evil.example.com. (34)
"""


def test_log_parsing():
    assert record_run.model_used(LINEAR_LOG) == "linear"
    assert not record_run.fell_back(LINEAR_LOG)
    assert record_run.regressors(LINEAR_LOG) == (15, 16, ["bt_mean_1h"])
    srcs, mb, secs = record_run.sources(LINEAR_LOG)
    assert srcs["sdo"] == "timeout" and mb == 18.3 and secs == 3.5
    assert record_run.model_used(HYBRID_FALLBACK_LOG) == "linear"
    assert record_run.fell_back(HYBRID_FALLBACK_LOG)
    assert record_run.regressors(HYBRID_FALLBACK_LOG) == (40, 40, [])  # the top-level model
    assert record_run.model_used(BASELINE_LOG) == "baseline"
    assert record_run.model_used(PERSISTENCE_LOG) == "persistence"
    assert record_run.fell_back(PERSISTENCE_LOG)


def test_dns_and_declared_domains():
    hosts = record_run.dns_hosts(DNS)
    assert hosts == ["evil.example.com", "services.swpc.noaa.gov", "soho.nascom.nasa.gov"]
    declared = ["soho.nascom.nasa.gov", "swpc.noaa.gov"]  # a parent domain covers subdomains
    assert record_run.undeclared(hosts, declared) == ["evil.example.com"]


def write_run(tmp, t0, variant="default", log=LINEAR_LOG, forecast=None, exit_code=0,
              runtime=21.0, candidate="lin-v1", dns=DNS, extra=()):
    d = tmp / f"{candidate}_{variant}_{t0}"
    d.mkdir(parents=True)
    (d / "run.log").write_text(log, encoding="utf-8")
    (d / "dns.txt").write_text(dns, encoding="utf-8")
    (d / "submission.json").write_text(json.dumps(
        {"declared_domains": ["soho.nascom.nasa.gov", "services.swpc.noaa.gov",
                              "example.com"]}), encoding="utf-8")
    if forecast is not None:
        (d / "forecast.json").write_text(json.dumps({"forecast_speed_kms": forecast}),
                                         encoding="utf-8")
    out = d / "rec.json"
    record_run.main(["--t0", t0, "--candidate", candidate, "--sha", "abcdef1234567",
                     "--variant", variant, "--log", str(d / "run.log"),
                     "--forecast", str(d / "forecast.json"), "--exit-code", str(exit_code),
                     "--runtime", str(runtime), "--build-seconds", "70", "--image-mb", "500",
                     "--dns", str(d / "dns.txt"), "--submission", str(d / "submission.json"),
                     "--started", f"{t0[:4]}-{t0[4:6]}-{t0[6:8]}T{t0[9:11]}:09:00Z",
                     "--out", str(out), *extra])
    return json.loads(out.read_text(encoding="utf-8"))


def test_record_validity_and_timing(tmp_path):
    ok = write_run(tmp_path, "20261026T060000Z", forecast=[400.0] * 72)
    assert ok["valid"] and ok["start_delay_s"] == 540 and ok["est_pull_s"] == 5.0
    assert ok["undeclared_hosts"] == [] and ok["dns_captured"]  # example.com covers evil.
    short = write_run(tmp_path, "20261026T120000Z", forecast=[400.0] * 71)
    assert not short["valid"] and short["invalid_reason"] == "not 72 values"
    failed = write_run(tmp_path, "20261026T180000Z", forecast=[400.0] * 72, exit_code=1)
    assert not failed["valid"] and failed["forecast"] is None  # exit code 0 is required too
    timeout = write_run(tmp_path, "20261027T000000Z", exit_code=124, runtime=900)
    assert timeout["timed_out"] and timeout["over_budget"]
    build = write_run(tmp_path, "20261027T060000Z", extra=("--build-failed",))
    assert build["build_failed"] and not build["valid"]


def test_events_rules():
    base = {"valid": True, "exit_code": 0, "runtime_s": 20, "image_mb": 500,
            "regressors_observed": 16, "regressors_total": 16, "sources": {"swepam": "ok"},
            "undeclared_hosts": [], "dns_captured": True, "model_used": "linear"}
    kinds = lambda r: {k for k, _, _ in events.run_events({**base, **r})}  # noqa: E731
    assert kinds({}) == {"run ok"}
    assert {"failed", "fell back"} <= kinds({"exit_code": 1, "valid": False, "fell_back": True})
    assert "timeout" in kinds({"timed_out": True, "exit_code": 124, "valid": False})
    assert "invalid output" in kinds({"valid": False, "invalid_reason": "not 72 values"})
    assert "slow" in kinds({"runtime_s": 700})
    assert "over budget" in kinds({"over_budget": True})
    assert "few regressors" in kinds({"regressors_observed": 10})
    assert kinds({"regressors_observed": 15}) == {"missing regressors", "run ok"}
    assert "undeclared host" in kinds({"undeclared_hosts": ["x.org"]})
    assert "source not ok" in kinds({"sources": {"sdo": "timeout"}})
    assert kinds({"build_failed": True}) == {"build failed"}
    sev = {k: s for k, s, _ in events.run_events({**base, "undeclared_hosts": ["x.org"]})}
    assert sev["undeclared host"] == "alert"


def test_candidate_events():
    prev = [{"name": "a", "sha": "1"}, {"name": "b", "sha": "2"}]
    cur = [{"name": "b", "sha": "3"}, {"name": "c", "sha": "4"}]
    assert sorted(events.candidate_events(prev, cur)) == [
        ("a", "1", "candidate removed"), ("b", "3", "candidate moved"),
        ("c", "4", "candidate added")]


def monthly_file(speeds, start):
    """An official-format hourly file for `speeds` (NaN -> status 9) from `start`."""
    lines = [":Product: test", "#-----"]
    for i, v in enumerate(speeds):
        t = start + pd.Timedelta(hours=i)
        s, val = (9, -9999.9) if np.isnan(v) else (0, v)
        lines.append(f"{t:%Y %m %d  %H%M}   61314   {t.hour * 3600:5d}    {s}        "
                     f"3.0   {val:8.1f}     1.00e+05")
    return "\n".join(lines) + "\n"


def test_cycle_and_daily_report(tmp_path):
    records, runs = tmp_path / "records", tmp_path / "runs"
    rng = np.random.default_rng(0)
    start = pd.Timestamp("2026-10-01T00:00Z")
    speed = 400 + 50 * np.sin(np.arange(24 * 40) / 30) + rng.normal(0, 5, 24 * 40)
    speed[100] = np.nan
    (records / "truth").mkdir(parents=True)
    hours = pd.date_range(start, periods=len(speed), freq="h", tz="UTC")
    for month in ("2026-10", "2026-11"):
        sel = hours.strftime("%Y-%m") == month
        (records / "truth" / f"{month.replace('-', '')}_ace_swepam_1h.txt").write_text(
            monthly_file(speed[sel], hours[sel][0]), encoding="utf-8")
    cands = [{"name": "lin-v1", "sha": "abcdef1234567"}, {"name": "hyb-v1", "sha": "1234567"}]
    cycles = pd.date_range("2026-10-20T00:00Z", "2026-10-25T00:00Z", freq="6h")
    for t in cycles:
        if t == cycles[3]:
            continue  # a missed cycle
        t0 = f"{t:%Y%m%dT%H%M%SZ}"
        good = [float(v) for v in speed[(hours > t)][:72] + 3]
        for cand in cands:
            fails = cand["name"] == "hyb-v1" and t == cycles[5]
            write_run(runs, t0, candidate=cand["name"], forecast=None if fails else good,
                      exit_code=1 if fails else 0)
        alerts = events.cycle(records, runs, t0, cands)
        if t == cycles[5]:
            assert "hyb-v1" in alerts and "lin-v1" not in alerts
    assert len(list(records.glob("runs/2026-10/*/*.json"))) == 2 * (len(cycles) - 1)
    ev_csv = pd.read_csv(next(records.glob("events/*.csv")))
    assert {"candidate added", "missing regressors", "failed", "cycle"} <= set(ev_csv["kind"])
    assert "no record" not in set(ev_csv["kind"])

    now = pd.Timestamp("2026-10-25T06:00Z")  # the last cycle (00Z) is still on schedule
    st_eval.main(["--records", str(records), "--no-fetch", "--now", f"{now:%Y%m%dT%H%M%SZ}",
                  "--alerts", str(tmp_path / "alerts.json")])
    text = (records / "README.md").read_text(encoding="utf-8")
    assert "lin-v1" in text and "hyb-v1" in text and "Missed cycles: 1" in text
    assert "| failed |" in text and "missing regressors: 40" in text  # info only counted
    sc = json.loads(next(records.glob("scores/*.json")).read_text(encoding="utf-8"))
    lin, hyb = sc["all"]["lin-v1 | default"], sc["all"]["hyb-v1 | default"]
    complete = [t for t in cycles if t != cycles[3] and t + pd.Timedelta(hours=74) <= now]
    assert lin["runs"] == len(complete) and hyb["failed runs"] == 1
    # the failed run scores as persistence, so the hybrid's otherwise identical forecasts
    # score worse; persistence is the same on both candidates' t0s
    assert hyb["model"]["s_final"] > lin["model"]["s_final"]
    assert hyb["persistence"]["s_final"] == pytest.approx(lin["persistence"]["s_final"])
    alerts = json.loads((tmp_path / "alerts.json").read_text(encoding="utf-8"))
    assert list(alerts) == ["schedule"] and "missed cycle" in alerts["schedule"][0]
    # a second evaluation doesn't log the same missed cycle again
    st_eval.main(["--records", str(records), "--no-fetch", "--now", f"{now:%Y%m%dT%H%M%SZ}",
                  "--alerts", str(tmp_path / "alerts2.json")])
    assert json.loads((tmp_path / "alerts2.json").read_text(encoding="utf-8")) == {}
