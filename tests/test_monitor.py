"""The Benchlab live monitor (ci/monitor/live.py): records, alerts, and no leaked links."""

import csv
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ci" / "monitor"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ci" / "staging"))
import evaluate as st_eval  # noqa: E402
import live  # noqa: E402

SUB = "sub_test"
FORECAST = [400.0 + i for i in range(72)]
SECRET = "https://bucket.s3.amazonaws.com/logs/x?X-Amz-Signature=deadbeef"


def run_log(t0, forecast=True, sub=SUB):
    body = json.dumps({"t0": t0, "forecast_speed_kms": FORECAST})
    return "\n".join([
        f"[run] t0={t0} submission={sub}",
        "[connectors] fetch https://services.swpc.noaa.gov/x failed from 10.0.3.7",
        "[fetch_inputs] inputs in 38.7 s, 19.7 MB: {'swepam': 'ok', 'pfss': 'ok'}",
        "[inference] bayes model: 42/42 regressors observed; imputed: none",
        "[inference] bayes model done in 38.9 s; forecast 400..471 km/s",
        *([f"[run] forecast {body}"] if forecast else []),
        "[run] produced 1486 bytes", "[upload] ok", "[run] done", ""])


def run(t0, stopped="2026-11-19T18:01:00Z", exit_code="0", output=1486, failure=None):
    return {"submission_id": SUB, "cycle_ts": t0, "status": "SUCCEEDED" if exit_code == "0"
            else "FAILED", "exit_code": exit_code, "failure_reason": failure,
            "output_bytes": str(output), "pull_seconds": "4", "run_seconds": "45",
            "started_at": t0, "stopped_at": stopped,
            "task_arn": "arn:aws:ecs:secret", "private_ip": "10.1.2.3", "eni_id": "eni-1",
            "stop_code": "EssentialContainerExited", "stopped_reason": "", "launch_retries": "0",
            "relaunched_after": None, "output_key": "secret/key"}


def portal(runs, logs):
    return {"submission_id": SUB, "eligibility": "ELIGIBLE", "conformance": "PASSED",
            "failure_reason": None, "phase": "live", "accepting": "false",
            "recent_runs": runs,
            "logs": [{"key": f"logs/{k}", "size": len(v), "url": f"{SECRET}&k={k}"}
                     for k, v in logs.items()],
            "upload": {"url": SECRET, "fields": {"AWSAccessKeyId": "AKIASECRET",
                                                 "signature": "SIGSECRET"}}}


def staging_record(forecast, t0="20261119T000000Z"):
    return {"schema": 1, "t0": t0, "candidate": "submitted", "sha": "942a451",
            "variant": "default", "valid": True, "forecast": forecast, "model_used": "bayes",
            "fell_back": False, "runtime_s": 40.0, "start_delay_s": 300,
            "regressors_observed": 42, "regressors_total": 42, "sources": {},
            "undeclared_hosts": []}


def events(records):
    rows = []
    for p in sorted((records / "events").glob("*.csv")):
        with p.open(encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    return rows


def test_monitor_records_alerts_and_never_leaks(tmp_path):
    records = tmp_path / "records"
    # staging's run of the submitted commit at the first t0, 2 km/s away
    st = records / "runs" / "2026-11" / "20261119T000000Z" / "submitted__default.json"
    st.parent.mkdir(parents=True)
    st.write_text(json.dumps(staging_record([v + 2 for v in FORECAST])), encoding="utf-8")
    logs = {"a": run_log("20261119T000000Z"),
            "c": run_log("20261119T120000Z", forecast=False)}
    runs = [run("20261119T000000Z"),                            # ok, compared with staging
            run("20261119T060000Z", exit_code="1", output=0),    # failed, no log
            run("20261119T120000Z"),                             # old version: no forecast line
            run("20261120T000000Z", stopped="2026-11-20T00:01:00Z")]  # log not listed yet
    # (no run at 20261119T180000Z: a missed cycle)
    fetched = []

    def fetch(url):
        fetched.append(url)
        return logs[url.rsplit("k=", 1)[1]].encode()

    now = datetime(2026, 11, 20, 1, 0, tzinfo=timezone.utc)
    alerts, summary = live.monitor(portal(runs, logs), records, now=now, fetch=fetch)
    assert len(fetched) == 2

    rec = json.loads((records / "runs/2026-11/20261119T000000Z/benchlab__live.json")
                     .read_text(encoding="utf-8"))
    assert rec["valid"] and rec["forecast"] == FORECAST and rec["model_used"] == "bayes"
    assert rec["regressors_observed"] == 42 and rec["est_pull_s"] == 4.0
    assert rec["staging"] == {"model_used": "bayes", "max_abs_diff_kms": 2.0}
    bad = json.loads((records / "runs/2026-11/20261119T060000Z/benchlab__live.json")
                     .read_text(encoding="utf-8"))
    assert not bad["valid"] and bad["exit_code"] == 1  # scored as persistence
    assert not (records / "runs/2026-11/20261119T120000Z/benchlab__live.json").exists()
    assert not (records / "runs/2026-11/20261120T000000Z/benchlab__live.json").exists()

    kinds = {(e["t0"], e["kind"]) for e in events(records)}
    assert ("20261119T000000Z", "vs staging") in kinds
    assert ("20261119T060000Z", "failed") in kinds
    assert ("20261119T120000Z", "no forecast in log") in kinds
    assert ("20261119T180000Z", "missed cycle") in kinds
    text = json.dumps(alerts)
    assert "failed" in text and "missed cycle" in text and "no forecast" not in text

    # nothing secret on the records branch: links, credentials, ARNs, IPs, object keys
    for p in records.rglob("*"):
        if p.is_file():
            body = p.read_text(encoding="utf-8")
            for leak in ("https://", "X-Amz", "AKIA", "SIGSECRET", "arn:aws", "10.0.3.7",
                         "10.1.2.3", "eni-1", "secret/key", "logs/a"):
                assert leak not in body, (p, leak)
    log_files = list((records / "live" / "logs").rglob("*.log"))
    assert len(log_files) == 2 and "<link>" in log_files[0].read_text(encoding="utf-8")

    # the daily staging evaluation reads the live record like a candidate's
    df = st_eval.load_runs(records)
    assert set(df["candidate"]) == {"benchlab", "submitted"}
    assert st_eval.health(df, None)[("benchlab", "live")]["failed"] == 1

    # a later pass: nothing downloaded again, no repeated alerts, the missing log now alerts
    later = datetime(2026, 11, 20, 4, 0, tzinfo=timezone.utc)
    alerts2, _ = live.monitor(portal(runs, logs), records, now=later, fetch=fetch)
    assert len(fetched) == 2
    lines = [x for v in alerts2.values() for x in v]
    assert any("20261120T000000Z" in x and "no log" in x for x in lines)
    assert not any("20261119T060000Z" in x or "20261119T180000Z" in x for x in lines)


def test_differs_from_staging_alerts(tmp_path):
    records = tmp_path / "records"
    st = records / "runs" / "2026-11" / "20261119T000000Z" / "submitted__default.json"
    st.parent.mkdir(parents=True)
    st.write_text(json.dumps(staging_record([v + 60 for v in FORECAST])), encoding="utf-8")
    logs = {"a": run_log("20261119T000000Z")}
    alerts, _ = live.monitor(portal([run("20261119T000000Z")], logs), records,
                             now=datetime(2026, 11, 19, 3, tzinfo=timezone.utc),
                             fetch=lambda url: logs["a"].encode())
    assert "differs from staging" in json.dumps(alerts)


def test_not_eligible_alerts_once(tmp_path):
    p = portal([], {})
    p["eligibility"], p["failure_reason"] = "FAILED", "build failed"
    now = datetime(2026, 11, 19, 3, tzinfo=timezone.utc)
    alerts, _ = live.monitor(p, tmp_path, now=now, fetch=None)
    assert "build failed" in json.dumps(alerts)
    alerts, _ = live.monitor(p, tmp_path, now=now, fetch=None)
    assert alerts == {}


def test_scrub_and_times():
    s = live.scrub("get https://a.b/c?X-Amz-Signature=1 from 192.168.0.1 sig X-Amz-Credential=AK")
    assert "https" not in s and "192.168" not in s and "AK" not in s
    for v in ("20261119T060000Z", "2026-11-19T06:00:00Z", "2026-11-19T06:00:00+00:00",
              "2026-11-19 06:00:00", "1795068000", "1795068000000"):
        assert live.to_t0(v) == "20261119T060000Z", v
    assert live.to_t0("") is None and live.to_t0("soon") is None


def test_unreadable_cycle_times_alert(tmp_path):
    r = run("20261119T000000Z")
    r["cycle_ts"] = "cycle seven"
    alerts, _ = live.monitor(portal([r], {}), tmp_path,
                             now=datetime(2026, 11, 19, 3, tzinfo=timezone.utc), fetch=None)
    assert "unreadable runs" in json.dumps(alerts)


def test_gzip_logs_dry_runs_and_binary(tmp_path):
    """Benchlab serves some logs gzipped; the run after an upload is off the 6-hourly marks."""
    logs = {"z": gzip.compress(run_log("20261119T150000Z").encode()),
            "b": bytes(range(256)) * 40}
    p = portal([run("20261119T150000Z")], {k: "x" for k in logs})
    alerts, _ = live.monitor(p, tmp_path, now=datetime(2026, 11, 19, 17, tzinfo=timezone.utc),
                             fetch=lambda url: logs[url.rsplit("k=", 1)[1]])
    rec = json.loads((tmp_path / "runs/2026-11/20261119T150000Z/benchlab__dryrun.json")
                     .read_text(encoding="utf-8"))
    assert rec["valid"] and rec["variant"] == "dryrun" and rec["model_used"] == "bayes"
    stored = list((tmp_path / "live" / "logs").rglob("*.log"))
    assert len(stored) == 1 and "<link>" in stored[0].read_text(encoding="utf-8")
    state = json.loads((tmp_path / "live" / "state.json").read_text(encoding="utf-8"))
    assert sum("error" in v for v in state["logs"].values()) == 1  # the binary one
    assert "missed cycle" not in json.dumps(alerts)  # a dry run alone defines no cycles


def test_scrub_build_log_aws_details():
    """Benchlab's build logs name the organisers' registry and account (CodeBuild)."""
    acct = "1234" * 3
    log = (f"The push refers to repository [{acct}.dkr.ecr.us-east-1.amazonaws.com/swp-prod]\n"
           f"#10 naming to {acct}.dkr.ecr.us-east-1.amazonaws.com/swp-prod:sub_x done\n"
           f"role arn:aws:iam::{acct}:role/build key AKIAABCDEFGHIJKLMNOP account {acct}\n"
           "[inference] minutes: 36213 from the archive; t0=20261119T150000Z 1486 bytes")
    s = live.scrub(log)
    for leak in (acct, "amazonaws", "arn:aws", "AKIA"):
        assert leak not in s, leak
    assert "36213" in s and "20261119T150000Z" in s and "1486" in s  # model lines intact
