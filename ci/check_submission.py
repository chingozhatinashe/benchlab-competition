#!/usr/bin/env python3
"""Static checks that src/ meets the Benchlab submission contract.

Rules come from PARTICIPANT_CONTRACT.md and src/README.md. Run from anywhere:

    python ci/check_submission.py

Exits non-zero on any error. Warnings are printed as GitHub annotations.
Set STRICT_TEAM_INFO=1 to turn placeholder team details into an error.
"""

import json
import os
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"

REQUIRED_AT_ROOT = ["Dockerfile", "submission.json", "README.md"]
REQUIRED_PLUMBING = ["run.sh", "upload.sh"]

# cpu_units -> (min memory_mib, max memory_mib, step)
FARGATE = {
    1024: (2048, 8192, 1024),
    2048: (4096, 16384, 1024),
    4096: (8192, 30720, 1024),
    8192: (16384, 32768, 4096),
}

PLACEHOLDER_TEAM = {"Team Corona"}
PLACEHOLDER_EMAIL = re.compile(r"@example\.(edu|com|org|invalid)$")
HOSTNAME = re.compile(r"^(?!-)[a-z0-9-]{1,63}(?<!-)(\.(?!-)[a-z0-9-]{1,63}(?<!-))+$")

errors = []
warnings = []


def error(msg):
    errors.append(msg)


def warn(msg):
    warnings.append(msg)


def check_files():
    for name in REQUIRED_AT_ROOT + REQUIRED_PLUMBING:
        if not (SRC / name).is_file():
            error(f"src/{name} is missing")


def check_submission_json():
    path = SRC / "submission.json"
    if not path.is_file():
        return
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        error(f"submission.json is not valid JSON: {e}")
        return

    keys = ["team_name", "contact_email", "cpu_units", "memory_mib", "declared_domains"]
    missing = [k for k in keys if k not in cfg]
    for key in missing:
        error(f"submission.json is missing '{key}'")
    if missing:
        return

    cpu, mem = cfg["cpu_units"], cfg["memory_mib"]
    if cpu not in FARGATE:
        error(f"cpu_units={cpu} is not one of {sorted(FARGATE)}")
    else:
        lo, hi, step = FARGATE[cpu]
        if not (isinstance(mem, int) and lo <= mem <= hi and (mem - lo) % step == 0):
            error(f"memory_mib={mem} is not valid for cpu_units={cpu} "
                  f"(allowed {lo}-{hi} in steps of {step})")

    domains = cfg["declared_domains"]
    if not isinstance(domains, list) or not domains:
        error("declared_domains must be a non-empty list")
    else:
        for d in domains:
            if not isinstance(d, str) or not HOSTNAME.match(d):
                error(f"declared_domains entry {d!r} is not a hostname")

    placeholders = []
    if cfg["team_name"] in PLACEHOLDER_TEAM:
        placeholders.append(f"team_name={cfg['team_name']!r}")
    if PLACEHOLDER_EMAIL.search(str(cfg["contact_email"])):
        placeholders.append(f"contact_email={cfg['contact_email']!r}")
    if placeholders:
        msg = "submission.json still has template values: " + ", ".join(placeholders)
        if os.environ.get("STRICT_TEAM_INFO") == "1":
            error(msg)
        else:
            warn(msg)


def check_dockerfile():
    path = SRC / "Dockerfile"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    if not re.search(r'^ENTRYPOINT\s+\["/app/run\.sh"\]', text, re.MULTILINE):
        error('Dockerfile must keep ENTRYPOINT ["/app/run.sh"]')
    if not re.search(r"^COPY\s+upload\.sh\s+/app/upload\.sh", text, re.MULTILINE):
        error("Dockerfile platform layer must COPY upload.sh /app/upload.sh")
    for image in re.findall(r"^FROM\s+(?:--platform=\S+\s+)?(\S+)", text, re.MULTILINE):
        if not image.startswith("public.ecr.aws/"):
            warn(f"Dockerfile FROM {image} pulls from Docker Hub and may hit rate limits "
                 f"on the shared build; prefer a public.ecr.aws/ mirror")


def check_requirements():
    path = SRC / "requirements.txt"
    if not path.is_file():
        return
    lines = [ln.split("#", 1)[0].strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    has_cpu_index = any("download.pytorch.org/whl/cpu" in ln for ln in lines)
    for ln in lines:
        name = re.split(r"[<>=!~\[ ;]", ln, maxsplit=1)[0].lower()
        if name == "torch" and ("+cpu" not in ln or not has_cpu_index):
            error(f"requirements.txt '{ln}': use the CPU-only torch wheel "
                  f"(+cpu with --extra-index-url https://download.pytorch.org/whl/cpu)")
        if name in ("tensorflow", "tensorflow-gpu"):
            error(f"requirements.txt '{ln}': use tensorflow-cpu")


def check_line_endings():
    for path in [SRC / "Dockerfile", *SRC.rglob("*.sh")]:
        if path.is_file() and b"\r\n" in path.read_bytes():
            error(f"{path.relative_to(SRC.parent)} has CRLF line endings")


def main():
    check_files()
    check_submission_json()
    check_dockerfile()
    check_requirements()
    check_line_endings()

    for msg in warnings:
        print(f"::warning::{msg}")
    for msg in errors:
        print(f"::error::{msg}")
    if errors:
        print(f"FAIL — {len(errors)} error(s)")
        return 1
    print(f"OK — submission contract checks passed ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
