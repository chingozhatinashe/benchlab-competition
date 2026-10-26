#!/usr/bin/env bash
# One staging cycle for one candidate: build its image and run it as Benchlab does.
#
#   bash ci/staging/run_candidate.sh CANDIDATE_DIR T0 NAME SHA OUT_DIR [VARIANT ...]
#
# CANDIDATE_DIR is a checkout of the candidate's commit; its src/ is built exactly as the
# platform builds it. Each variant runs in a fresh container, as local_tests/test_local.sh
# runs it: argv t0, INFERENCE_TIMESTAMP, SUBMISSION_ID, OUTPUT_POST pointing at the mock
# upload endpoint (src/local_tests/mock_platform.py of main), the CPU and memory declared
# in the candidate's submission.json, and a 900 s limit. "default" sets no SWP_MODEL (the
# candidate's submitted model); other variants set SWP_MODEL=<variant> (e.g. baseline, the
# fallback). The container's DNS queries are captured with tcpdump to check the declared
# domains. One JSON record per variant goes to OUT_DIR (ci/staging/record_run.py).
set -uo pipefail

CAND_DIR=${1:?candidate dir}
T0=${2:?t0}
NAME=${3:?name}
SHA=${4:?sha}
OUT=${5:?out dir}
shift 5
VARIANTS=("$@")
[ ${#VARIANTS[@]} -eq 0 ] && VARIANTS=(default baseline)

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MAIN_SRC=$(cd "$HERE/../../src" && pwd)
SUB="$CAND_DIR/src/submission.json"
STARTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
mkdir -p "$OUT/raw"
OUT=$(cd "$OUT" && pwd)
RAW="$OUT/raw"  # logs, forecasts and DNS captures; the records themselves go to OUT

record() {  # record VARIANT [record_run.py args...]
  local v=$1
  shift
  python3 "$HERE/record_run.py" --t0 "$T0" --candidate "$NAME" --sha "$SHA" --variant "$v" \
    --submission "$SUB" --started "$STARTED" --out "$OUT/${NAME}__${v}.json" "$@"
}

CPU=$(( $(jq .cpu_units "$SUB") / 1024 ))
MEM="$(jq .memory_mib "$SUB")m"
IMAGE="staging-$(echo "$NAME" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9_.\n-' '-')"

echo "== build $NAME ($SHA) cpu=$CPU mem=$MEM"
B0=$(date +%s)
if ! docker build --platform linux/amd64 -t "$IMAGE" "$CAND_DIR/src" > "$RAW/build.log" 2>&1; then
  tail -20 "$RAW/build.log"
  for v in "${VARIANTS[@]}"; do
    record "$v" --build-failed --build-seconds $(( $(date +%s) - B0 )) --log "$RAW/build.log"
  done
  exit 0
fi
BUILD_S=$(( $(date +%s) - B0 ))
IMAGE_MB=$(( $(docker image inspect "$IMAGE" --format '{{.Size}}') / 1048576 ))
echo "built in ${BUILD_S}s, ${IMAGE_MB} MB"

command -v tcpdump > /dev/null || sudo apt-get install -y -qq tcpdump > /dev/null
MOCK_DIR=$(mktemp -d)
(cd "$MOCK_DIR" && exec python3 "$MAIN_SRC/local_tests/mock_platform.py") &
MOCK=$!
trap 'kill $MOCK 2>/dev/null || true' EXIT
sleep 1

for v in "${VARIANTS[@]}"; do
  rm -f "$MOCK_DIR/out/forecast.json"
  ENVS=(-e "SUBMISSION_ID=staging-$NAME" -e "INFERENCE_TIMESTAMP=$T0"
        -e 'OUTPUT_POST={"url":"http://host.docker.internal:8899/","fields":{"key":"local"}}')
  [ "$v" != "default" ] && ENVS+=(-e "SWP_MODEL=$v")
  # DNS queries from containers (the default bridge is in 172.16.0.0/12)
  sudo tcpdump -i any -nn -l 'udp dst port 53 and src net 172.16.0.0/12' \
    > "$RAW/$v.dns" 2> /dev/null &
  TCPDUMP=$!
  sleep 1
  R0=$(date +%s.%N)
  timeout 900 docker run --rm --name "run-$IMAGE-$v" --cpus="$CPU" --memory="$MEM" \
    --add-host=host.docker.internal:host-gateway "${ENVS[@]}" "$IMAGE" "$T0" \
    > "$RAW/$v.log" 2>&1
  CODE=$?
  RUNTIME=$(python3 -c "import time; print(round(time.time() - $R0, 1))")
  [ "$CODE" -eq 124 ] && docker rm -f "run-$IMAGE-$v" > /dev/null 2>&1  # the client was killed
  sleep 1
  sudo kill "$TCPDUMP" 2> /dev/null
  wait "$TCPDUMP" 2> /dev/null
  [ -f "$MOCK_DIR/out/forecast.json" ] && cp "$MOCK_DIR/out/forecast.json" "$RAW/$v.forecast.json"
  echo "== $NAME/$v: exit $CODE in ${RUNTIME}s"
  grep -E '^\[(inference|fetch_inputs)\]' "$RAW/$v.log" | tail -6
  record "$v" --log "$RAW/$v.log" --forecast "$RAW/$v.forecast.json" --exit-code "$CODE" \
    --runtime "$RUNTIME" --build-seconds "$BUILD_S" --image-mb "$IMAGE_MB" --dns "$RAW/$v.dns"
done
exit 0
