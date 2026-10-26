#!/usr/bin/env bash
# Report staging alerts in the workflow run: one error annotation per alert and a section
# in the job summary, then exit 1 so the run fails and GitHub emails the repository owner
# (issues are off on this repository). Runs after the records are pushed, so a failed run
# loses nothing.
#
#   bash ci/staging/report_alerts.sh alerts.json
#
# alerts.json: {name: [markdown lines]} (events.py cycle / evaluate.py).
set -euo pipefail

ALERTS=${1:?alerts.json}
[ -s "$ALERTS" ] || { echo "no alerts"; exit 0; }
COUNT=$(jq '[.[] | length] | add // 0' "$ALERTS")
[ "$COUNT" -gt 0 ] || { echo "no alerts"; exit 0; }

{
  echo "## Staging alerts ($COUNT)"
  jq -r 'to_entries[] | "\n### \(.key)\n", .value[]' "$ALERTS"
  echo
  echo "Records and the daily report: the \`staging-records\` branch."
} >> "${GITHUB_STEP_SUMMARY:-/dev/stderr}"

jq -r 'to_entries[] | .key as $n | .value[] | "::error title=Staging: \($n)::\(ltrimstr("- "))"' "$ALERTS"
exit 1
