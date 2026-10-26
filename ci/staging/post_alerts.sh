#!/usr/bin/env bash
# Post staging alerts as GitHub issues: one open issue per candidate (or "schedule"),
# labelled staging-alert, titled "Staging: <name>"; new alerts are added as a comment
# when it is already open. GitHub emails the repository owner on each. Close by hand.
#
#   bash ci/staging/post_alerts.sh alerts.json "<run url>"
#
# alerts.json: {name: [markdown lines]} (events.py cycle / evaluate.py). Needs GH_TOKEN.
set -euo pipefail

ALERTS=${1:?alerts.json}
RUN_URL=${2:-}
[ -s "$ALERTS" ] || exit 0
NAMES=$(jq -r 'keys[]' "$ALERTS")
[ -n "$NAMES" ] || { echo "no alerts"; exit 0; }

gh label create staging-alert --color B60205 \
  --description "Staging environment problems" --force > /dev/null 2>&1 || true

while IFS= read -r name; do
  title="Staging: $name"
  body=$(jq -r --arg n "$name" '.[$n][]' "$ALERTS")
  body="$body"$'\n\n'"Run: $RUN_URL"$'\n\n'"Records and the daily report: the \`staging-records\` branch."
  number=$(gh issue list --label staging-alert --state open --limit 100 --json number,title \
             | jq -r --arg t "$title" '.[] | select(.title == $t) | .number' | head -1)
  if [ -n "$number" ]; then
    gh issue comment "$number" --body "$body" > /dev/null
    echo "commented on #$number ($title)"
  else
    gh issue create --title "$title" --label staging-alert --body "$body" > /dev/null
    echo "opened an issue: $title"
  fi
done <<< "$NAMES"
