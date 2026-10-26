#!/usr/bin/env bash
# The staging records branch (an orphan branch holding only run records, events, truth
# and reports), checked out in its own directory so main's checkout is untouched.
#
#   bash ci/staging/records.sh checkout records       # creates the branch on first use
#   bash ci/staging/records.sh push records "message"
#
# Needs GH_TOKEN (the workflow's GITHUB_TOKEN, contents: write) and GITHUB_REPOSITORY.
set -euo pipefail

BRANCH=staging-records
CMD=${1:?checkout or push}
DIR=${2:?directory}

case "$CMD" in
  checkout)
    mkdir -p "$DIR"
    cd "$DIR"
    git init -q
    git remote add origin "https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
    git config user.name "github-actions[bot]"
    git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
    if git fetch -q --depth 1 origin "$BRANCH" 2>/dev/null; then
      git checkout -q -b "$BRANCH" FETCH_HEAD
    else
      git checkout -q --orphan "$BRANCH"
      printf '# Staging records\n\nWritten by .github/workflows/staging.yml on main.\n' > README.md
      echo "created the $BRANCH branch"
    fi
    ;;
  push)
    MESSAGE=${3:?message}
    cd "$DIR"
    git add -A
    if git diff --cached --quiet; then
      echo "records: nothing to commit"
      exit 0
    fi
    git commit -q -m "$MESSAGE"
    for attempt in 1 2 3; do
      if git push -q origin "HEAD:$BRANCH"; then
        echo "records: pushed ($MESSAGE)"
        exit 0
      fi
      # someone else pushed first: replay this commit on top and retry
      git fetch -q --depth 50 origin "$BRANCH" && git rebase -q FETCH_HEAD || true
      sleep $((attempt * 5))
    done
    echo "::error::could not push the records branch"
    exit 1
    ;;
  *)
    echo "usage: records.sh checkout|push DIR [MESSAGE]" >&2
    exit 2
    ;;
esac
