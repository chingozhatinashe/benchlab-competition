#!/usr/bin/env bash
# Package src/ into submission.zip for upload, leaving out dev-only content.
#
#   bash package_submission.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

rm -f submission.zip
(cd src && zip -qr ../submission.zip . \
    -x 'data/*' 'experiments/*' 'training/*' 'local_tests/*' \
       '*__pycache__*' '*.pyc' '*.ipynb_checkpoints*' '*.egg-info*')

for f in Dockerfile submission.json README.md; do
  unzip -l submission.zip "$f" > /dev/null \
    || { echo "FAIL: $f missing from ZIP root"; exit 1; }
done

echo "OK — submission.zip ($(wc -c < submission.zip) bytes)"
