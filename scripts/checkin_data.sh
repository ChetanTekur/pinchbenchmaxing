#!/usr/bin/env bash
# Check current training data into git so we never lose it.
#
# Usage:
#   bash scripts/checkin_data.sh v17          # saves as data/checked_in/v17/
#   bash scripts/checkin_data.sh v17 "balanced data, 1082 examples"

set -euo pipefail

VERSION="${1:?Usage: bash scripts/checkin_data.sh <version> [description]}"
DESC="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

WORKSPACE="${PBM_WORKSPACE:-/workspace/synthbench}"
DEST="data/checked_in/$VERSION"

mkdir -p "$DEST"
cp "$WORKSPACE/data/train.jsonl" "$DEST/train.jsonl"
cp "$WORKSPACE/data/val.jsonl" "$DEST/val.jsonl"

# Save metadata
python3 -c "
import json
from collections import Counter
counts = Counter()
for line in open('$DEST/train.jsonl'):
    if line.strip():
        counts[json.loads(line).get('task_id','')] += 1
meta = {
    'version': '$VERSION',
    'description': '$DESC',
    'total_train': sum(counts.values()),
    'per_task': dict(sorted(counts.items())),
}
json.dump(meta, open('$DEST/metadata.json', 'w'), indent=2)
print(f'Checked in {sum(counts.values())} train examples as {\"$DEST\"}'  )
"

echo ""
echo "To commit: git add $DEST && git commit -m 'Check in $VERSION training data'"
