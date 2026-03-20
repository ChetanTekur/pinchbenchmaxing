#!/usr/bin/env bash
# Push dataset to HuggingFace for versioning and public access.
#
# Usage:
#   bash scripts/push_to_hf.sh                    # push with auto-generated message
#   bash scripts/push_to_hf.sh "v7 pre-rebalance" # push with custom message
#
# Requires: HF_TOKEN set in environment (add to set_env.sh)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

WORKSPACE="${PBM_WORKSPACE:-/workspace/synthbench}"
DATA_DIR="$WORKSPACE/data"
MSG="${1:-dataset snapshot $(date '+%Y-%m-%d %H:%M')}"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set"
    echo "  Add to your env file: echo 'export HF_TOKEN=\"hf_...\"' >> $WORKSPACE/set_env.sh"
    echo "  Then: source $WORKSPACE/set_env.sh"
    exit 1
fi

REPO=$(python3 -c "from utils.config import load_config; print(load_config().huggingface.dataset_repo)")

echo "Pushing to huggingface.co/datasets/$REPO"
echo "  Message: $MSG"
echo ""

python3 -c "
from huggingface_hub import HfApi
from pathlib import Path
import os

api = HfApi(token=os.environ['HF_TOKEN'])
api.create_repo('$REPO', repo_type='dataset', exist_ok=True)

data_dir = Path('$DATA_DIR')
files = ['train.jsonl', 'val.jsonl', 'scores.json', 'loop_state.json']

# Also upload dataset card
card = Path('dataset_card.md')
if card.exists():
    files_to_upload = [(str(card), 'README.md')]
else:
    files_to_upload = []

for f in files:
    p = data_dir / f
    if p.exists():
        files_to_upload.append((str(p), f))

for local, remote in files_to_upload:
    size_mb = Path(local).stat().st_size / 1024 / 1024
    print(f'  Uploading {remote} ({size_mb:.1f} MB)...')
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=remote,
        repo_id='$REPO',
        repo_type='dataset',
        commit_message='$MSG',
    )

print()
print(f'Done! https://huggingface.co/datasets/$REPO')
"
