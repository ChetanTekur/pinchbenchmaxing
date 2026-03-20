#!/usr/bin/env bash
# Upload backup tar to HuggingFace dataset repo.
#
# Usage:
#   bash scripts/backup_to_hf.sh /workspace/pbm_backup_with_adapters.tar.gz

set -euo pipefail

FILE="${1:-/workspace/pbm_backup_with_adapters.tar.gz}"

if [ ! -f "$FILE" ]; then
    echo "ERROR: File not found: $FILE"
    exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set"
    exit 1
fi

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && python3 -c "from utils.config import load_config; print(load_config().huggingface.dataset_repo)")
FILENAME=$(basename "$FILE")
SIZE=$(du -h "$FILE" | cut -f1)

echo "Uploading $FILENAME ($SIZE) to $REPO..."

python3 -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ['HF_TOKEN'])
api.upload_file(
    path_or_fileobj='$FILE',
    path_in_repo='backups/$FILENAME',
    repo_id='$REPO',
    repo_type='dataset',
    commit_message='Backup: $FILENAME'
)
print('Done! https://huggingface.co/datasets/$REPO')
"
