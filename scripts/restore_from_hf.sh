#!/usr/bin/env bash
# Restore dataset and adapters from HuggingFace after region move.
#
# Usage:
#   bash scripts/restore_from_hf.sh
#
# Requires: pip install huggingface_hub

set -euo pipefail

WORKSPACE="${PBM_WORKSPACE:-/workspace/synthbench}"

echo "=== Restoring from HuggingFace ==="
echo "  Workspace: $WORKSPACE"

mkdir -p "$WORKSPACE/data" "$WORKSPACE/logs"

# Download dataset files
echo ""
echo "--- Downloading dataset ---"
python3 -c "
from huggingface_hub import hf_hub_download
import os
data_dir = '$WORKSPACE/data'
for f in ['train.jsonl', 'val.jsonl', 'scores.json', 'loop_state.json']:
    try:
        hf_hub_download('cptekur/pinchbench-clawd', f, local_dir=data_dir, repo_type='dataset')
        print(f'  Got {f}')
    except Exception as e:
        print(f'  Skipped {f}: {e}')
"

# Download adapters backup
echo ""
echo "--- Downloading adapters backup ---"
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('cptekur/pinchbench-clawd', 'backups/pbm_backup_with_adapters.tar.gz',
                local_dir='$WORKSPACE', repo_type='dataset')
print('  Got backup')
"

# Extract adapters
echo ""
echo "--- Extracting adapters ---"
cd "$WORKSPACE"
if [ -f "backups/pbm_backup_with_adapters.tar.gz" ]; then
    tar xzf backups/pbm_backup_with_adapters.tar.gz
    echo "  Extracted"
fi

# Set up env file if missing
if [ ! -f "$WORKSPACE/set_env.sh" ]; then
    echo ""
    echo "--- set_env.sh not found, creating from template ---"
    cp /root/pbm/scripts/set_env.sh "$WORKSPACE/set_env.sh"
    echo "  IMPORTANT: edit $WORKSPACE/set_env.sh and fill in your API keys"
    echo "  vim $WORKSPACE/set_env.sh"
fi

echo ""
echo "=== Restore complete ==="
echo ""
echo "  Next steps:"
echo "    source $WORKSPACE/set_env.sh"
echo "    cd /root/pbm"
echo "    tmux new -s loop"
echo "    python3 orchestrator.py run --model qwen35-9b-clawd-v8 --dry-run"
