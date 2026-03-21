#!/usr/bin/env bash
# Reset loop state for a fresh model run.
#
# Usage:
#   bash scripts/reset_state.sh              # reset to defaults
#   bash scripts/reset_state.sh --keep-data  # reset state but keep dataset

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

WORKSPACE="${PBM_WORKSPACE:-/workspace/synthbench}"
STATE_FILE="$WORKSPACE/data/loop_state.json"

echo "Resetting loop state: $STATE_FILE"

python3 -c "
import json
s = {
    'iteration': 0,
    'scores': {},
    'weak_tasks': [],
    'failure_analysis': {},
    'history': [],
    'model_version': 0,
    'current_ollama_model': '',
    'eval_version': -1,
    'model_history': [],
    'best_avg_score': 0.0,
    'best_version': 0,
    'pause_reason': '',
    'last_analysis': {},
    'model_validated': False,
    'data_gen_version': -1,
    'action_history': [],
    'budget_spent_usd': 0.0
}
json.dump(s, open('$STATE_FILE', 'w'), indent=2)
print('Done — fresh state for new model')
"
