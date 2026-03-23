#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

echo "=== Removing bad data for 5 tasks ==="
python3 scripts/remove_task_data.py task_08_memory task_13_image_gen task_14_humanizer task_15_daily_summary task_19_spreadsheet_summary

echo ""
echo "=== Regenerating with pilot-validate-refine ==="
python3 -m datagen.dynamic_gen run --tasks task_08_memory,task_13_image_gen,task_14_humanizer,task_15_daily_summary,task_19_spreadsheet_summary --min-per-task 50

echo ""
echo "=== Validating ==="
python3 -m datagen.validate_data
python3 -m datagen.inspect_data stats
