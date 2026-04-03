#!/usr/bin/env bash
# Run benchmark 2x each for multiple models and produce a comparison table.
#
# Usage:
#   bash scripts/variance_test.sh
#
# Prerequisites:
#   - All models must be registered in Ollama
#   - startup.sh must have been run

set -uo pipefail

WORKSPACE="${PBM_WORKSPACE:-/workspace/synthbench}"
LOG_DIR="$WORKSPACE/logs/variance_test"
RESULTS_FILE="$LOG_DIR/results.txt"
BENCH_DIR="$WORKSPACE/skill"
JUDGE="${PBM_JUDGE_MODEL:-openrouter/anthropic/claude-opus-4.5}"

mkdir -p "$LOG_DIR"

MODELS=("qwen35-9b-clawd-v21" "qwen35-9b-clawd-v23" "qwen35-9b-clawd-v29")
RUNS_PER_MODEL=2

# Check and register models
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "=== Variance Test ==="
echo "  Models: ${MODELS[*]}"
echo "  Runs per model: $RUNS_PER_MODEL"
echo ""
echo "--- Checking models ---"
for m in "${MODELS[@]}"; do
    if ollama list 2>/dev/null | grep -q "^${m}"; then
        echo "  [OK] $m"
    else
        echo "  [MISSING] $m -- registering..."
        # Extract version number (e.g. qwen35-9b-clawd-v23 -> 23)
        ver=$(echo "$m" | grep -oP 'v\K\d+$')
        if [ -n "$ver" ]; then
            bash "$SCRIPT_DIR/register_model.sh" "v${ver}" "$m" 2>&1 | tail -3
            if ollama list 2>/dev/null | grep -q "^${m}"; then
                echo "  [OK] $m registered"
            else
                echo "  [FAIL] $m could not be registered"
            fi
        else
            echo "  [FAIL] could not parse version from $m"
        fi
    fi
done
echo ""

# Parse scores from a log file
parse_scores() {
    local log_file="$1"
    python3 -c "
import re, json
text = open('$log_file', errors='replace').read()
scores = {}
for m in re.finditer(r'Task (task_\w+):\s*([01](?:\.\d+)?)\s*/\s*1\.0', text):
    scores[m.group(1)] = float(m.group(2))
if scores:
    avg = sum(scores.values()) / len(scores)
    print(f'{avg:.4f}')
    # Also dump per-task to a json sidecar
    json.dump(scores, open('${log_file}.scores.json', 'w'), indent=2)
else:
    print('FAILED')
"
}

# Run benchmarks
echo "--- Running benchmarks ---"
echo ""

declare -A ALL_SCORES

for model in "${MODELS[@]}"; do
    if ! ollama list 2>/dev/null | grep -q "^${model}"; then
        echo "SKIPPING $model (not registered)"
        echo ""
        continue
    fi

    for run in $(seq 1 $RUNS_PER_MODEL); do
        log_file="$LOG_DIR/${model}_run${run}.log"
        echo "[$model] Run $run/$RUNS_PER_MODEL..."
        echo "  Log: $log_file"
        echo "  Started: $(date '+%H:%M:%S')"

        cd "$BENCH_DIR"
        ./scripts/run.sh --model "ollama/${model}" --judge "$JUDGE" > "$log_file" 2>&1
        rc=$?
        cd - > /dev/null

        if [ $rc -ne 0 ]; then
            echo "  FAILED (exit $rc)"
            ALL_SCORES["${model}_run${run}"]="FAILED"
        else
            score=$(parse_scores "$log_file")
            ALL_SCORES["${model}_run${run}"]="$score"
            pct=$(python3 -c "print(f'{float(\"$score\")*100:.1f}%')" 2>/dev/null || echo "?%")
            echo "  Score: $score ($pct)"
        fi
        echo "  Finished: $(date '+%H:%M:%S')"
        echo ""
    done
done

# Summary table
echo "==========================================================="
echo "  VARIANCE TEST RESULTS"
echo "==========================================================="
echo ""
printf "  %-30s %8s %8s %8s\n" "Model" "Run 1" "Run 2" "Delta"
printf "  %-30s %8s %8s %8s\n" "-----" "-----" "-----" "-----"

for model in "${MODELS[@]}"; do
    s1="${ALL_SCORES[${model}_run1]:-N/A}"
    s2="${ALL_SCORES[${model}_run2]:-N/A}"

    if [[ "$s1" != "N/A" && "$s1" != "FAILED" && "$s2" != "N/A" && "$s2" != "FAILED" ]]; then
        delta=$(python3 -c "print(f'{abs(float(\"$s1\") - float(\"$s2\"))*100:.1f}%')")
        pct1=$(python3 -c "print(f'{float(\"$s1\")*100:.1f}%')")
        pct2=$(python3 -c "print(f'{float(\"$s2\")*100:.1f}%')")
        printf "  %-30s %8s %8s %8s\n" "$model" "$pct1" "$pct2" "$delta"
    else
        printf "  %-30s %8s %8s %8s\n" "$model" "$s1" "$s2" "---"
    fi
done

echo ""
echo "  Per-task scores: $LOG_DIR/*.scores.json"
echo "==========================================================="

# Also write to file
{
    echo "Variance Test $(date '+%Y-%m-%d %H:%M')"
    echo ""
    for model in "${MODELS[@]}"; do
        s1="${ALL_SCORES[${model}_run1]:-N/A}"
        s2="${ALL_SCORES[${model}_run2]:-N/A}"
        echo "$model: run1=$s1 run2=$s2"
    done
} > "$RESULTS_FILE"
echo "  Results saved: $RESULTS_FILE"
