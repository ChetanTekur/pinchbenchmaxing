#!/usr/bin/env bash
# Compare v21 gold data vs v22 training data
set -euo pipefail

echo "=== V21 GOLD ==="
python3 -c "
import json
from collections import Counter
c = Counter()
for line in open('/workspace/synthbench/data/gold_v21/train.jsonl'):
    if line.strip():
        c[json.loads(line)['task_id']] += 1
for k,v in sorted(c.items()):
    print(f'  {k}: {v}')
print(f'  TOTAL: {sum(c.values())}')
"

echo ""
echo "=== V22 SNAPSHOT ==="
if [ -f /workspace/synthbench/data/data_snapshot_v22.json ]; then
    python3 -c "
import json
d = json.load(open('/workspace/synthbench/data/data_snapshot_v22.json'))
for k,v in sorted(d.get('per_task', d).items()):
    if k.startswith('task_'):
        print(f'  {k}: {v}')
print(f'  TOTAL: {d.get(\"total\", sum(v for k,v in d.items() if k.startswith(\"task_\")))}')
"
else
    echo "  No snapshot file found"
fi

echo ""
echo "=== DIFF (v22 - v21) ==="
python3 -c "
import json
from collections import Counter

v21 = Counter()
for line in open('/workspace/synthbench/data/gold_v21/train.jsonl'):
    if line.strip():
        v21[json.loads(line)['task_id']] += 1

v22 = {}
if True:
    try:
        d = json.load(open('/workspace/synthbench/data/data_snapshot_v22.json'))
        v22 = {k:v for k,v in d.items() if k.startswith('task_')}
        if not v22:
            v22 = d.get('per_task', {})
    except:
        pass

all_tasks = sorted(set(list(v21.keys()) + list(v22.keys())))
for t in all_tasks:
    a = v21.get(t, 0)
    b = v22.get(t, 0)
    diff = b - a
    marker = '' if diff == 0 else f' ← {\"ADDED\" if diff > 0 else \"REMOVED\"} {abs(diff)}'
    print(f'  {t}: {a} → {b} ({diff:+d}){marker}')
"
