#!/usr/bin/env python3
"""Compare data snapshots across multiple versions.

Usage:
  python scripts/compare_versions.py                    # all available snapshots
  python scripts/compare_versions.py 21 23 25 29        # specific versions
"""
import json
import os
import sys
from pathlib import Path

WS = os.environ.get("PBM_WORKSPACE", "/workspace/synthbench")
DATA_DIR = Path(WS) / "data"


def load_snapshot(version):
    snap_file = DATA_DIR / f"data_snapshot_v{version}.json"
    if not snap_file.exists():
        return None
    snap = json.loads(snap_file.read_text())
    return snap.get("per_task", {})


def main():
    if len(sys.argv) > 1:
        versions = [int(v) for v in sys.argv[1:]]
    else:
        # Find all available snapshots
        versions = []
        for f in sorted(DATA_DIR.glob("data_snapshot_v*.json")):
            try:
                v = int(f.stem.replace("data_snapshot_v", ""))
                versions.append(v)
            except ValueError:
                pass

    if not versions:
        print("No snapshots found")
        return

    # Load all snapshots
    data = {}
    for v in versions:
        snap = load_snapshot(v)
        if snap:
            data[v] = snap
        else:
            print(f"  v{v}: snapshot not found, skipping")

    if not data:
        print("No valid snapshots loaded")
        return

    # Collect all task IDs
    all_tasks = sorted(set(t for counts in data.values() for t in counts))
    versions_found = sorted(data.keys())

    # Header
    header = f"{'Task':<40}"
    for v in versions_found:
        header += f" {'v'+str(v):>6}"
    # Add delta column (last vs first)
    if len(versions_found) >= 2:
        header += f" {'delta':>6}"
    print(header)
    print("-" * len(header))

    # Rows
    total = {v: 0 for v in versions_found}
    for t in all_tasks:
        row = f"{t:<40}"
        vals = []
        for v in versions_found:
            c = data[v].get(t, 0)
            total[v] += c
            vals.append(c)
            row += f" {c:>6}"
        if len(versions_found) >= 2:
            d = vals[-1] - vals[0]
            marker = " <--" if d != 0 else ""
            row += f" {d:>+6}{marker}"
        print(row)

    # Total row
    row = f"{'TOTAL':<40}"
    for v in versions_found:
        row += f" {total[v]:>6}"
    if len(versions_found) >= 2:
        d = total[versions_found[-1]] - total[versions_found[0]]
        row += f" {d:>+6}"
    print("-" * len(header))
    print(row)

    # Show which versions had significant changes
    if len(versions_found) >= 2:
        print(f"\nBiggest changes (v{versions_found[0]} -> v{versions_found[-1]}):")
        changes = []
        for t in all_tasks:
            first = data[versions_found[0]].get(t, 0)
            last = data[versions_found[-1]].get(t, 0)
            if first != last:
                changes.append((t, first, last, last - first))
        changes.sort(key=lambda x: abs(x[3]), reverse=True)
        for t, first, last, d in changes[:10]:
            print(f"  {t:<40} {first:>4} -> {last:>4} ({d:>+4})")


if __name__ == "__main__":
    main()
