#!/usr/bin/env python3
"""
Semantic deduplication — removes near-identical training examples.

After multiple topup rounds, tasks accumulate similar examples (same phrasing,
same tool sequence). Training on duplicates wastes capacity and causes overfitting.

Uses TF-IDF cosine similarity on user messages + Jaccard similarity on tool calls.
Both thresholds must be exceeded to flag a pair as duplicate.

Usage:
  python dedup.py report                   # show duplicate clusters (no changes)
  python dedup.py run                      # remove duplicates, keep highest-scored
  python dedup.py run --threshold 0.90     # stricter threshold
"""

import json
import sys
import argparse
import re
from pathlib import Path
from collections import defaultdict

from utils.config import load_config

_cfg        = load_config()
TRAIN_FILE  = _cfg.train_file
VAL_FILE    = _cfg.val_file
SCORES_FILE = _cfg.data_dir / "scores.json"
REPORT_FILE = _cfg.data_dir / "dedup_report.json"

DEFAULT_TEXT_THRESHOLD = 0.85
DEFAULT_TOOL_THRESHOLD = 0.80


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_user_message(example: dict) -> str:
    """Get the first user message from an example."""
    for msg in example.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def extract_tool_calls(example: dict) -> set[str]:
    """Get the set of tool names called in an example."""
    tools = set()
    for msg in example.get("messages", []):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            for m in re.finditer(r'"name"\s*:\s*"(\w+)"', content):
                tools.add(m.group(1))
    return tools


def jaccard_similarity(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_example_key(example: dict) -> str:
    """Build a stable key for an example (same as llm_judge)."""
    task_id = example.get("task_id", "unknown")
    user_msg = extract_user_message(example)
    return f"{task_id}::{user_msg[:80]}"


# ─────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def find_duplicates(
    examples: list[dict],
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    tool_threshold: float = DEFAULT_TOOL_THRESHOLD,
) -> list[list[int]]:
    """Find clusters of duplicate examples. Returns list of index clusters."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("ERROR: scikit-learn required for dedup. Install: pip install scikit-learn")
        sys.exit(1)

    if len(examples) < 2:
        return []

    # Group by task_id to only compare within the same task
    task_groups = defaultdict(list)
    for i, ex in enumerate(examples):
        task_groups[ex.get("task_id", "unknown")].append(i)

    all_clusters = []

    for task_id, indices in task_groups.items():
        if len(indices) < 2:
            continue

        # Get user messages for this task group
        user_messages = [extract_user_message(examples[i]) for i in indices]
        tool_sets = [extract_tool_calls(examples[i]) for i in indices]

        # Filter out empty messages
        valid = [(idx, msg, tools) for idx, msg, tools
                 in zip(indices, user_messages, tool_sets) if msg.strip()]
        if len(valid) < 2:
            continue

        valid_indices = [v[0] for v in valid]
        valid_messages = [v[1] for v in valid]
        valid_tools = [v[2] for v in valid]

        # TF-IDF cosine similarity
        try:
            vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
            tfidf = vectorizer.fit_transform(valid_messages)
            sim_matrix = cosine_similarity(tfidf)
        except ValueError:
            # All messages might be identical or empty after stopword removal
            continue

        # Find pairs exceeding both thresholds
        merged = set()  # indices already in a cluster
        for i in range(len(valid_indices)):
            if valid_indices[i] in merged:
                continue
            cluster = [valid_indices[i]]
            for j in range(i + 1, len(valid_indices)):
                if valid_indices[j] in merged:
                    continue
                text_sim = sim_matrix[i][j]
                tool_sim = jaccard_similarity(valid_tools[i], valid_tools[j])
                if text_sim >= text_threshold and tool_sim >= tool_threshold:
                    cluster.append(valid_indices[j])
                    merged.add(valid_indices[j])
            if len(cluster) > 1:
                all_clusters.append(cluster)
                merged.update(cluster)

    return all_clusters


def select_best_in_cluster(
    examples: list[dict], cluster: list[int], scores: dict,
) -> int:
    """Pick the best example index from a cluster (highest judge score)."""
    best_idx = cluster[0]
    best_score = -1

    for idx in cluster:
        key = build_example_key(examples[idx])
        s = scores.get(key, {}).get("score", 0)
        if s > best_score:
            best_score = s
            best_idx = idx

    return best_idx


# ─────────────────────────────────────────────────────────────────────────────
# COMMANDS
# ─────────────────────────────────────────────────────────────────────────────

def load_all_examples() -> tuple[list[dict], list[str]]:
    """Load all examples from train + val, tracking which file each came from."""
    examples = []
    sources  = []  # "train" or "val" per example
    for path, label in [(TRAIN_FILE, "train"), (VAL_FILE, "val")]:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
                sources.append(label)
            except json.JSONDecodeError:
                continue
    return examples, sources


def cmd_report(text_threshold: float, tool_threshold: float):
    """Show duplicate clusters without removing anything."""
    examples, _ = load_all_examples()
    clusters = find_duplicates(examples, text_threshold, tool_threshold)

    if not clusters:
        print("No duplicates found.")
        return

    total_removable = sum(len(c) - 1 for c in clusters)
    print(f"Found {len(clusters)} duplicate clusters ({total_removable} removable examples)\n")

    for i, cluster in enumerate(clusters[:20]):  # show first 20
        task = examples[cluster[0]].get("task_id", "?")
        print(f"  Cluster {i+1} ({task}): {len(cluster)} examples")
        for idx in cluster[:3]:
            user = extract_user_message(examples[idx])[:80]
            print(f"    [{idx}] {user}")
        if len(cluster) > 3:
            print(f"    ... and {len(cluster) - 3} more")


def cmd_run(text_threshold: float, tool_threshold: float):
    """Remove duplicates, keeping the highest-scored example per cluster."""
    examples, sources = load_all_examples()

    scores = {}
    if SCORES_FILE.exists():
        scores = json.loads(SCORES_FILE.read_text())

    clusters = find_duplicates(examples, text_threshold, tool_threshold)
    if not clusters:
        print("No duplicates found. Dataset is clean.")
        report = {"total_before": len(examples), "total_after": len(examples),
                  "removed": 0, "percent_removed": 0, "clusters": []}
        REPORT_FILE.write_text(json.dumps(report, indent=2))
        return

    # Determine which indices to remove
    remove_indices = set()
    cluster_report = []

    for cluster in clusters:
        best_idx = select_best_in_cluster(examples, cluster, scores)
        removed = [idx for idx in cluster if idx != best_idx]
        remove_indices.update(removed)

        cluster_report.append({
            "task": examples[cluster[0]].get("task_id", "?"),
            "size": len(cluster),
            "kept_idx": best_idx,
            "removed_count": len(removed),
        })

    # Rewrite files
    train_lines = []
    val_lines = []
    for i, (ex, src) in enumerate(zip(examples, sources)):
        if i in remove_indices:
            continue
        line = json.dumps(ex)
        if src == "train":
            train_lines.append(line)
        else:
            val_lines.append(line)

    TRAIN_FILE.write_text("\n".join(train_lines) + "\n" if train_lines else "")
    VAL_FILE.write_text("\n".join(val_lines) + "\n" if val_lines else "")

    total_before = len(examples)
    total_after  = total_before - len(remove_indices)
    pct_removed  = round(len(remove_indices) / max(total_before, 1) * 100, 1)

    report = {
        "total_before": total_before,
        "total_after":  total_after,
        "removed":      len(remove_indices),
        "percent_removed": pct_removed,
        "clusters":     cluster_report,
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2))

    print(f"\n{'─'*50}")
    print(f"Dedup complete:")
    print(f"  Before: {total_before}")
    print(f"  After:  {total_after}")
    print(f"  Removed: {len(remove_indices)} ({pct_removed}%)")
    print(f"  Clusters: {len(clusters)}")

    if pct_removed > 30:
        print(f"\n  ⚠ WARNING: Removed {pct_removed}% — generation may lack diversity")


def main():
    parser = argparse.ArgumentParser(description="Semantic deduplication of training examples")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Remove duplicates")
    run_p.add_argument("--threshold", type=float, default=DEFAULT_TEXT_THRESHOLD,
                       help="Text cosine similarity threshold")
    run_p.add_argument("--tool-threshold", type=float, default=DEFAULT_TOOL_THRESHOLD,
                       help="Tool Jaccard similarity threshold")

    report_p = sub.add_parser("report", help="Show duplicates without removing")
    report_p.add_argument("--threshold", type=float, default=DEFAULT_TEXT_THRESHOLD)
    report_p.add_argument("--tool-threshold", type=float, default=DEFAULT_TOOL_THRESHOLD)

    args = parser.parse_args()
    text_t = getattr(args, "threshold", DEFAULT_TEXT_THRESHOLD)
    tool_t = getattr(args, "tool_threshold", DEFAULT_TOOL_THRESHOLD)

    if args.command == "run":
        cmd_run(text_t, tool_t)
    elif args.command == "report":
        cmd_report(text_t, tool_t)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
