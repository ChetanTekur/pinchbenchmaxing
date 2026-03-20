"""
Data tool implementations for PinchBench Maxing agent.

Tools: inspect_data, generate_data, generate_adversarial, score_data,
       filter_data, repair_data, dedup_data, rebalance_data, snapshot, push_hf

Each function signature: def tool_name(args: dict, cfg, state) -> dict
All subprocess calls use Popen with line-buffered stdout to avoid deadlocks.
"""

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from agents.base import log_print, _write_log

_PROJECT_ROOT = Path(__file__).parent.parent


def _run_script(cmd: list[str], label: str, env: dict | None = None) -> tuple[int, str]:
    """
    Run a subprocess with line-buffered stdout, streaming to log.
    Returns (returncode, captured_output).
    """
    import os
    merged = {**os.environ, **(env or {})}
    log_print(f"  [{label}] $ {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, env=merged,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    lines = []
    for line in proc.stdout:
        line = line.rstrip('\n')
        print(f"    {line}", flush=True)
        _write_log(f"    {line}")
        lines.append(line)
    proc.wait()
    return proc.returncode, '\n'.join(lines)


# ── inspect_data ─────────────────────────────────────────────────────────────

def inspect_data(args: dict, cfg, state) -> dict:
    """Inspect the training dataset and return statistics."""
    try:
        import re
        rc, output = _run_script(
            [sys.executable, "-m", "datagen.inspect_data", "stats"],
            "inspect_data",
        )
        if rc != 0:
            return {"status": "error", "error": f"inspect_data exited with code {rc}"}

        by_task: dict[str, int] = {}
        for line in output.splitlines():
            m = re.match(r'\s+(task_\w+)\s+(\d+)', line)
            if m:
                by_task[m.group(1)] = int(m.group(2))

        total = sum(by_task.values())
        counts = list(by_task.values())

        if counts:
            avg = total / len(counts)
            min_count = min(counts)
            max_count = max(counts)
            balance_ratio = round(min_count / max_count, 3) if max_count > 0 else 0.0
            target = cfg.data.examples_per_task
            overweight = [t for t, c in by_task.items() if c > target * 1.5]
            underweight = [t for t, c in by_task.items() if c < target * 0.5]
        else:
            balance_ratio = 0.0
            overweight = []
            underweight = []

        return {
            "status": "success",
            "result": {
                "total": total,
                "per_task": by_task,
                "balance_ratio": balance_ratio,
                "overweight": overweight,
                "underweight": underweight,
            },
            "cost_usd": 0.0,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── generate_data ────────────────────────────────────────────────────────────

def generate_data(args: dict, cfg, state) -> dict:
    """Generate targeted training data for specified tasks."""
    try:
        tasks = args.get("tasks", [])
        min_per_task = args.get("min_per_task", 10)
        diagnosis_file = args.get("diagnosis_file")

        if not tasks:
            return {"status": "error", "error": "No tasks specified"}

        script = str(_PROJECT_ROOT / "datagen" / "targeted_topup.py")
        generated = {}
        total = 0

        for task in tasks:
            cmd = [
                sys.executable, script, "run",
                "--tasks", task,
                "--min-per-task", str(min_per_task),
            ]
            if diagnosis_file:
                cmd.extend(["--diagnosis-file", str(diagnosis_file)])

            rc, output = _run_script(cmd, f"generate:{task}")
            generated[task] = {"returncode": rc}
            if rc == 0:
                total += min_per_task  # approximate
            elif rc == 2:
                generated[task]["note"] = "no new data needed"

        return {
            "status": "success",
            "result": {
                "generated": total,
                "per_task": generated,
            },
            "cost_usd": total * 0.04,  # estimate: $0.04/example via Claude Batch API
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── generate_adversarial ─────────────────────────────────────────────────────

def generate_adversarial(args: dict, cfg, state) -> dict:
    """Generate adversarial training examples from benchmark failure transcripts."""
    try:
        tasks = args.get("tasks", [])
        n_per_task = args.get("n_per_task", 10)

        if not tasks:
            return {"status": "error", "error": "No tasks specified"}

        script = str(_PROJECT_ROOT / "datagen" / "adversarial_gen.py")
        log_dir = str(cfg.data_dir.parent / "logs")
        generated = {}
        total = 0

        for task in tasks:
            cmd = [
                sys.executable, script, "run",
                "--log-dir", log_dir,
                "--tasks", task,
                "--n-per-task", str(n_per_task),
            ]
            rc, output = _run_script(cmd, f"adversarial:{task}")
            generated[task] = {"returncode": rc}
            if rc == 0:
                total += n_per_task

        return {
            "status": "success",
            "result": {
                "generated": total,
                "per_task": generated,
            },
            "cost_usd": total * 0.05,  # adversarial is slightly more expensive
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── score_data ───────────────────────────────────────────────────────────────

def score_data(args: dict, cfg, state) -> dict:
    """Run the LLM judge on all unscored examples."""
    try:
        script = str(_PROJECT_ROOT / "datagen" / "llm_judge.py")
        rc, output = _run_script([sys.executable, script, "run"], "score_data")

        if rc != 0:
            return {"status": "error", "error": f"llm_judge.py exited with code {rc}"}

        scores_file = cfg.data_dir / "scores.json"
        total_scored = 0
        new_scored = 0
        if scores_file.exists():
            scores = json.loads(scores_file.read_text())
            total_scored = len(scores)
            # Estimate new scored from output
            import re
            for line in output.splitlines():
                m = re.search(r'(\d+)\s+new', line, re.IGNORECASE)
                if m:
                    new_scored = int(m.group(1))
                    break

        return {
            "status": "success",
            "result": {
                "total_scored": total_scored,
                "new_scored": new_scored,
            },
            "cost_usd": new_scored * 0.01,  # estimate
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── filter_data ──────────────────────────────────────────────────────────────

def filter_data(args: dict, cfg, state) -> dict:
    """Filter training data below minimum judge score."""
    try:
        min_score = args.get("min_score", cfg.data.min_judge_score)
        script = str(_PROJECT_ROOT / "datagen" / "llm_judge.py")

        # Count before
        train_file = cfg.train_file
        before = 0
        if train_file.exists():
            before = sum(1 for line in train_file.read_text().splitlines() if line.strip())

        rc, output = _run_script(
            [sys.executable, script, "filter", "--min", str(min_score)],
            "filter_data",
        )

        if rc != 0:
            return {"status": "error", "error": f"filter exited with code {rc}"}

        # Count after
        after = 0
        if train_file.exists():
            after = sum(1 for line in train_file.read_text().splitlines() if line.strip())

        removed = before - after

        return {
            "status": "success",
            "result": {
                "kept": after,
                "removed": removed,
            },
            "cost_usd": 0.0,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── repair_data ──────────────────────────────────────────────────────────────

def repair_data(args: dict, cfg, state) -> dict:
    """Repair borderline examples by re-generating with targeted prompts."""
    try:
        min_score = args.get("min_score", 2)
        max_score = args.get("max_score", 3)
        script = str(_PROJECT_ROOT / "datagen" / "example_repair.py")

        if not Path(script).exists():
            return {"status": "error", "error": "example_repair.py not found"}

        rc, output = _run_script(
            [sys.executable, script, "run",
             "--min-score", str(min_score),
             "--max-score", str(max_score)],
            "repair_data",
        )

        if rc != 0:
            return {"status": "error", "error": f"repair exited with code {rc}"}

        # Read the repair report if available
        report_file = cfg.data_dir / "repair_report.json"
        result = {"attempted": 0, "improved": 0, "failed": 0}
        if report_file.exists():
            report = json.loads(report_file.read_text())
            result["attempted"] = report.get("attempted", 0)
            result["improved"] = report.get("improved", 0)
            result["failed"] = report.get("failed", 0)

        return {
            "status": "success",
            "result": result,
            "cost_usd": result["attempted"] * 0.03,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── dedup_data ───────────────────────────────────────────────────────────────

def dedup_data(args: dict, cfg, state) -> dict:
    """Deduplicate semantically similar training examples."""
    try:
        threshold = args.get("threshold", 0.85)
        script = str(_PROJECT_ROOT / "datagen" / "dedup.py")

        if not Path(script).exists():
            return {"status": "error", "error": "dedup.py not found"}

        # Count before
        train_file = cfg.train_file
        before = 0
        if train_file.exists():
            before = sum(1 for line in train_file.read_text().splitlines() if line.strip())

        cmd = [sys.executable, script, "run"]
        # Pass threshold if the script supports it
        cmd.extend(["--threshold", str(threshold)])

        rc, output = _run_script(cmd, "dedup_data")

        # Read report
        report_file = cfg.data_dir / "dedup_report.json"
        if report_file.exists():
            report = json.loads(report_file.read_text())
            removed = report.get("removed", 0)
            after = before - removed
            percent = report.get("percent_removed", 0)
        else:
            after = 0
            if train_file.exists():
                after = sum(1 for line in train_file.read_text().splitlines() if line.strip())
            removed = before - after
            percent = round(removed / before * 100, 1) if before > 0 else 0

        return {
            "status": "success",
            "result": {
                "before": before,
                "after": after,
                "removed": removed,
                "percent": percent,
            },
            "cost_usd": 0.0,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── rebalance_data ───────────────────────────────────────────────────────────

def rebalance_data(args: dict, cfg, state) -> dict:
    """Rebalance the dataset by trimming overweight tasks."""
    try:
        target = args.get("target", 120)

        # Count before
        train_file = cfg.train_file
        before = 0
        if train_file.exists():
            before = sum(1 for line in train_file.read_text().splitlines() if line.strip())

        rc, output = _run_script(
            [sys.executable, "-m", "datagen.rebalance", "--target", str(target)],
            "rebalance_data",
        )

        if rc != 0:
            return {"status": "error", "error": f"rebalance exited with code {rc}"}

        # Count after
        after = 0
        if train_file.exists():
            after = sum(1 for line in train_file.read_text().splitlines() if line.strip())

        # Get per-task counts after rebalance
        import re
        per_task = {}
        for line in output.splitlines():
            m = re.match(r'\s+(task_\w+)\s+(\d+)', line)
            if m:
                per_task[m.group(1)] = int(m.group(2))

        return {
            "status": "success",
            "result": {
                "before": before,
                "after": after,
                "trimmed": before - after,
                "per_task": per_task,
            },
            "cost_usd": 0.0,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── snapshot ─────────────────────────────────────────────────────────────────

def snapshot(args: dict, cfg, state) -> dict:
    """Create a timestamped snapshot of train.jsonl and val.jsonl."""
    try:
        label = args.get("label", "snapshot")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_dir = cfg.data_dir / "snapshots" / f"{label}_{ts}"
        snap_dir.mkdir(parents=True, exist_ok=True)

        copied = []
        for fname in ["train.jsonl", "val.jsonl"]:
            src = cfg.data_dir / fname
            if src.exists():
                shutil.copy2(str(src), str(snap_dir / fname))
                copied.append(fname)

        log_print(f"  [snapshot] {', '.join(copied)} -> {snap_dir}")

        return {
            "status": "success",
            "result": {
                "path": str(snap_dir),
                "files": copied,
            },
            "cost_usd": 0.0,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── push_hf ──────────────────────────────────────────────────────────────────

def push_hf(args: dict, cfg, state) -> dict:
    """Push the current dataset to HuggingFace."""
    try:
        message = args.get("message", f"snapshot {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        script = str(_PROJECT_ROOT / "scripts" / "push_to_hf.sh")

        rc, output = _run_script(
            ["bash", script, message],
            "push_hf",
        )

        if rc != 0:
            return {"status": "error", "error": f"push_to_hf.sh exited with code {rc}"}

        try:
            repo = cfg.huggingface.dataset_repo
        except AttributeError:
            repo = "unknown"

        return {
            "status": "success",
            "result": {
                "repo": repo,
                "files_pushed": ["train.jsonl", "val.jsonl", "scores.json"],
            },
            "cost_usd": 0.0,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
