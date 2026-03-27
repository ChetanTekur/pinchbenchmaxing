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

from agents.base import log_print, _write_log, TASK_IDS

_PROJECT_ROOT = Path(__file__).parent.parent


def _post_curation_check(train_file: Path, min_per_task: int = 40) -> dict | None:
    """Check if any task dropped below min after a curation step.
    Returns None if OK, or a warning dict with tasks that need data."""
    from collections import Counter
    counts = Counter()
    if train_file.exists():
        for line in train_file.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    counts[rec.get("task_id", "unknown")] += 1
                except json.JSONDecodeError:
                    pass

    missing = [t for t in TASK_IDS if counts.get(t, 0) == 0]
    below_min = {t: counts.get(t, 0) for t in TASK_IDS if 0 < counts.get(t, 0) < min_per_task}

    if missing or below_min:
        return {
            "missing_tasks": missing,
            "below_min_tasks": below_min,
            "warning": (
                f"Curation dropped coverage! "
                f"{len(missing)} tasks now at 0, {len(below_min)} below {min_per_task}. "
                f"Run generate_data to backfill before training."
            ),
        }
    return None


def _run_script(cmd: list[str], label: str, env: dict | None = None) -> tuple[int, str]:
    """
    Run a subprocess with line-buffered stdout, streaming to log.
    Returns (returncode, captured_output).
    """
    import os
    merged = {**os.environ, **(env or {})}
    # Ensure PYTHONPATH includes project root so datagen/stages can find utils/
    merged.setdefault("PYTHONPATH", str(_PROJECT_ROOT))
    if str(_PROJECT_ROOT) not in merged.get("PYTHONPATH", ""):
        merged["PYTHONPATH"] = str(_PROJECT_ROOT) + ":" + merged.get("PYTHONPATH", "")
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

def check_diversity(args: dict, cfg, state) -> dict:
    """Analyze per-task diversity: prompt uniqueness, turn spread, tool combos."""
    try:
        rc, output = _run_script(
            [sys.executable, "-m", "datagen.inspect_data", "diversity"],
            "check_diversity",
        )
        if rc != 0:
            return {"status": "error", "error": f"diversity check exited with code {rc}"}

        # Parse the diversity results from the output
        import re
        from agents.base import TASK_IDS
        low_diversity = []
        missing = []

        for line in output.splitlines():
            if "LOW DIVERSITY" in line:
                m = re.match(r'\s+(task_\w+)', line)
                if m:
                    low_diversity.append(m.group(1))
            if "MISSING" in line:
                m = re.match(r'\s+(task_\w+)', line)
                if m:
                    missing.append(m.group(1))

        return {
            "status": "success",
            "result": {
                "low_diversity_tasks": low_diversity,
                "missing_tasks": missing,
                "needs_attention": len(low_diversity) + len(missing),
            },
            "cost_usd": 0.0,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def inspect_data(args: dict, cfg, state) -> dict:
    """Inspect the training dataset and return statistics."""
    try:
        import re
        from agents.base import TASK_IDS

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

        # Ensure ALL 23 tasks are represented (zeros included)
        for task_id in TASK_IDS:
            if task_id not in by_task:
                by_task[task_id] = 0

        total = sum(by_task.values())
        counts = list(by_task.values())
        missing = [t for t in TASK_IDS if by_task.get(t, 0) == 0]

        if counts:
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

        result = {
            "total": total,
            "per_task": by_task,
            "balance_ratio": balance_ratio,
            "overweight": overweight,
            "underweight": underweight,
            "missing_tasks": missing,
        }

        # Cache in state so turn context always shows data status
        state.last_data_summary = {
            "total": total,
            "missing": missing,
            "below_40": {t: c for t, c in by_task.items() if 0 < c < 40},
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }

        return {"status": "success", "result": result, "cost_usd": 0.0}
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

        # Guard: if Claude passes a string instead of list, fix it
        if isinstance(tasks, str):
            tasks = [t.strip() for t in tasks.split(",") if t.strip()]
        # Guard: validate task IDs look correct
        tasks = [t for t in tasks if t.startswith("task_")]
        if not tasks:
            return {"status": "error", "error": "No valid task IDs (must start with 'task_')"}

        # Guard: don't regenerate data for tasks that already score well
        # EXCEPTION: always allow if task is below training minimum (can't train without it)
        if state.scores:
            import json as _json
            from collections import Counter as _Counter
            _counts = _Counter()
            if cfg.train_file.exists():
                for _line in cfg.train_file.read_text().splitlines():
                    if _line.strip():
                        try:
                            _counts[_json.loads(_line).get("task_id", "")] += 1
                        except _json.JSONDecodeError:
                            pass
            _min = cfg._data.get("data", {}).get("min_per_task", 30)
            protected = [t for t in tasks
                         if state.scores.get(t, 0) >= 0.5 and _counts.get(t, 0) >= _min]
            if protected:
                log_print(f"  [generate_data] SKIPPING {len(protected)} tasks scoring ≥50% with ≥{_min} examples: {protected}")
                tasks = [t for t in tasks if t not in protected]
            if not tasks:
                return {"status": "success", "result": {"generated": 0, "note": "All requested tasks already have sufficient data"}, "cost_usd": 0}

        # Use targeted_topup (task definitions that match what scores well)
        # NOT dynamic_gen (reads PinchBench .md files which have different definitions)
        script = str(_PROJECT_ROOT / "datagen" / "targeted_topup.py")
        generated = {}
        total_generated = 0

        for task in tasks:
            cmd = [
                sys.executable, script, "run",
                "--tasks", task,
                "--min-per-task", str(min_per_task),
            ]
            if diagnosis_file:
                cmd.extend(["--diagnosis-file", str(diagnosis_file)])

            rc, output = _run_script(cmd, f"generate:{task}")

            # Parse actual count from output instead of assuming
            import re
            actual = 0
            for line in output.splitlines():
                m = re.search(r'\+\s*(\d+)', line)
                if m:
                    actual += int(m.group(1))
            generated[task] = {"returncode": rc, "added": actual}
            total_generated += actual

            if rc == 2:
                generated[task]["note"] = "no new data needed"

        return {
            "status": "success",
            "result": {
                "generated": total_generated,
                "per_task": generated,
            },
            "cost_usd": total_generated * 0.04,  # actual examples generated
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

        result = {"kept": after, "removed": removed}

        # Post-curation coverage check
        coverage = _post_curation_check(train_file)
        if coverage:
            result["coverage_warning"] = coverage["warning"]
            result["needs_backfill"] = coverage
            log_print(f"  [filter_data] ⚠ {coverage['warning']}")

        return {"status": "success", "result": result, "cost_usd": 0.0}
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

        result = {
            "before": before,
            "after": after,
            "removed": removed,
            "percent": percent,
        }

        # Post-curation coverage check
        coverage = _post_curation_check(train_file)
        if coverage:
            result["coverage_warning"] = coverage["warning"]
            result["needs_backfill"] = coverage
            log_print(f"  [dedup_data] ⚠ {coverage['warning']}")

        return {"status": "success", "result": result, "cost_usd": 0.0}
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

        result = {
            "before": before,
            "after": after,
            "trimmed": before - after,
            "per_task": per_task,
        }

        # Post-curation coverage check
        coverage = _post_curation_check(train_file)
        if coverage:
            result["coverage_warning"] = coverage["warning"]
            result["needs_backfill"] = coverage
            log_print(f"  [rebalance_data] ⚠ {coverage['warning']}")

        return {
            "status": "success",
            "result": result,
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


# ── validate_data ─────────────────────────────────────────────────────────────

def validate_data(args: dict, cfg, state) -> dict:
    """Run comprehensive data quality validation."""
    try:
        from datagen.validate_data import run_validation
        fix = args.get("fix", False)
        report = run_validation(fix=fix)
        return {
            "status": "success",
            "result": {
                "total_examples": report["total_examples"],
                "clean": report["clean"],
                "with_issues": report["with_issues"],
                "critical_high": report["critical_high"],
                "by_check": report.get("by_check", {}),
                "worst_tasks": report.get("worst_tasks", {}),
                "ready_for_training": report["critical_high"] == 0,
            },
            "cost_usd": 0,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "cost_usd": 0}


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
