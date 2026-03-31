"""
Data tool implementations for PinchBench Maxing agent.

Tools: inspect_data, generate_data, score_data,
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


def _prune_stale_scores(cfg) -> int:
    """Remove score entries that don't match any current training example.
    Returns number of stale entries removed."""
    scores_file = cfg.data_dir / "scores.json"
    if not scores_file.exists():
        return 0

    scores = json.loads(scores_file.read_text())
    before = len(scores)

    # Build valid keys from current train data
    valid_keys = set()
    train_file = cfg.data_dir / "train.jsonl"
    if train_file.exists():
        for line in train_file.read_text().splitlines():
            if not line.strip():
                continue
            try:
                ex = json.loads(line)
                task_id = ex.get("task_id", "")
                msgs = ex.get("messages", [])
                user_msgs = [m for m in msgs if m.get("role") == "user"]
                user_text = user_msgs[0]["content"][:80] if user_msgs else ""
                valid_keys.add(f"{task_id}|{user_text}")
                valid_keys.add(f"{task_id}::{user_text}")
            except (json.JSONDecodeError, IndexError, KeyError):
                pass

    pruned = {k: v for k, v in scores.items() if k in valid_keys}
    removed = before - len(pruned)
    if removed > 0:
        scores_file.write_text(json.dumps(pruned, indent=2))
        log_print(f"  [scores] Pruned {removed} stale entries ({before} → {len(pruned)})")
    return removed


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
        benchmark_log = args.get("benchmark_log")

        # Auto-use current_diagnosis.json if diagnose was run and no explicit file given
        if not diagnosis_file:
            auto_diag = cfg.data_dir / "current_diagnosis.json"
            if auto_diag.exists():
                diagnosis_file = str(auto_diag)
                log_print(f"  [generate_data] Using diagnosis from {auto_diag}")

        # Auto-detect benchmark log from current model if not provided
        if not benchmark_log:
            model_name = state.current_ollama_model or ""
            if model_name:
                log_dir = cfg.data_dir.parent / "logs"
                clean_name = model_name.replace("/", "_")
                candidate = log_dir / f"bench_{clean_name}.log"
                if candidate.exists():
                    benchmark_log = str(candidate)
                    log_print(f"  [generate_data] Using benchmark log: {candidate.name}")

        if not tasks:
            return {"status": "error", "error": "No tasks specified"}

        # Guard: if Claude passes a string instead of list, fix it
        if isinstance(tasks, str):
            tasks = [t.strip() for t in tasks.split(",") if t.strip()]
        # Guard: validate task IDs look correct
        tasks = [t for t in tasks if t.startswith("task_")]
        if not tasks:
            return {"status": "error", "error": "No valid task IDs (must start with 'task_')"}

        # Smart guard: use data analyzer to decide which tasks need generation
        try:
            from datagen.data_analyzer import get_task_recommendation, MAX_ADD_PER_CYCLE
            skip = []
            for t in tasks:
                rec = get_task_recommendation(t, cfg, state.to_dict())
                if rec in ("LEAVE_ALONE", "INFRASTRUCTURE", "TRIM"):
                    skip.append((t, rec))
            if skip:
                for t, rec in skip:
                    log_print(f"  [generate_data] SKIPPING {t}: analyzer says {rec}")
                tasks = [t for t in tasks if t not in [s[0] for s in skip]]
            if not tasks:
                return {"status": "success", "result": {"generated": 0, "note": "Analyzer: all tasks sufficient"}, "cost_usd": 0}

            # Per-cycle cap: max +20 per task from session start
            baseline = getattr(state, "baseline_task_counts", {}) or {}
            if baseline:
                from collections import Counter as _Ctr
                current = _Ctr()
                if cfg.train_file.exists():
                    for _line in cfg.train_file.read_text().splitlines():
                        if _line.strip():
                            try:
                                current[json.loads(_line).get("task_id", "")] += 1
                            except json.JSONDecodeError:
                                pass
                capped = []
                for t in tasks:
                    added = current.get(t, 0) - baseline.get(t, 0)
                    if added >= MAX_ADD_PER_CYCLE:
                        capped.append(t)
                        log_print(f"  [generate_data] CAPPED {t}: already +{added} this session (max {MAX_ADD_PER_CYCLE})")
                tasks = [t for t in tasks if t not in capped]
                if not tasks:
                    return {"status": "success", "result": {"generated": 0, "note": f"All tasks hit +{MAX_ADD_PER_CYCLE} cap"}, "cost_usd": 0}
        except Exception as e:
            log_print(f"  [generate_data] Analyzer failed ({e}), proceeding without guard")

        # Use dynamic_gen with pilot-validate-refine flow
        # Each example is validated against task requirements before entering the dataset
        script = str(_PROJECT_ROOT / "datagen" / "dynamic_gen.py")

        # Count before generation
        from collections import Counter as _Before
        before = _Before()
        if cfg.train_file.exists():
            for _line in cfg.train_file.read_text().splitlines():
                if _line.strip():
                    try:
                        before[json.loads(_line).get("task_id", "")] += 1
                    except json.JSONDecodeError:
                        pass

        cmd = [
            sys.executable, script, "run",
            "--tasks", ",".join(tasks),
            "--min-per-task", str(min_per_task),
        ]
        if diagnosis_file:
            cmd.extend(["--diagnosis-file", str(diagnosis_file)])
        if benchmark_log:
            cmd.extend(["--benchmark-log", str(benchmark_log)])

        rc, output = _run_script(cmd, "generate")

        # Count after generation to compute actual delta per task
        after = _Before()
        if cfg.train_file.exists():
            for _line in cfg.train_file.read_text().splitlines():
                if _line.strip():
                    try:
                        after[json.loads(_line).get("task_id", "")] += 1
                    except json.JSONDecodeError:
                        pass

        generated = {}
        total_generated = 0
        for task in tasks:
            added = after.get(task, 0) - before.get(task, 0)
            generated[task] = {"returncode": rc, "added": added}
            total_generated += added

        return {
            "status": "success",
            "result": {
                "generated": total_generated,
                "per_task": generated,
            },
            "cost_usd": total_generated * 0.04,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── score_data ───────────────────────────────────────────────────────────────

def score_data(args: dict, cfg, state) -> dict:
    """Run the LLM judge on all unscored examples."""
    try:
        # Prune stale scores before judging (prevents inflated counts)
        pruned = _prune_stale_scores(cfg)

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
                "stale_scores_pruned": pruned,
            },
            "cost_usd": new_scored * 0.01,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── filter_data ──────────────────────────────────────────────────────────────

def filter_data(args: dict, cfg, state) -> dict:
    """Filter training data below minimum judge score.

    Params:
      min_score (int): Remove examples scoring below this (default from config)
      force (bool): Bypass baseline protection — remove bad examples regardless
                    of when they were created. Use after diagnosis confirms bad data.
      tasks (list): Only filter these specific tasks (others untouched). Use with
                    force to surgically clean diagnosed tasks.

    Default mode: baseline protection prevents removing pre-session examples.
    Force mode: removes all low-scoring examples, only respecting hard floor of 15/task.
    """
    try:
        # Prune stale scores before filtering (ensures score keys match current data)
        pruned = _prune_stale_scores(cfg)

        min_score = args.get("min_score", cfg.data.min_judge_score)
        force = args.get("force", False)
        target_tasks = set(args.get("tasks", []))  # empty = all tasks
        train_file = cfg.train_file
        _min = cfg._data.get("data", {}).get("min_per_task", 30)
        HARD_FLOOR = 15  # absolute minimum — never drop below this even in force mode

        # Baseline counts from session start
        baseline = getattr(state, "baseline_task_counts", {}) or {}
        if force:
            log_print(f"  [filter_data] FORCE MODE: bypassing baseline protection, hard floor {HARD_FLOOR}/task")
            if target_tasks:
                log_print(f"  [filter_data] Targeting: {sorted(target_tasks)}")
        elif baseline:
            log_print(f"  [filter_data] Session baseline: {sum(baseline.values())} examples across {len(baseline)} tasks — will not drop below these")

        # Count per task BEFORE filtering
        from collections import Counter as _Counter
        before_counts = _Counter()
        if train_file.exists():
            for line in train_file.read_text().splitlines():
                if line.strip():
                    try:
                        before_counts[json.loads(line).get("task_id", "")] += 1
                    except json.JSONDecodeError:
                        pass

        # In default mode, protect tasks near minimum
        protected_tasks = set()
        if not force:
            protected_tasks = {t for t, c in before_counts.items() if c <= _min + 5}
            if protected_tasks:
                log_print(f"  [filter_data] PROTECTING {len(protected_tasks)} tasks near minimum: {sorted(protected_tasks)}")

        # Load scores
        scores_file = cfg.data_dir / "scores.json"
        scores = {}
        if scores_file.exists():
            scores = json.loads(scores_file.read_text())

        before = sum(before_counts.values())
        kept_lines = []
        removed_count = 0
        kept_per_task = _Counter()

        if train_file.exists():
            for line in train_file.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    ex = json.loads(line)
                    task_id = ex.get("task_id", "")

                    # If targeting specific tasks, skip others
                    if target_tasks and task_id not in target_tasks:
                        kept_lines.append(line)
                        kept_per_task[task_id] += 1
                        continue

                    # Never remove from protected tasks (default mode only)
                    if task_id in protected_tasks:
                        kept_lines.append(line)
                        kept_per_task[task_id] += 1
                        continue

                    # Check score
                    msgs = ex.get("messages", [])
                    user_msgs = [m for m in msgs if m.get("role") == "user"]
                    user_text = user_msgs[0]["content"][:80] if user_msgs else ""
                    key = f"{task_id}|{user_text}"
                    alt_key = f"{task_id}::{user_text}"

                    score_entry = scores.get(key) or scores.get(alt_key)
                    if score_entry and score_entry.get("score", 5) < min_score:
                        if force:
                            # Force mode: only respect hard floor
                            if kept_per_task[task_id] < HARD_FLOOR:
                                kept_lines.append(line)
                                kept_per_task[task_id] += 1
                            else:
                                removed_count += 1
                        else:
                            # Default mode: respect baseline
                            task_baseline = baseline.get(task_id, 0)
                            if task_baseline > 0 and kept_per_task[task_id] < task_baseline:
                                kept_lines.append(line)
                                kept_per_task[task_id] += 1
                            else:
                                removed_count += 1
                    else:
                        kept_lines.append(line)
                        kept_per_task[task_id] += 1
                except json.JSONDecodeError:
                    kept_lines.append(line)

            train_file.write_text("\n".join(kept_lines) + "\n" if kept_lines else "")

        after = len(kept_lines)
        log_print(f"  [filter_data] {before} → {after} (removed {removed_count}, protected {len(protected_tasks)} tasks)")

        result = {
            "kept": after,
            "removed": removed_count,
            "stale_scores_pruned": pruned,
            "per_task_counts": dict(kept_per_task),
        }

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


# ── compare_data ─────────────────────────────────────────────────────────────

def compare_data(args: dict, cfg, state) -> dict:
    """Compare current training data against the gold/best version from HuggingFace.

    Returns per-task diff showing additions and removals. Flags tasks where
    data was reduced from the gold baseline, especially for well-performing tasks.
    """
    try:
        from collections import Counter

        # Count current data
        current = Counter()
        train_file = cfg.data_dir / "train.jsonl"
        if train_file.exists():
            for line in train_file.read_text().splitlines():
                if line.strip():
                    try:
                        current[json.loads(line).get("task_id", "")] += 1
                    except json.JSONDecodeError:
                        pass

        # Get gold data counts from HuggingFace
        best_v = args.get("version", state.best_version)
        if not best_v or best_v <= 0:
            return {"status": "error", "error": "No best version to compare against"}

        gold = Counter()
        try:
            from huggingface_hub import HfApi, hf_hub_download
            api = HfApi()
            hf_repo = cfg.huggingface.dataset_repo

            commits = api.list_repo_commits(hf_repo, repo_type="dataset")
            target_revision = None
            for commit in commits:
                if f"Pre-v{best_v} " in (commit.title or ""):
                    target_revision = commit.commit_id
                    break

            if not target_revision:
                return {"status": "error", "error": f"No 'Pre-v{best_v}' commit on HuggingFace"}

            gold_path = hf_hub_download(hf_repo, "train.jsonl", revision=target_revision, repo_type="dataset")
            for line in open(gold_path).readlines():
                if line.strip():
                    try:
                        gold[json.loads(line).get("task_id", "")] += 1
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            return {"status": "error", "error": f"Failed to fetch gold data: {e}"}

        # Build diff with score context
        scores = state.scores or {}
        all_tasks = sorted(set(list(current.keys()) + list(gold.keys())))

        diff = []
        warnings = []
        for t in all_tasks:
            g = gold.get(t, 0)
            c = current.get(t, 0)
            delta = c - g
            bench = scores.get(t, 0)
            entry = {
                "task": t,
                "gold": g,
                "current": c,
                "delta": delta,
                "bench_score": round(bench, 2),
            }
            diff.append(entry)

            # Flag dangerous reductions
            if g > 0 and delta < 0:
                pct_lost = abs(delta) / g * 100
                if bench >= 0.70 and pct_lost > 10:
                    entry["warning"] = f"PROTECTED task lost {pct_lost:.0f}% of gold data"
                    warnings.append(f"{t}: lost {abs(delta)} examples ({pct_lost:.0f}%), bench={bench:.0%}")
                elif pct_lost > 20:
                    entry["warning"] = f"Lost {pct_lost:.0f}% of gold data"
                    warnings.append(f"{t}: lost {abs(delta)} examples ({pct_lost:.0f}%), bench={bench:.0%}")

        # Summary
        total_gold = sum(gold.values())
        total_current = sum(current.values())
        tasks_reduced = sum(1 for d in diff if d["delta"] < 0)
        tasks_added = sum(1 for d in diff if d["delta"] > 0)

        log_print(f"  [compare_data] Gold v{best_v}: {total_gold} | Current: {total_current} | Delta: {total_current - total_gold:+d}")
        log_print(f"  [compare_data] Tasks reduced: {tasks_reduced} | Tasks added to: {tasks_added}")
        if warnings:
            for w in warnings:
                log_print(f"  [compare_data] WARNING: {w}")

        return {
            "status": "success",
            "result": {
                "gold_version": best_v,
                "gold_total": total_gold,
                "current_total": total_current,
                "diff": diff,
                "warnings": warnings,
                "safe_to_train": len(warnings) == 0,
            },
            "cost_usd": 0.0,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── read_benchmark_transcript ────────────────────────────────────────────────

def read_benchmark_transcript(args: dict, cfg, state) -> dict:
    """Read the raw benchmark transcript for specific tasks.

    Returns the actual model output — what it typed, what tools it called,
    what errors it got. This is the most diagnostic signal for understanding
    WHY a task fails. Much more useful than summarized diagnoses.

    Params:
      tasks: list of task IDs to read transcripts for
      version: model version (default: current). Used to find the right log file.
      max_chars: max characters per task transcript (default: 3000)
    """
    try:
        tasks = args.get("tasks", [])
        if isinstance(tasks, str):
            tasks = [t.strip() for t in tasks.split(",") if t.strip()]
        if not tasks:
            return {"status": "error", "error": "No tasks specified"}

        max_chars = args.get("max_chars", 3000)
        version = args.get("version", state.model_version)

        # Find the benchmark log file
        log_dir = cfg.data_dir.parent / "logs"
        model_name = state.current_ollama_model or ""

        # Try version-specific log first
        log_file = None
        if model_name:
            clean_name = model_name.replace("/", "_")
            candidate = log_dir / f"bench_{clean_name}.log"
            if candidate.exists():
                log_file = candidate

        # Fall back to latest log
        if not log_file:
            logs = sorted(log_dir.glob("bench_*.log"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
            if logs:
                log_file = logs[0]

        if not log_file:
            return {"status": "error", "error": "No benchmark log found"}

        log_text = log_file.read_text(errors="replace")
        log_print(f"  [read_transcript] Reading from {log_file.name}")

        import re
        transcripts = {}
        for task_id in tasks:
            # Extract section between "starting task: {task}" and next "starting task:"
            pattern = rf'starting task: .*?{re.escape(task_id)}.*?(?=starting task:|$)'
            m = re.search(pattern, log_text, re.DOTALL)
            if m:
                raw = m.group()
                # Truncate if too long but keep the ending (score + notes)
                if len(raw) > max_chars:
                    # Keep first 1/3 and last 2/3 (the end has the score and judge notes)
                    head = raw[:max_chars // 3]
                    tail = raw[-(max_chars * 2 // 3):]
                    raw = head + "\n... [truncated] ...\n" + tail
                transcripts[task_id] = raw
                log_print(f"  [read_transcript] {task_id}: {len(raw)} chars")
            else:
                transcripts[task_id] = "(not found in log)"
                log_print(f"  [read_transcript] {task_id}: not found")

        return {
            "status": "success",
            "result": {
                "log_file": log_file.name,
                "transcripts": transcripts,
            },
            "cost_usd": 0.0,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── restore_gold_data ────────────────────────────────────────────────────────

def restore_gold_data(args: dict, cfg, state) -> dict:
    """Restore training data from the best-scoring version via HuggingFace.

    If tasks is provided, only restore those tasks from gold — keep current
    data for all other tasks. This preserves improvements while reverting
    regressions.
    """
    try:
        version = args.get("version", state.best_version)
        target_tasks = args.get("tasks", [])
        if isinstance(target_tasks, str):
            target_tasks = [t.strip() for t in target_tasks.split(",") if t.strip()]

        if not version or version <= 0:
            return {"status": "error", "error": "No best version to restore from"}

        log_print(f"  [restore_gold_data] Downloading v{version} from HuggingFace...")
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        hf_repo = cfg.huggingface.dataset_repo

        commits = api.list_repo_commits(hf_repo, repo_type="dataset")
        target_revision = None
        for commit in commits:
            if f"Pre-v{version} " in (commit.title or ""):
                target_revision = commit.commit_id
                break

        if not target_revision:
            return {"status": "error", "error": f"No 'Pre-v{version}' commit found on HuggingFace"}

        log_print(f"  [restore_gold_data] Found revision {target_revision[:8]}")

        if target_tasks:
            # Selective restore: only replace specified tasks, keep others
            log_print(f"  [restore_gold_data] Selective restore for: {target_tasks}")

            # Download gold train to temp location
            gold_path = hf_hub_download(hf_repo, "train.jsonl", revision=target_revision, repo_type="dataset")

            # Load gold examples for target tasks
            gold_by_task = {}
            for line in open(gold_path).readlines():
                if line.strip():
                    try:
                        ex = json.loads(line)
                        tid = ex.get("task_id", "")
                        if tid in target_tasks:
                            gold_by_task.setdefault(tid, []).append(line.strip())
                    except json.JSONDecodeError:
                        pass

            # Load current train, keep non-target tasks, replace target tasks with gold
            kept_lines = []
            train_file = cfg.data_dir / "train.jsonl"
            if train_file.exists():
                for line in train_file.read_text().splitlines():
                    if not line.strip():
                        continue
                    try:
                        tid = json.loads(line).get("task_id", "")
                        if tid not in target_tasks:
                            kept_lines.append(line)
                    except json.JSONDecodeError:
                        kept_lines.append(line)

            # Add gold examples for target tasks
            for tid in target_tasks:
                gold_examples = gold_by_task.get(tid, [])
                kept_lines.extend(gold_examples)
                log_print(f"  [restore_gold_data] {tid}: replaced with {len(gold_examples)} gold examples")

            train_file.write_text("\n".join(kept_lines) + "\n" if kept_lines else "")
        else:
            # Full restore: replace everything
            for fname in ["train.jsonl", "val.jsonl"]:
                path = hf_hub_download(hf_repo, fname, revision=target_revision, repo_type="dataset")
                shutil.copy2(path, str(cfg.data_dir / fname))

        # Count restored data (train + val, matching inspect_data)
        from collections import Counter
        train_count = 0
        val_count = 0
        counts = Counter()
        for fname, label in [("train.jsonl", "train"), ("val.jsonl", "val")]:
            f = cfg.data_dir / fname
            if f.exists():
                for line in f.read_text().splitlines():
                    if line.strip():
                        try:
                            counts[json.loads(line).get("task_id", "")] += 1
                            if label == "train":
                                train_count += 1
                            else:
                                val_count += 1
                        except json.JSONDecodeError:
                            pass

        # Prune scores.json to only contain entries matching restored data
        scores_file = cfg.data_dir / "scores.json"
        if scores_file.exists():
            scores = json.loads(scores_file.read_text())
            before_scores = len(scores)

            # Build set of valid keys from restored train data
            valid_keys = set()
            train_file = cfg.data_dir / "train.jsonl"
            if train_file.exists():
                for line in train_file.read_text().splitlines():
                    if not line.strip():
                        continue
                    try:
                        ex = json.loads(line)
                        task_id = ex.get("task_id", "")
                        msgs = ex.get("messages", [])
                        user_msgs = [m for m in msgs if m.get("role") == "user"]
                        user_text = user_msgs[0]["content"][:80] if user_msgs else ""
                        valid_keys.add(f"{task_id}|{user_text}")
                        valid_keys.add(f"{task_id}::{user_text}")
                    except (json.JSONDecodeError, IndexError, KeyError):
                        pass

            pruned = {k: v for k, v in scores.items() if k in valid_keys}
            scores_file.write_text(json.dumps(pruned, indent=2))
            log_print(f"  [restore_gold_data] Pruned scores.json: {before_scores} → {len(pruned)} (removed {before_scores - len(pruned)} stale entries)")

        log_print(f"  [restore_gold_data] Restored {train_count} train + {val_count} val = {sum(counts.values())} total across {len(counts)} tasks")

        return {
            "status": "success",
            "result": {
                "version": version,
                "total_examples": sum(counts.values()),
                "train": train_count,
                "val": val_count,
                "tasks": len(counts),
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
