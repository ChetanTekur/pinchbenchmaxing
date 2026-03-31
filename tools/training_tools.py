"""
Training tool implementations for PinchBench Maxing agent.

Tools: train, convert, register, validate_model, benchmark, check_disk

Each function signature: def tool_name(args: dict, cfg, state) -> dict
All subprocess calls use Popen with line-buffered stdout to avoid deadlocks.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from agents.base import log_print, _write_log, TASK_IDS

_PROJECT_ROOT = Path(__file__).parent.parent


def _run_script(cmd: list[str], label: str, env: dict | None = None) -> tuple[int, str]:
    """
    Run a subprocess with line-buffered stdout, streaming to log.
    Returns (returncode, captured_output).
    """
    merged = {**os.environ, **(env or {})}
    # Ensure PYTHONPATH includes project root
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


# ── train ────────────────────────────────────────────────────────────────────

def _check_cuda_compatibility() -> dict:
    """Verify CUDA/GPU compatibility before training. Catches mismatches early."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"ok": False, "error": "CUDA not available — no GPU detected"}

        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        torch_version = torch.__version__

        # Quick test: try allocating a small tensor on GPU
        try:
            t = torch.zeros(1, device="cuda")
            del t
        except RuntimeError as e:
            return {
                "ok": False,
                "error": f"CUDA kernel error on {gpu_name} (CUDA {cuda_version}, "
                         f"PyTorch {torch_version}): {e}. "
                         f"The Docker image may not support this GPU. "
                         f"Try: pip install --force-reinstall torch --index-url "
                         f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')[:3]}"
            }

        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        log_print(f"  [preflight] GPU: {gpu_name} ({vram_gb:.0f} GB)")
        log_print(f"  [preflight] CUDA: {cuda_version}, PyTorch: {torch_version}")

        return {"ok": True, "gpu": gpu_name, "vram_gb": round(vram_gb, 1),
                "cuda": cuda_version, "torch": torch_version}
    except Exception as e:
        return {"ok": False, "error": f"GPU check failed: {e}"}


def _cleanup_old_ollama_models(state) -> list[str]:
    """Remove old Ollama models, keeping only the current and best-ever versions."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return []

        # Parse ollama list output: NAME  ID  SIZE  MODIFIED
        lines = result.stdout.strip().split("\n")[1:]  # skip header
        models = []
        for line in lines:
            parts = line.split()
            if parts:
                models.append(parts[0])  # model name (e.g. qwen35-9b-clawd-v25:latest)

        # Determine which versions to keep
        keep_versions = set()
        if state.model_version:
            keep_versions.add(state.model_version)
        if state.best_version:
            keep_versions.add(state.best_version)

        removed = []
        for model in models:
            # Extract version number from model name (e.g. "qwen35-9b-clawd-v25:latest" -> 25)
            m = re.search(r'-v(\d+)', model)
            if m:
                v = int(m.group(1))
                if v not in keep_versions:
                    rm_result = subprocess.run(
                        ["ollama", "rm", model], capture_output=True, text=True, timeout=30
                    )
                    if rm_result.returncode == 0:
                        removed.append(model)
                        log_print(f"  [train] Removed old model: {model}")

        return removed
    except Exception as e:
        log_print(f"  [train] Ollama cleanup failed: {e}")
        return []


def _check_data_coverage(cfg) -> dict:
    """
    HARD GATE: Refuse to train if data coverage is insufficient.
    This is NOT a suggestion — training literally cannot proceed without this.
    """
    train_file = cfg.train_file
    if not train_file.exists():
        return {"ok": False, "error": "train.jsonl does not exist"}

    import json
    from collections import Counter
    counts = Counter()
    for line in train_file.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rec = json.loads(line)
                counts[rec.get("task_id", "unknown")] += 1
            except json.JSONDecodeError:
                pass

    min_per_task = cfg._data.get("data", {}).get("min_per_task", 40)
    missing = [t for t in TASK_IDS if counts.get(t, 0) == 0]
    below_min = {t: counts.get(t, 0) for t in TASK_IDS
                 if 0 < counts.get(t, 0) < min_per_task}

    errors = []
    if missing:
        errors.append(
            f"BLOCKED: {len(missing)} tasks have ZERO examples: {missing}. "
            f"Generate data for these tasks before training."
        )
    if below_min:
        errors.append(
            f"BLOCKED: {len(below_min)} tasks below minimum ({min_per_task}): "
            f"{below_min}. Generate more data for these tasks."
        )

    if errors:
        # Include actionable fix so orchestrator knows exactly what to do
        all_tasks_to_fix = missing + list(below_min.keys())
        return {
            "ok": False,
            "error": " | ".join(errors),
            "fix": {
                "tool": "generate_data",
                "tasks": all_tasks_to_fix,
                "min_per_task": min_per_task,
            },
        }

    return {"ok": True, "task_counts": dict(counts)}


def _check_data_quality(cfg) -> dict:
    """
    Quality gate: allow training if data is ≥90% clean.
    Only blocks on truly critical issues (invalid tool names, malformed JSON).
    Missing args and missing required tools are warnings, not blockers.
    """
    try:
        from datagen.validate_data import run_validation
        report = run_validation(fix=False)
        critical = report.get("critical_high", 0)
        total = report.get("total_examples", 0)
        clean = report.get("clean", 0)

        if total == 0:
            return {"ok": False, "error": "BLOCKED: no training data found"}

        clean_pct = clean / total * 100
        if clean_pct < 90:
            return {
                "ok": False,
                "error": (
                    f"BLOCKED: only {clean_pct:.0f}% clean ({clean}/{total}). "
                    f"Need ≥90%. {critical} critical/high issues found."
                ),
                "critical_count": critical,
            }

        if critical > 0:
            log_print(f"  [train] WARNING: {critical} critical/high issues in {total} examples "
                      f"({clean_pct:.0f}% clean) — proceeding since ≥90%")

        return {"ok": True, "clean": clean, "total": total, "clean_pct": round(clean_pct, 1)}
    except Exception as e:
        # If validation itself fails, don't block training
        log_print(f"  [train] Warning: data quality check failed: {e}")
        return {"ok": True}


def train(args: dict, cfg, state) -> dict:
    """Fine-tune the model: prepare SFT data, run Unsloth LoRA training."""
    try:
        # HARD GATE 1: check data coverage — cannot be bypassed
        coverage = _check_data_coverage(cfg)
        if not coverage["ok"]:
            log_print(f"  [train] {coverage['error']}")
            return {"status": "error", "error": coverage["error"]}

        # HARD GATE 2: check data quality — cannot be bypassed
        quality = _check_data_quality(cfg)
        if not quality["ok"]:
            log_print(f"  [train] {quality['error']}")
            return {"status": "error", "error": quality["error"]}

        # HARD GATE 3: gold data diff — block if well-performing tasks lost data
        if state.best_version > 0 and state.scores:
            from .data_tools import compare_data as _compare
            diff_result = _compare({"version": state.best_version}, cfg, state)
            if diff_result.get("status") == "success":
                warnings = diff_result["result"].get("warnings", [])
                if warnings:
                    warn_text = "; ".join(warnings[:5])
                    log_print(f"  [train] DATA SAFETY WARNING: {warn_text}")
                    return {
                        "status": "error",
                        "error": (
                            f"BLOCKED: Gold data integrity check failed. "
                            f"{len(warnings)} task(s) lost significant data vs gold v{state.best_version}: "
                            f"{warn_text}. "
                            f"Call compare_data to see full diff. Restore gold data or acknowledge the changes."
                        ),
                    }

        # Snapshot data distribution to log — permanent record of what each version trained on
        from collections import Counter as _Counter
        _snap = _Counter()
        if cfg.train_file.exists():
            for _line in cfg.train_file.read_text().splitlines():
                if _line.strip():
                    try:
                        _snap[json.loads(_line).get("task_id", "")] += 1
                    except json.JSONDecodeError:
                        pass
        version = args.get("version", "?")
        log_print(f"  [train] DATA SNAPSHOT (v{version}):")
        log_print(f"  [train]   Total: {sum(_snap.values())} examples")
        for tid in sorted(_snap.keys()):
            log_print(f"  [train]   {tid}: {_snap[tid]}")
        # Build changelog from action history — what data changes were made this session
        changelog = []
        for action in state.action_history:
            act = action.get("action", "")
            summary = action.get("result_summary", "")[:100]
            if act in ("generate_data", "filter_data",
                       "dedup_data", "rebalance_data", "validate_data"):
                changelog.append(f"{act}: {summary}")

        # Save snapshot + changelog for cross-version comparison
        snap_file = cfg.data_dir / f"data_snapshot_v{version}.json"
        snap_file.write_text(json.dumps({
            "version": version,
            "total": sum(_snap.values()),
            "per_task": dict(_snap),
            "changelog": changelog,
        }, indent=2))
        log_print(f"  [train]   Saved: {snap_file}")

        # HARD GATE 3: check disk space (auto-clean old Ollama models if needed)
        root_free = shutil.disk_usage("/").free / (1024**3)
        if root_free < 15:
            log_print(f"  [train] Low disk: {root_free:.1f} GB free. Cleaning old Ollama models...")
            cleaned = _cleanup_old_ollama_models(state)
            root_free = shutil.disk_usage("/").free / (1024**3)
            if cleaned:
                log_print(f"  [train] After cleanup: {root_free:.1f} GB free")
            if root_free < 15:
                return {
                    "status": "error",
                    "error": f"BLOCKED: Only {root_free:.1f} GB free on root (need >=15). "
                             f"Tried auto-cleaning old Ollama models but still not enough. "
                             f"Manually free space on root partition.",
                }

        # Preflight: check CUDA compatibility before spending time
        cuda_check = _check_cuda_compatibility()
        if not cuda_check["ok"]:
            return {"status": "error", "error": cuda_check["error"]}

        version = args["version"]
        base_name = cfg._data["model"]["name"]  # unversioned base name
        versioned_name = f"{base_name}-v{version}"

        log_print(f"  [train] Training {versioned_name}")

        # Set PBM_MODEL_NAME so all derived paths use versioned name
        train_env = {"PBM_MODEL_NAME": versioned_name}

        start_time = time.time()

        # Stage 1: prepare SFT data
        rc, output = _run_script(
            [sys.executable, "-m", "stages.prepare"],
            "train:prepare",
            env=train_env,
        )
        if rc != 0:
            return {"status": "error", "error": f"prepare stage failed (exit {rc})"}

        # Stage 2: fine-tune
        rc, output = _run_script(
            [sys.executable, "-m", "stages.finetune"],
            "train:finetune",
            env=train_env,
        )
        if rc != 0:
            return {"status": "error", "error": f"finetune stage failed (exit {rc})"}

        duration_minutes = round((time.time() - start_time) / 60, 1)

        # Extract final loss from output
        loss_final = None
        for line in reversed(output.splitlines()):
            m = re.search(r"'loss':\s*([0-9.]+)", line)
            if m:
                loss_final = float(m.group(1))
                break

        # Update state with new model version
        state.record_model(version, versioned_name)
        log_print(f"  [train] Done: {versioned_name}, loss={loss_final}, {duration_minutes}min")

        return {
            "status": "success",
            "result": {
                "model_name": versioned_name,
                "loss_final": loss_final,
                "duration_minutes": duration_minutes,
            },
            "cost_usd": 0.0,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── convert ──────────────────────────────────────────────────────────────────

def convert(args: dict, cfg, state) -> dict:
    """Convert a fine-tuned model to GGUF format."""
    try:
        cuda_check = _check_cuda_compatibility()
        if not cuda_check["ok"]:
            return {"status": "error", "error": cuda_check["error"]}

        version = args["version"]
        base_name = cfg._data["model"]["name"]
        versioned_name = f"{base_name}-v{version}"

        convert_env = {"PBM_MODEL_NAME": versioned_name}

        rc, output = _run_script(
            [sys.executable, "-m", "stages.convert"],
            "convert",
            env=convert_env,
        )
        if rc != 0:
            return {"status": "error", "error": f"convert stage failed (exit {rc})"}

        # Check GGUF file exists — temporarily set env for cfg resolution
        old_val = os.environ.get("PBM_MODEL_NAME")
        os.environ["PBM_MODEL_NAME"] = versioned_name
        gguf_path = cfg.gguf_file
        if old_val is None:
            os.environ.pop("PBM_MODEL_NAME", None)
        else:
            os.environ["PBM_MODEL_NAME"] = old_val

        size_mb = 0
        if gguf_path.exists():
            size_mb = round(gguf_path.stat().st_size / 1024 / 1024, 1)
        else:
            return {"status": "error", "error": f"GGUF not found at {gguf_path}"}

        return {
            "status": "success",
            "result": {
                "gguf_path": str(gguf_path),
                "size_mb": size_mb,
            },
            "cost_usd": 0.0,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── register ─────────────────────────────────────────────────────────────────

def register(args: dict, cfg, state) -> dict:
    """Register a GGUF model in Ollama with correct chat template."""
    try:
        version = args["version"]
        model_name = args["model_name"]

        base_name = cfg._data["model"]["name"]
        versioned_name = f"{base_name}-v{version}"

        # Resolve GGUF path
        old_val = os.environ.get("PBM_MODEL_NAME")
        os.environ["PBM_MODEL_NAME"] = versioned_name
        gguf_path = str(cfg.gguf_file)
        if old_val is None:
            os.environ.pop("PBM_MODEL_NAME", None)
        else:
            os.environ["PBM_MODEL_NAME"] = old_val

        script = str(_PROJECT_ROOT / "scripts" / "register_model.sh")

        rc, output = _run_script(
            ["bash", script],
            "register",
            env={"OLLAMA_MODEL": model_name, "GGUF_PATH": gguf_path},
        )

        if rc != 0:
            return {"status": "error", "error": f"register_model.sh failed (exit {rc})"}

        return {
            "status": "success",
            "result": {
                "registered": True,
                "ollama_name": model_name,
            },
            "cost_usd": 0.0,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── validate_model ───────────────────────────────────────────────────────────

def validate_model(args: dict, cfg, state) -> dict:
    """Validate a model by running probe prompts against it."""
    try:
        version = args.get("version")
        base_name = cfg._data["model"]["name"]
        versioned_name = f"{base_name}-v{version}" if version else cfg.model_name

        validate_env = {"PBM_MODEL_NAME": versioned_name}

        rc, output = _run_script(
            [sys.executable, "-m", "stages.validate_model"],
            "validate_model",
            env=validate_env,
        )

        return {
            "status": "success" if rc == 0 else "error",
            "result": {
                "valid": rc == 0,
                "output": output[-2000:] if len(output) > 2000 else output,
            },
            "cost_usd": 0.0,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── benchmark ────────────────────────────────────────────────────────────────

def benchmark(args: dict, cfg, state) -> dict:
    """Run PinchBench benchmark against a model."""
    try:
        model_name = args["model_name"]
        ollama_model = f"ollama/{model_name}"

        script = str(_PROJECT_ROOT / "scripts" / "benchmark_run.sh")

        rc, output = _run_script(
            ["bash", script, ollama_model],
            "benchmark",
            env={"PBM_WORKSPACE": str(cfg.workspace)},
        )

        # Find the log file
        safe_name = ollama_model.replace("/", "_").replace(":", "_").replace(" ", "_")
        log_path = cfg.data_dir.parent / "logs" / f"bench_{safe_name}.log"

        # Parse scores from log
        scores = {}
        text = ""
        if log_path.exists():
            text = log_path.read_text(errors="replace")
        else:
            text = output

        for m in re.finditer(r'Task (task_\w+):\s*([01](?:\.\d+)?)\s*/\s*1\.0', text):
            if m.group(1) in TASK_IDS:
                scores[m.group(1)] = float(m.group(2))

        # Fallback: try JSON blobs
        if not scores:
            for blob in re.findall(r'\{[^{}]*"task_\d{2}_\w+"[^{}]*\}', text):
                try:
                    obj = json.loads(blob)
                    for k, v in obj.items():
                        if k in TASK_IDS and isinstance(v, (int, float)):
                            scores[k] = float(v)
                except json.JSONDecodeError:
                    pass

        avg_score = round(sum(scores.values()) / len(scores), 4) if scores else 0.0

        # Update state with benchmark results
        if scores:
            state.record_eval(scores)
            state.weak_tasks = [t for t in TASK_IDS if t in scores
                                and scores[t] < 0.5]  # threshold from config ideally

        # Save per-version artifacts for cross-version comparison
        if scores:
            version = state.model_version
            versions_dir = cfg.data_dir.parent / "versions" / f"v{version}"
            versions_dir.mkdir(parents=True, exist_ok=True)
            # Save scores
            (versions_dir / "scores.json").write_text(json.dumps({
                "version": version,
                "model": model_name,
                "avg_score": avg_score,
                "scores": scores,
            }, indent=2))
            # Symlink benchmark log
            log_link = versions_dir / "benchmark.log"
            try:
                if log_link.is_symlink() or log_link.exists():
                    log_link.unlink()
                if log_path.exists():
                    log_link.symlink_to(log_path)
            except OSError:
                pass
            # Copy data snapshot if it exists
            snap = cfg.data_dir / f"data_snapshot_v{version}.json"
            if snap.exists():
                import shutil
                shutil.copy2(str(snap), str(versions_dir / "data_snapshot.json"))
            log_print(f"  [benchmark] Artifacts saved to {versions_dir}")

        return {
            "status": "success" if scores else "error",
            "result": {
                "scores": scores,
                "avg_score": avg_score,
                "log_path": str(log_path),
            },
            "cost_usd": 0.0,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── check_disk ───────────────────────────────────────────────────────────────

def check_disk(args: dict, cfg, state) -> dict:
    """Check available disk space on workspace and root partitions."""
    try:
        # Workspace partition
        workspace_path = cfg.workspace
        if not workspace_path.exists():
            workspace_path = Path("/workspace")

        try:
            ws_usage = shutil.disk_usage(str(workspace_path))
            workspace_free_gb = round(ws_usage.free / (1024 ** 3), 2)
        except OSError:
            workspace_free_gb = -1

        # Root partition
        root_usage = shutil.disk_usage("/")
        root_free_gb = round(root_usage.free / (1024 ** 3), 2)

        warning = workspace_free_gb < 10 or root_free_gb < 5

        return {
            "status": "success",
            "result": {
                "workspace_free_gb": workspace_free_gb,
                "root_free_gb": root_free_gb,
                "warning": warning,
            },
            "cost_usd": 0.0,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
