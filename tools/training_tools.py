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


def train(args: dict, cfg, state) -> dict:
    """Fine-tune the model: prepare SFT data, run Unsloth LoRA training."""
    try:
        # HARD GATE: check data coverage — cannot be bypassed
        coverage = _check_data_coverage(cfg)
        if not coverage["ok"]:
            log_print(f"  [train] {coverage['error']}")
            return {"status": "error", "error": coverage["error"]}

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
