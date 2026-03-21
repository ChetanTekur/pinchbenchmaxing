#!/usr/bin/env python3
"""
Agentic Orchestrator — Claude decides what to do next.

Replaces the fixed pipeline with a Claude-powered decision loop.
Each turn: Claude sees the state → calls a tool → state updates → repeat.

Usage:
  python orchestrator.py run                          # start/resume
  python orchestrator.py run --model qwen35-9b-clawd-v7  # seed model version
  python orchestrator.py run --dry-run                # Claude decides but tools don't execute
  python orchestrator.py status                       # show state
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import anthropic

from utils.config import load_config
from agents.base import AgentState, setup_file_logger, log_print

PROJECT_ROOT = Path(__file__).parent
STATE_FILE_NAME = "loop_state.json"


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────

def load_state(state_file: Path) -> AgentState:
    if state_file.exists():
        return AgentState.from_dict(json.loads(state_file.read_text()))
    return AgentState()


def save_state(state: AgentState, state_file: Path) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state.to_dict(), indent=2))


def _format_result(tool_name: str, r: dict) -> str:
    """Format tool results in a human-readable way."""
    if not isinstance(r, dict):
        return str(r)[:200]

    if tool_name == "inspect_data":
        total = r.get("total", "?")
        overweight = r.get("overweight", [])
        underweight = r.get("underweight", [])
        parts = [f"{total} examples"]
        if overweight:
            parts.append(f"overweight: {overweight}")
        if underweight:
            parts.append(f"underweight: {underweight}")
        return " | ".join(parts)

    elif tool_name == "benchmark":
        avg = r.get("avg_score", "?")
        scores = r.get("scores", {})
        zeros = [t for t, s in scores.items() if s == 0.0]
        return f"avg={avg} | {len(scores)} tasks scored | {len(zeros)} at zero: {zeros}"

    elif tool_name == "generate_data":
        total = r.get("generated", 0)
        per = r.get("per_task", {})
        return f"{total} examples generated across {len(per)} tasks"

    elif tool_name == "generate_adversarial":
        total = r.get("generated", 0)
        per = r.get("per_task", {})
        return f"{total} adversarial examples across {len(per)} tasks"

    elif tool_name == "train":
        name = r.get("model_name", "?")
        loss = r.get("loss_final", "?")
        dur = r.get("duration_minutes", "?")
        return f"{name} | final loss: {loss} | {dur} min"

    elif tool_name == "convert":
        path = r.get("gguf_path", "?")
        size = r.get("size_mb", "?")
        return f"GGUF: {size} MB"

    elif tool_name == "score_data":
        return f"scored {r.get('total_scored', '?')} examples ({r.get('new_scored', 0)} new)"

    elif tool_name == "filter_data":
        return f"kept {r.get('kept', '?')}, removed {r.get('removed', '?')}"

    elif tool_name == "dedup_data":
        return f"before={r.get('before', '?')}, after={r.get('after', '?')}, removed {r.get('removed', '?')} ({r.get('percent', '?')}%)"

    elif tool_name == "rebalance_data":
        return f"before={r.get('before', '?')}, after={r.get('after', '?')}, trimmed {r.get('trimmed', '?')}"

    elif tool_name == "validate_data":
        clean = r.get("clean", "?")
        total = r.get("total_examples", "?")
        critical = r.get("critical_high", 0)
        ready = r.get("ready_for_training", False)
        return f"{clean}/{total} clean | {critical} critical | ready={ready}"

    elif tool_name == "diagnose":
        summary = r.get("summary", "")[:150]
        n_causes = len(r.get("root_causes", []))
        n_fixes = len(r.get("data_fixes", []))
        return f"{n_causes} root causes, {n_fixes} data fixes | {summary}"

    elif tool_name == "plan_strategy":
        plan = r.get("plan", [])
        total = r.get("total_examples", 0)
        return f"{len(plan)} tasks planned, {total} examples total"

    elif tool_name == "check_disk":
        wfree = r.get("workspace_free_gb", "?")
        rfree = r.get("root_free_gb", "?")
        warn = " WARNING: LOW SPACE" if r.get("warning") else ""
        return f"workspace: {wfree} GB free, root: {rfree} GB free{warn}"

    elif tool_name == "snapshot":
        return f"saved to {r.get('path', '?')}"

    elif tool_name == "push_hf":
        return f"pushed {r.get('files_pushed', '?')} files to {r.get('repo', '?')}"

    # Fallback
    return ", ".join(f"{k}={v}" for k, v in list(r.items())[:5])[:200]


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT BUILDING
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt(cfg) -> str:
    """Load orchestrator prompt template and fill from config.
    Uses $var syntax (string.Template) so curly braces in the markdown
    don't conflict with Python's str.format()."""
    from string import Template
    template_path = PROJECT_ROOT / "prompts" / "orchestrator.md"
    raw = template_path.read_text()

    # Convert {var} placeholders to $var for string.Template
    # Only replace known variables, leave everything else untouched
    variables = {
        "benchmark_name": cfg.benchmark.name,
        "benchmark_url": getattr(cfg.benchmark, 'url', 'https://pinchbench.com'),
        "total_tasks": str(cfg.benchmark.total_tasks),
        "target_score": f"{cfg.loop.target_score:.0%}",
        "model_base": cfg.base_model,
        "model_name": cfg.model_name,
        "max_new_per_task": str(cfg.loop.max_new_per_task),
        "max_total_per_task": str(cfg.loop.max_total_per_task),
        "total_new_cap": str(cfg.loop.total_new_examples_cap),
        "budget_usd": str(cfg.orchestrator.budget_usd),
        "gpu_rate": str(cfg.orchestrator.gpu_rate_per_hour),
    }

    # Simple replacement — handles {var} without choking on other braces
    result = raw
    for key, value in variables.items():
        result = result.replace("{" + key + "}", value)
        # Also handle format specs like {target_score:.0%}
        result = result.replace("{" + key + ":.0%}", value)

    return result


def build_turn_context(state: AgentState, cfg) -> str:
    """Build the user message for this turn — fresh state + compressed history."""
    # Current scores
    scores_summary = ""
    if state.scores:
        weak = [f"{t}: {s:.2f}" for t, s in sorted(state.scores.items()) if s < cfg.loop.target_score]
        strong = [t for t, s in state.scores.items() if s >= cfg.loop.target_score]
        scores_summary = (
            f"Weak tasks ({len(weak)}):\n"
            + "\n".join(f"  {w}" for w in weak)
            + f"\n\nStrong tasks ({len(strong)}): {', '.join(sorted(strong))}"
        )

    # Action history (compress old entries)
    history_text = ""
    history = state.action_history or []
    if len(history) > 10:
        old = history[:-5]
        recent = history[-5:]
        old_summary = ", ".join(f"{a['action']}" for a in old)
        history_text = f"Earlier actions: {old_summary}\n\nRecent:\n"
        for a in recent:
            history_text += f"  Turn {a['turn']}: {a['action']} → {a.get('result_summary', '?')}\n"
    else:
        for a in history:
            history_text += f"  Turn {a['turn']}: {a['action']} → {a.get('result_summary', '?')}\n"

    budget_remaining = cfg.orchestrator.budget_usd - state.budget_spent_usd

    return f"""## Current State

Model: v{state.model_version} ({state.current_ollama_model or 'none'})
Score: {state.avg_score:.3f} ({state.avg_score*100:.1f}%)
Best ever: {state.best_avg_score:.3f} (v{state.best_version})
Target: {cfg.loop.target_score}
Iteration: {state.iteration}
Budget remaining: ${budget_remaining:.2f}

## Scores
{scores_summary or '(no scores yet)'}

## Action History
{history_text or '(first turn — no actions yet)'}

## What should we do next?
Examine the state above and take ONE action. Call a tool, or respond with "DONE: <reason>" to end the session.
"""


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_orchestrator(cfg, state: AgentState, state_file: Path, dry_run: bool = False):
    from tools.registry import TOOL_SCHEMAS, execute_tool

    # ── Preflight checks — verify environment before any work ────────────
    log_print(f"\n{'='*62}")
    log_print(f"  ORCHESTRATOR AGENT — PREFLIGHT")
    log_print(f"{'='*62}")

    # 1. API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log_print("[PREFLIGHT] FAIL: ANTHROPIC_API_KEY not set")
        sys.exit(1)
    log_print("[PREFLIGHT] OK: ANTHROPIC_API_KEY set")

    # 2. CUDA / GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            cuda_ver = torch.version.cuda
            # Test kernel compatibility
            try:
                t = torch.zeros(1, device="cuda")
                del t
                log_print(f"[PREFLIGHT] OK: GPU {gpu_name} ({vram_gb:.0f} GB, CUDA {cuda_ver})")
            except RuntimeError as e:
                log_print(f"[PREFLIGHT] FAIL: GPU {gpu_name} CUDA kernel incompatible")
                log_print(f"  Error: {e}")
                log_print(f"  Docker image CUDA doesn't match this GPU.")
                log_print(f"  Fix: pip install --force-reinstall torch --index-url "
                          f"https://download.pytorch.org/whl/cu{cuda_ver.replace('.','')[:3]}")
                sys.exit(1)
        else:
            log_print("[PREFLIGHT] WARN: No CUDA GPU detected — training will fail")
    except ImportError:
        log_print("[PREFLIGHT] WARN: PyTorch not installed — training will fail")

    # 3. Disk space
    import shutil
    workspace = cfg.workspace
    if workspace.exists():
        free_gb = shutil.disk_usage(str(workspace)).free / (1024**3)
        if free_gb < 10:
            log_print(f"[PREFLIGHT] FAIL: Only {free_gb:.1f} GB free on workspace (need ≥10)")
            sys.exit(1)
        log_print(f"[PREFLIGHT] OK: {free_gb:.0f} GB free on workspace")
    else:
        log_print(f"[PREFLIGHT] WARN: Workspace {workspace} not found")

    # 4. Ollama running
    try:
        import httpx
        r = httpx.get("http://127.0.0.1:11434/", timeout=5)
        log_print("[PREFLIGHT] OK: Ollama running")
    except Exception:
        log_print("[PREFLIGHT] WARN: Ollama not running — benchmark will fail")

    # 5. OpenClaw gateway
    try:
        r = httpx.get("http://127.0.0.1:18789/health", timeout=5)
        log_print("[PREFLIGHT] OK: OpenClaw gateway running")
    except Exception:
        log_print("[PREFLIGHT] WARN: OpenClaw gateway not running — benchmark will fail")

    # 6. Data files exist
    if cfg.train_file.exists():
        n_train = sum(1 for l in cfg.train_file.read_text().splitlines() if l.strip())
        log_print(f"[PREFLIGHT] OK: train.jsonl exists ({n_train} examples)")
    else:
        log_print("[PREFLIGHT] INFO: No train.jsonl yet — will need to generate data")

    log_print(f"[PREFLIGHT] All checks passed")
    log_print("")

    # ── Start orchestrator ─────────────────────────────────────────────────
    client = anthropic.Anthropic(api_key=api_key)
    model = cfg.orchestrator.model
    max_actions = cfg.orchestrator.max_actions
    budget_total = cfg.orchestrator.budget_usd

    system_prompt = build_system_prompt(cfg)
    consecutive_failures = 0

    log_print(f"{'='*62}")
    log_print(f"  ORCHESTRATOR AGENT")
    log_print(f"  Claude model : {model}")
    log_print(f"  Budget       : ${budget_total}")
    log_print(f"  Max actions  : {max_actions}")
    log_print(f"  Dry run      : {dry_run}")
    log_print(f"  Target       : {cfg.loop.target_score:.0%}")
    log_print(f"  Current score: {state.avg_score:.1%}")
    log_print(f"{'='*62}")

    for turn in range(1, max_actions + 1):
        budget_remaining = budget_total - state.budget_spent_usd

        # ── Guardrails ─────────────────────────────────────────────────────
        if budget_remaining < 5.0:
            log_print(f"\n[ORCHESTRATOR AGENT] Budget exhausted (${budget_remaining:.2f} remaining). Stopping.")
            break

        if consecutive_failures >= cfg.orchestrator.auto_pause.max_consecutive_failures:
            log_print(f"\n[ORCHESTRATOR AGENT] {consecutive_failures} consecutive failures. Stopping.")
            break

        # ── Call Claude ────────────────────────────────────────────────────
        turn_context = build_turn_context(state, cfg)

        log_print(f"\n{'─'*62}")
        log_print(f"  TURN {turn}/{max_actions}  (budget: ${budget_remaining:.2f})")
        log_print(f"{'─'*62}")

        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": turn_context}],
                tools=TOOL_SCHEMAS,
            )
        except Exception as e:
            log_print(f"[ORCHESTRATOR AGENT] Claude API error: {e}")
            consecutive_failures += 1
            continue

        # Track orchestrator API cost (~$0.03 per turn)
        state.budget_spent_usd += 0.03

        # ── Process response ───────────────────────────────────────────────
        # Check if Claude returned text (DONE signal) or a tool call
        tool_use_block = None
        text_block = None

        for block in response.content:
            if block.type == "tool_use":
                tool_use_block = block
            elif block.type == "text":
                text_block = block

        # Check for DONE signal
        if text_block and "DONE" in (text_block.text or ""):
            log_print(f"\n[ORCHESTRATOR AGENT] {text_block.text}")
            state.action_history.append({
                "turn": turn,
                "action": "DONE",
                "result_summary": text_block.text[:200],
                "cost_usd": 0,
                "timestamp": datetime.utcnow().isoformat(),
            })
            save_state(state, state_file)
            break

        if not tool_use_block:
            # Claude responded with text but no DONE and no tool call
            if text_block:
                for line in text_block.text.strip().splitlines():
                    log_print(f"[ORCHESTRATOR AGENT] {line}")
            continue

        # ── Execute tool ───────────────────────────────────────────────────
        tool_name = tool_use_block.name
        tool_args = tool_use_block.input or {}

        # Show Claude's reasoning if it explained before calling the tool
        if text_block and text_block.text and text_block.text.strip():
            thinking = text_block.text.strip()
            # Show full reasoning, not truncated
            for line in thinking.splitlines():
                log_print(f"[ORCHESTRATOR AGENT] Thinking: {line}")

        args_str = json.dumps(tool_args, default=str)
        if len(args_str) > 100:
            args_str = args_str[:100] + "..."
        log_print(f"[ORCHESTRATOR AGENT] Action: {tool_name}({args_str})")

        if dry_run:
            log_print(f"[ORCHESTRATOR AGENT] DRY RUN — skipping execution")
            result = {"status": "dry_run", "result": {}, "cost_usd": 0}
        else:
            try:
                result = execute_tool(tool_name, tool_args, cfg, state)
            except Exception as e:
                result = {"status": "error", "error": str(e), "cost_usd": 0}

        # ── Log result ─────────────────────────────────────────────────────
        status = result.get("status", "unknown")
        cost = result.get("cost_usd", 0)
        state.budget_spent_usd += cost

        if status == "error":
            consecutive_failures += 1
            error_msg = result.get("error", "unknown error")
            log_print(f"[ORCHESTRATOR AGENT] FAILED: {error_msg[:300]}")
            result_summary = f"ERROR: {error_msg[:200]}"
        else:
            consecutive_failures = 0
            r = result.get("result", {})
            result_summary = _format_result(tool_name, r)
            log_print(f"[ORCHESTRATOR AGENT] Result: {result_summary}")

        if cost > 0:
            log_print(f"[ORCHESTRATOR AGENT] Cost: ${cost:.2f}")

        # ── Update state ───────────────────────────────────────────────────
        state.action_history.append({
            "turn": turn,
            "action": tool_name,
            "args": {k: str(v)[:50] for k, v in tool_args.items()},  # truncate for storage
            "result_summary": result_summary[:200],
            "status": status,
            "cost_usd": cost,
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Some tools update state directly (benchmark updates scores, train updates version)
        # The tool implementations handle this via the state object passed to them

        save_state(state, state_file)

    # ── Session summary ────────────────────────────────────────────────────
    log_print(f"\n{'='*62}")
    log_print(f"  ORCHESTRATOR AGENT — SESSION COMPLETE")
    log_print(f"  Turns used   : {len(state.action_history)}/{max_actions}")
    log_print(f"  Budget spent : ${state.budget_spent_usd:.2f} / ${budget_total}")
    log_print(f"  Score        : {state.avg_score:.1%}")
    log_print(f"{'='*62}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Agentic orchestrator for PinchBench")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run the orchestrator")
    run_p.add_argument("--model", type=str, default="",
                       help="Set current model (e.g. qwen35-9b-clawd-v7)")
    run_p.add_argument("--scores", type=str, default="",
                       help="Seed scores as JSON")
    run_p.add_argument("--log", type=str, default="",
                       help="Seed scores from benchmark log file")
    run_p.add_argument("--dry-run", action="store_true",
                       help="Claude decides but tools don't execute")

    sub.add_parser("status", help="Show current state")

    args = parser.parse_args()

    if args.command == "run":
        # Initialize logger
        log_dir = cfg.data_dir.parent / "logs"
        setup_file_logger(log_dir)

        state_file = cfg.data_dir / STATE_FILE_NAME
        state = load_state(state_file)

        # Auto-detect model change — if config.yaml model differs from what's
        # in state, the user switched models. Reset to avoid stale history.
        raw_state = state.to_dict()
        old_base = raw_state.get("base_model", "")
        if old_base and old_base != cfg.base_model:
            log_print(f"[ORCHESTRATOR AGENT] Model changed: {old_base} → {cfg.base_model}")
            log_print(f"[ORCHESTRATOR AGENT] Resetting state for new model")
            state = AgentState()

        # Stamp which base model this state belongs to
        state.base_model = cfg.base_model

        # Seed model
        if args.model:
            state.current_ollama_model = args.model
            m = re.search(r'-v(\d+)$', args.model)
            if m:
                state.model_version = int(m.group(1))
            log_print(f"[ORCHESTRATOR AGENT] Model: {args.model} (v{state.model_version})")

        # Seed scores
        if args.scores:
            from loop import parse_scores_from_json_str
            seeded = parse_scores_from_json_str(args.scores)
            state.record_eval(seeded)
            log_print(f"[ORCHESTRATOR AGENT] Seeded {len(seeded)} scores")
        elif args.log:
            from loop import parse_scores_from_log
            seeded = parse_scores_from_log(args.log)
            state.record_eval(seeded)
            log_print(f"[ORCHESTRATOR AGENT] Seeded {len(seeded)} scores from {args.log}")

        # Clear action history for new session (keeps budget tracking)
        state.action_history = []

        save_state(state, state_file)
        run_orchestrator(cfg, state, state_file, dry_run=args.dry_run)

    elif args.command == "status":
        state_file = cfg.data_dir / STATE_FILE_NAME
        if not state_file.exists():
            print("No state found. Run: python orchestrator.py run")
            return
        state = load_state(state_file)
        print(f"\nModel: v{state.model_version} ({state.current_ollama_model})")
        print(f"Score: {state.avg_score:.3f} ({state.avg_score*100:.1f}%)")
        print(f"Best: {state.best_avg_score:.3f} (v{state.best_version})")
        print(f"Budget spent: ${state.budget_spent_usd:.2f}")
        print(f"Actions this session: {len(state.action_history)}")
        if state.action_history:
            print(f"\nRecent actions:")
            for a in state.action_history[-5:]:
                print(f"  Turn {a['turn']}: {a['action']} → {a.get('result_summary', '?')[:80]}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
