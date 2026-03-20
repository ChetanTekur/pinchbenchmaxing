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

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log_print("[orchestrator] ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    model = cfg.orchestrator.model
    max_actions = cfg.orchestrator.max_actions
    budget_total = cfg.orchestrator.budget_usd

    system_prompt = build_system_prompt(cfg)
    consecutive_failures = 0

    log_print(f"\n{'='*62}")
    log_print(f"  ORCHESTRATOR SESSION")
    log_print(f"  Model: {model}")
    log_print(f"  Budget: ${budget_total}")
    log_print(f"  Max actions: {max_actions}")
    log_print(f"  Dry run: {dry_run}")
    log_print(f"{'='*62}")

    for turn in range(1, max_actions + 1):
        budget_remaining = budget_total - state.budget_spent_usd

        # ── Guardrails ─────────────────────────────────────────────────────
        if budget_remaining < 5.0:
            log_print(f"\n[orchestrator] Budget exhausted (${budget_remaining:.2f} remaining). Stopping.")
            break

        if consecutive_failures >= cfg.orchestrator.auto_pause.max_consecutive_failures:
            log_print(f"\n[orchestrator] {consecutive_failures} consecutive failures. Stopping.")
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
            log_print(f"[orchestrator] Claude API error: {e}")
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
            log_print(f"\n[orchestrator] {text_block.text}")
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
                log_print(f"[orchestrator] Claude says: {text_block.text[:200]}")
            continue

        # ── Execute tool ───────────────────────────────────────────────────
        tool_name = tool_use_block.name
        tool_args = tool_use_block.input or {}

        log_print(f"[orchestrator] Action: {tool_name}({json.dumps(tool_args, default=str)[:150]})")

        if dry_run:
            log_print(f"[orchestrator] DRY RUN — skipping execution")
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
            log_print(f"[orchestrator] FAILED: {error_msg[:200]}")
            result_summary = f"ERROR: {error_msg[:100]}"
        else:
            consecutive_failures = 0
            # Build a concise result summary for the action history
            r = result.get("result", {})
            if isinstance(r, dict):
                result_summary = ", ".join(f"{k}={v}" for k, v in list(r.items())[:5])
            else:
                result_summary = str(r)[:200]
            log_print(f"[orchestrator] Result: {result_summary[:200]}")

        if cost > 0:
            log_print(f"[orchestrator] Cost: ${cost:.2f}")

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
    log_print(f"  SESSION COMPLETE")
    log_print(f"  Turns: {len(state.action_history)}")
    log_print(f"  Budget spent: ${state.budget_spent_usd:.2f}")
    log_print(f"  Score: {state.avg_score:.3f} ({state.avg_score*100:.1f}%)")
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

        # Seed model
        if args.model:
            state.current_ollama_model = args.model
            m = re.search(r'-v(\d+)$', args.model)
            if m:
                state.model_version = int(m.group(1))
            log_print(f"[orchestrator] Model: {args.model} (v{state.model_version})")

        # Seed scores
        if args.scores:
            from loop import parse_scores_from_json_str
            seeded = parse_scores_from_json_str(args.scores)
            state.record_eval(seeded)
            log_print(f"[orchestrator] Seeded {len(seeded)} scores")
        elif args.log:
            from loop import parse_scores_from_log
            seeded = parse_scores_from_log(args.log)
            state.record_eval(seeded)
            log_print(f"[orchestrator] Seeded {len(seeded)} scores from {args.log}")

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
