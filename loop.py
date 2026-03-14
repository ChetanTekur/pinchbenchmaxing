#!/usr/bin/env python3
"""
AutoResearch Orchestrator — agentic MLE loop for PinchBench.

Full pipeline per iteration:
  EvalAgent → EvalAnalysisAgent → DataAgent → CuratorAgent → TrainerAgent → (repeat)

EvalAgent is skipped if eval is already current for the model version.
The loop PAUSES (exit code 3) when human intervention is needed:
  - Score regressed >5% below best ever
  - No improvement for N consecutive iterations
  - DataAgent generates 0 new examples
  - CuratorAgent leaves train.jsonl empty
  - Training fails to produce a verified Ollama model

Usage:
  python loop.py run                   # run up to max_iterations
  python loop.py run --scores '{...}'  # seed known scores (skips first eval)
  python loop.py run --log /tmp/x.log  # seed scores from benchmark log
  python loop.py status                # show history + version table
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from utils.config import load_config
from agents import AgentState, EvalAgent, EvalAnalysisAgent, DataAgent, CuratorAgent, TrainerAgent
from agents.base import TASK_IDS, PauseException


# ── Tuning parameters ─────────────────────────────────────────────────────────
REGRESSION_THRESHOLD  = 0.05   # pause if score drops >5 pts below best ever
NO_IMPROVE_LIMIT      = 3      # pause if no improvement for this many iterations


# ─────────────────────────────────────────────────────────────────────────────
# STATE PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def load_state(state_file: Path) -> AgentState:
    if state_file.exists():
        return AgentState.from_dict(json.loads(state_file.read_text()))
    return AgentState()


def save_state(state: AgentState, state_file: Path) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state.to_dict(), indent=2))
    print(f"[orchestrator] State saved → {state_file}")


# ─────────────────────────────────────────────────────────────────────────────
# SCORE SEEDING
# ─────────────────────────────────────────────────────────────────────────────

def parse_scores_from_log(log_path: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    text = Path(log_path).read_text(errors="replace")
    for m in re.compile(r'(task_\d{2}_\w+)["\s:]*\s+([01](?:\.\d+)?)').finditer(text):
        if m.group(1) in TASK_IDS:
            scores[m.group(1)] = float(m.group(2))
    for blob in re.compile(r'\{[^{}]*"task_\d{2}_\w+"[^{}]*\}').findall(text):
        try:
            obj = json.loads(blob)
            for k, v in obj.items():
                if k in TASK_IDS and isinstance(v, (int, float)):
                    scores[k] = float(v)
        except json.JSONDecodeError:
            pass
    return scores


def parse_scores_from_json_str(json_str: str) -> dict[str, float]:
    raw = json.loads(json_str)
    scores: dict[str, float] = {}
    for k, v in raw.items():
        if k in TASK_IDS:
            scores[k] = float(v)
        else:
            matched = [t for t in TASK_IDS if t.startswith(k)]
            if len(matched) == 1:
                scores[matched[0]] = float(v)
    return scores


def count_train_examples(cfg) -> int:
    f = cfg.train_file
    if not f.exists():
        return 0
    return sum(1 for line in f.read_text().splitlines() if line.strip())


# ─────────────────────────────────────────────────────────────────────────────
# STAGE RUNNER  (single agent, with gate checking)
# ─────────────────────────────────────────────────────────────────────────────

def run_stage(agent, state: AgentState, cfg) -> AgentState:
    """
    Run one agent. On failure, update state and re-raise.
    PauseException propagates directly to the main loop.
    """
    print(f"\n{'─'*62}")
    print(f"  STAGE: {agent.name.upper()}")
    print(f"{'─'*62}")
    try:
        return agent.run(state, cfg)
    except PauseException:
        raise   # propagate directly — loop handles it
    except SystemExit as exc:
        raise RuntimeError(
            f"{agent.name} agent exited with code {exc.code}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"{agent.name} agent failed: {exc}") from exc


def pause(state: AgentState, state_file: Path, reason: str) -> None:
    """Mark state as paused, save, and raise PauseException."""
    state.pause_reason = reason
    state.history.append(state.snapshot("paused"))
    save_state(state, state_file)
    print(f"\n{'!'*62}")
    print(f"  ⏸  LOOP PAUSED — human review required")
    print(f"  Reason: {reason}")
    print(f"{'!'*62}")
    raise PauseException(reason)


def stage_failed(state: AgentState, state_file: Path,
                 agent_name: str, error: str) -> None:
    """Record failure, save, and exit — does NOT re-raise (caller exits)."""
    state.history[-1]["status"]       = f"failed:{agent_name}"
    state.history[-1]["failed_agent"] = agent_name
    state.history[-1]["error"]        = error
    save_state(state, state_file)
    print(f"\n{'!'*62}")
    print(f"  ✗  STAGE FAILED: {agent_name.upper()}")
    print(f"  {error}")
    print(f"  Fix the issue and re-run. State is saved.")
    print(f"{'!'*62}")


# ─────────────────────────────────────────────────────────────────────────────
# COMMANDS
# ─────────────────────────────────────────────────────────────────────────────

def cmd_status(args, cfg) -> None:
    state_file = cfg.data_dir / "loop_state.json"
    if not state_file.exists():
        print("No loop state found. Run 'python loop.py run' to start.")
        return
    state = load_state(state_file)

    print(f"\nLifetime iterations : {state.iteration}")
    print(f"Current avg score   : {state.avg_score:.3f}")
    print(f"Best avg score      : {state.best_avg_score:.3f}  (v{state.best_version})")
    print(f"Current model       : v{state.model_version}  ({state.current_ollama_model})")
    print(f"Weak tasks          : {state.weak_tasks}")

    if state.model_history:
        print(f"\nVersion history:")
        for entry in sorted(state.model_history, key=lambda h: h["version"]):
            print(f"  v{entry['version']:>2}  {entry['ollama_name']:<30}  "
                  f"avg={entry['avg_score']:.3f}  {entry['timestamp'][:10]}")

    print(f"\nIteration history:")
    for entry in state.history:
        ts      = entry.get("timestamp", "?")[:19]
        status  = entry.get("status", "?")
        avg     = entry.get("avg_score", "?")
        best    = entry.get("best_score", "?")
        n_weak  = entry.get("n_weak_tasks", "?")
        v       = entry.get("model_version", "?")
        pause   = f"  ← {entry['pause_reason']}" if entry.get("pause_reason") else ""
        print(f"  [{ts}] iter={entry['iteration']:>3}  v{v}  "
              f"avg={avg}  best={best}  weak={n_weak}  {status}{pause}")


def cmd_run(args, cfg) -> None:
    state_file = cfg.data_dir / "loop_state.json"
    state      = load_state(state_file)

    # ── Seed scores if provided ───────────────────────────────────────────────
    if args.scores:
        seeded = parse_scores_from_json_str(args.scores)
        state.record_eval(seeded)
        print(f"[orchestrator] Seeded {len(seeded)} scores from --scores.")
    elif args.log:
        seeded = parse_scores_from_log(args.log)
        state.record_eval(seeded)
        print(f"[orchestrator] Seeded {len(seeded)} scores from {args.log}.")

    # ── Seed model version if running a named model ───────────────────────────
    if args.model and not state.current_ollama_model:
        state.current_ollama_model = args.model
        print(f"[orchestrator] Set current model to: {args.model}")

    max_iter  = cfg.loop.max_iterations
    target    = cfg.loop.target_score
    threshold = cfg.loop.weak_task_threshold

    pipeline = [
        EvalAgent(),
        EvalAnalysisAgent(),
        DataAgent(),
        CuratorAgent(),
        TrainerAgent(),
    ]

    consecutive_no_improve = 0

    # ── Main loop ─────────────────────────────────────────────────────────────
    for step in range(1, max_iter + 1):
        state.iteration += 1

        print(f"\n{'='*62}")
        print(f"  ITERATION {step}/{max_iter}  (lifetime #{state.iteration})")
        print(f"  Avg score: {state.avg_score:.3f}  |  Best: {state.best_avg_score:.3f}  "
              f"|  Target: {target}")
        print(f"  Model: v{state.model_version} ({state.current_ollama_model or 'none'})")
        print(f"  Weak tasks ({len(state.weak_tasks)}): {state.weak_tasks}")
        print(f"{'='*62}")

        state.history.append(state.snapshot("running"))
        save_state(state, state_file)

        # ── STAGE 1: EVAL ────────────────────────────────────────────────────
        # Skip if eval is already current for this model version
        if not state.eval_is_current:
            try:
                state = run_stage(EvalAgent(), state, cfg)
            except PauseException:
                raise
            except RuntimeError as exc:
                stage_failed(state, state_file, "eval", str(exc))
                sys.exit(1)

            # GATE: regression check
            prev_best = state.best_avg_score
            if state.model_version > 0 and \
               state.avg_score < (prev_best - REGRESSION_THRESHOLD) and \
               prev_best > 0:
                pause(state, state_file,
                      f"Regression detected: avg={state.avg_score:.3f} is "
                      f"{prev_best - state.avg_score:.3f} below best "
                      f"({prev_best:.3f} at v{state.best_version}). "
                      f"Inspect eval_analysis report before continuing.")

            save_state(state, state_file)

        # GATE: target reached
        if state.avg_score >= target:
            print(f"\n[orchestrator] 🎯 Target {target} reached! Done.")
            state.history[-1].update(state.snapshot("target_reached"))
            save_state(state, state_file)
            break

        # GATE: no weak tasks
        if not state.weak_tasks:
            print("[orchestrator] No weak tasks — all above threshold.")
            state.history[-1].update(state.snapshot("no_weak_tasks"))
            save_state(state, state_file)
            break

        # GATE: no improvement for N iterations
        if step > 1 and state.avg_score <= state.history[-2].get("avg_score", 0):
            consecutive_no_improve += 1
            if consecutive_no_improve >= NO_IMPROVE_LIMIT:
                pause(state, state_file,
                      f"No improvement for {NO_IMPROVE_LIMIT} consecutive iterations. "
                      f"Review EvalAnalysisAgent report and data quality before continuing.")
        else:
            consecutive_no_improve = 0

        # ── STAGE 2: ANALYSIS ────────────────────────────────────────────────
        try:
            state = run_stage(EvalAnalysisAgent(), state, cfg)
        except PauseException:
            raise
        except RuntimeError as exc:
            # Analysis failure is non-fatal — log and continue
            print(f"[orchestrator] WARNING: analysis failed: {exc}  (continuing)")

        save_state(state, state_file)

        # ── STAGE 3: DATA ────────────────────────────────────────────────────
        pre_count = count_train_examples(cfg)
        try:
            state = run_stage(DataAgent(), state, cfg)
        except PauseException:
            raise
        except RuntimeError as exc:
            stage_failed(state, state_file, "data", str(exc))
            sys.exit(1)

        post_count = count_train_examples(cfg)

        # GATE: must generate new examples
        if post_count <= pre_count:
            pause(state, state_file,
                  f"DataAgent generated 0 new examples "
                  f"(before={pre_count}, after={post_count}). "
                  f"All tasks may be at target count, or topup.py failed.")

        print(f"[orchestrator] New examples: {pre_count} → {post_count} "
              f"(+{post_count - pre_count})")
        save_state(state, state_file)

        # ── STAGE 4: CURATOR ─────────────────────────────────────────────────
        try:
            state = run_stage(CuratorAgent(), state, cfg)
        except PauseException:
            raise
        except RuntimeError as exc:
            stage_failed(state, state_file, "curator", str(exc))
            sys.exit(1)

        # GATE: verify train.jsonl is non-empty (CuratorAgent also checks this,
        # but belt-and-suspenders)
        final_count = count_train_examples(cfg)
        if final_count == 0:
            pause(state, state_file,
                  "train.jsonl is empty after curation. "
                  "Lower min_judge_score in config.yaml or regenerate data.")

        save_state(state, state_file)

        # ── STAGE 5: TRAINER ─────────────────────────────────────────────────
        try:
            state = run_stage(TrainerAgent(), state, cfg)
        except PauseException:
            raise
        except RuntimeError as exc:
            stage_failed(state, state_file, "trainer", str(exc))
            sys.exit(1)

        # TrainerAgent sets eval_version = -1 (new model needs fresh eval)
        state.history[-1].update(state.snapshot("trained"))
        save_state(state, state_file)

        print(f"\n[orchestrator] Iteration {step} complete. "
              f"New model: v{state.model_version} ({state.current_ollama_model})")
        print(f"[orchestrator] Next iteration will begin with eval of "
              f"'{state.current_ollama_model}'.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description="AutoResearch orchestrator for PinchBench"
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run the agent pipeline")
    seed  = run_p.add_mutually_exclusive_group()
    seed.add_argument("--scores", type=str,
                      help='Seed scores as JSON: \'{"task_09_files": 0.0, ...}\'')
    seed.add_argument("--log", type=str,
                      help="Seed scores from a benchmark log file")
    run_p.add_argument("--model", type=str, default="",
                       help="Set current Ollama model name (without ollama/ prefix)")

    sub.add_parser("status", help="Show pipeline history and version table")

    args = parser.parse_args()
    if args.command == "run":
        try:
            cmd_run(args, cfg)
        except PauseException as exc:
            sys.exit(3)   # 3 = paused, distinct from error (1) / no-data (2)
    elif args.command == "status":
        cmd_status(args, cfg)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
