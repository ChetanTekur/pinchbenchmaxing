#!/usr/bin/env python3
"""
AutoResearch Orchestrator — agentic MLE loop for PinchBench.

Pipeline (one iteration):
  DataAgent → CuratorAgent → TrainerAgent → EvalAgent

Each agent has a single responsibility and communicates via AgentState.
Agents are independently runnable:
  python -m agents.eval_agent
  python -m agents.data_agent
  python -m agents.curator_agent
  python -m agents.trainer_agent

Usage:
  python loop.py run                   # run up to max_iterations
  python loop.py run --scores '{...}'  # seed with known scores
  python loop.py run --log /tmp/x.log  # seed from benchmark log
  python loop.py status                # show history
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from utils.config import load_config
from agents import AgentState, EvalAgent, DataAgent, CuratorAgent, TrainerAgent
from agents.base import TASK_IDS


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
    print(f"\n[orchestrator] State saved → {state_file}")


# ─────────────────────────────────────────────────────────────────────────────
# SCORE SEEDING  (manual override via --scores / --log)
# ─────────────────────────────────────────────────────────────────────────────

def parse_scores_from_log(log_path: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    text = Path(log_path).read_text(errors="replace")
    pattern = re.compile(
        r'(task_\d{2}_\w+)["\s:]*\s+([01](?:\.\d+)?)\s*(?:/\s*1\.0)?'
    )
    for m in pattern.finditer(text):
        if m.group(1) in TASK_IDS:
            scores[m.group(1)] = float(m.group(2))
    json_pat = re.compile(r'\{[^{}]*"task_\d{2}_\w+"[^{}]*\}')
    for blob in json_pat.findall(text):
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


# ─────────────────────────────────────────────────────────────────────────────
# COMMANDS
# ─────────────────────────────────────────────────────────────────────────────

def cmd_status(args, cfg) -> None:
    state_file = cfg.data_dir / "loop_state.json"
    if not state_file.exists():
        print("No loop state found. Run 'python loop.py run' to start.")
        return
    state = load_state(state_file)
    print(f"Lifetime iterations: {state.iteration}")
    print(f"Current avg score:   {state.avg_score:.3f}")
    print(f"Weak tasks:          {state.weak_tasks}")
    print(f"\nHistory:")
    for entry in state.history:
        ts     = entry.get("timestamp", "?")[:19]
        status = entry.get("status", "?")
        avg    = entry.get("avg_score", "?")
        n_weak = entry.get("n_weak_tasks", "?")
        print(f"  [{ts}] iter={entry['iteration']:>3}  avg={avg}  "
              f"weak={n_weak}  status={status}")


def cmd_run(args, cfg) -> None:
    state_file = cfg.data_dir / "loop_state.json"
    state      = load_state(state_file)

    # ── Seed scores if provided ───────────────────────────────────────────────
    if args.scores:
        state.scores = parse_scores_from_json_str(args.scores)
        print(f"[orchestrator] Seeded {len(state.scores)} scores from --scores.")
    elif args.log:
        state.scores = parse_scores_from_log(args.log)
        print(f"[orchestrator] Seeded {len(state.scores)} scores from {args.log}.")
    elif state.scores:
        print(f"[orchestrator] Resumed with {len(state.scores)} scores from state.")

    max_iter  = cfg.loop.max_iterations
    target    = cfg.loop.target_score
    threshold = cfg.loop.weak_task_threshold

    # ── Agent pipeline ────────────────────────────────────────────────────────
    pipeline = [DataAgent(), CuratorAgent(), TrainerAgent(), EvalAgent()]

    # Bootstrap: if no scores, run EvalAgent first for a baseline
    if not state.scores:
        print("[orchestrator] No prior scores — running baseline benchmark...")
        state = EvalAgent().run(state, cfg)
        if not state.scores:
            print("[orchestrator] WARNING: baseline parse failed. Treating all as weak.")
            state.scores     = {t: 0.0 for t in TASK_IDS}
            state.weak_tasks = list(TASK_IDS)
        save_state(state, state_file)

    # ── Main loop ─────────────────────────────────────────────────────────────
    for step in range(1, max_iter + 1):
        state.iteration += 1

        print(f"\n{'='*62}")
        print(f"  ITERATION {step}/{max_iter}  (lifetime #{state.iteration})")
        print(f"  Avg score: {state.avg_score:.3f}  |  Target: {target}")
        print(f"  Weak tasks ({len(state.weak_tasks)}): {state.weak_tasks}")
        print(f"{'='*62}")

        if state.avg_score >= target:
            print(f"[orchestrator] Target {target} reached. Done.")
            state.history.append(state.snapshot("target_reached"))
            save_state(state, state_file)
            break

        if not state.weak_tasks:
            print("[orchestrator] No weak tasks. All above threshold.")
            state.history.append(state.snapshot("no_weak_tasks"))
            save_state(state, state_file)
            break

        state.history.append(state.snapshot("running"))
        save_state(state, state_file)

        for agent in pipeline:
            print(f"\n── {agent.name.upper()} AGENT {'─'*40}")
            try:
                state = agent.run(state, cfg)
            except SystemExit as exc:
                if exc.code == 2:
                    print(f"[orchestrator] {agent.name} agent: exited with code 2 "
                          f"(no new data). Stopping.", file=sys.stderr)
                    state.history[-1]["status"] = "no_new_data"
                    state.history[-1]["failed_agent"] = agent.name
                    save_state(state, state_file)
                    sys.exit(2)
                raise
            except Exception as exc:
                print(f"\n[orchestrator] ✗ {agent.name.upper()} AGENT FAILED: {exc}",
                      file=sys.stderr)
                print(f"[orchestrator] Pipeline halted — fix {agent.name} before continuing.",
                      file=sys.stderr)
                state.history[-1]["status"]       = f"failed:{agent.name}"
                state.history[-1]["failed_agent"] = agent.name
                state.history[-1]["error"]        = str(exc)
                save_state(state, state_file)
                sys.exit(1)

        state.history[-1].update(state.snapshot("complete"))
        save_state(state, state_file)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description="AutoResearch orchestrator: agentic MLE loop for PinchBench"
    )
    sub = parser.add_subparsers(dest="command")

    run_parser  = sub.add_parser("run",    help="Run the agent pipeline")
    score_group = run_parser.add_mutually_exclusive_group()
    score_group.add_argument("--scores", type=str,
        help='Seed scores as JSON: \'{"task_09_files": 0.0, ...}\'')
    score_group.add_argument("--log", type=str,
        help="Seed scores from a benchmark log file")

    sub.add_parser("status", help="Show pipeline history")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args, cfg)
    elif args.command == "status":
        cmd_status(args, cfg)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
