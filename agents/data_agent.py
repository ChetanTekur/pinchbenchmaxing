"""
DataAgent — generates targeted training data using diagnosis from EvalAnalysisAgent.

Three generation strategies, selected per-task based on the diagnosis:

1. Targeted topup (default) — diagnosis-aware meta-prompts with weighted variations
2. Adversarial generation — for tasks that scored 0, generates from benchmark transcripts
3. Plain topup (fallback) — round-robin when no diagnosis is available

The diagnosis flows from EvalAnalysisAgent → state.last_analysis → data_agent →
current_diagnosis.json → targeted_topup.py → meta-prompt. This closes the feedback
loop: benchmark failures directly shape what training data gets generated.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from .base import Agent, AgentState

_PROJECT_ROOT = Path(__file__).parent.parent


class DataAgent(Agent):
    name = "data"

    def run(self, state: AgentState, cfg) -> AgentState:
        if not state.weak_tasks:
            self.log("No weak tasks — skipping data generation.")
            return state

        self.log(f"Weak tasks ({len(state.weak_tasks)}): {state.weak_tasks}")

        # ── Build per-task diagnosis from EvalAnalysisAgent output ──────────
        diagnosis = self._build_per_task_diagnosis(state)
        diag_file = cfg.data_dir / "current_diagnosis.json"
        diag_file.parent.mkdir(parents=True, exist_ok=True)
        diag_file.write_text(json.dumps(diagnosis, indent=2))

        if diagnosis:
            self.log(f"Diagnosis covers {len(diagnosis)} tasks:")
            for task, d in sorted(diagnosis.items()):
                cause = d.get("root_cause", "")[:80]
                action = d.get("data_action", "topup")
                self.log(f"  {task}: [{action}] {cause}")
        else:
            self.log("No diagnosis available — falling back to plain topup")

        # ── Classify tasks by generation strategy ──────────────────────────
        tasks_topup = []
        tasks_adversarial = []

        for task in state.weak_tasks:
            strategy = self._select_strategy(task, state, cfg)
            if strategy == "adversarial":
                tasks_adversarial.append(task)
            else:
                tasks_topup.append(task)

        # ── Execute strategies ─────────────────────────────────────────────

        # Strategy 1: Targeted topup (diagnosis-aware)
        if tasks_topup:
            min_per_task = cfg.loop.examples_per_weak_task
            self.log(f"\n[TARGETED TOPUP] {len(tasks_topup)} tasks "
                     f"(min {min_per_task} new examples each)")
            script = str(_PROJECT_ROOT / "targeted_topup.py")
            rc = self.run_cmd(
                [sys.executable, script, "run",
                 "--diagnosis-file", str(diag_file),
                 "--tasks", ",".join(tasks_topup),
                 "--min-per-task", str(min_per_task)],
                check=False,
            )
            if rc == 2:
                self.log("All topup tasks already at target count.")
            elif rc != 0:
                raise subprocess.CalledProcessError(rc, ["targeted_topup.py"])

        # Strategy 2: Adversarial generation from benchmark transcripts
        if tasks_adversarial:
            self.log(f"\n[ADVERSARIAL] {len(tasks_adversarial)} tasks")
            log_dir = cfg.data_dir.parent / "logs"
            script = str(_PROJECT_ROOT / "adversarial_gen.py")

            n_per_task = cfg.get("data", {})
            if hasattr(n_per_task, "get"):
                n_per_task = n_per_task.get("adversarial_examples_per_task", 3)
            else:
                n_per_task = 3

            rc = self.run_cmd(
                [sys.executable, script, "run",
                 "--log-dir", str(log_dir),
                 "--tasks", ",".join(tasks_adversarial),
                 "--n-per-task", str(n_per_task)],
                check=False,
            )
            if rc != 0:
                self.log(f"WARNING: adversarial generation exited {rc} (non-fatal)")

        return state

    # ── Helpers ────────────────────────────────────────────────────────────

    def _build_per_task_diagnosis(self, state: AgentState) -> dict:
        """Merge EvalAnalysisAgent's root_causes + data_fixes into per-task dict."""
        result = {}
        analysis = state.last_analysis or {}

        # From root_causes
        for rc in analysis.get("root_causes", []):
            for task in rc.get("affected_tasks", []):
                result.setdefault(task, {})
                result[task]["root_cause"] = rc.get("cause", "")
                result[task]["fix"] = rc.get("fix", "")
                result[task]["confidence"] = rc.get("confidence", "low")

        # From data_fixes
        for df in analysis.get("data_fixes", []):
            task = df.get("task", "")
            if task:
                result.setdefault(task, {})
                result[task]["data_action"] = df.get("action", "topup")
                result[task]["priority"] = df.get("priority", "medium")
                result[task]["reason"] = df.get("reason", "")

        # From failure_analysis (simpler per-task notes from EvalAgent)
        for task, note in state.failure_analysis.items():
            result.setdefault(task, {})
            result[task].setdefault("root_cause", note)

        return result

    def _select_strategy(self, task: str, state: AgentState, cfg) -> str:
        """Pick generation strategy for a task based on score + diagnosis."""
        analysis = state.last_analysis or {}

        # Check if diagnosis says to regenerate
        for df in analysis.get("data_fixes", []):
            if df.get("task") == task and df.get("action") in ("regenerate", "delete"):
                return "adversarial"  # regenerate = adversarial (learn from failure)

        # Tasks that scored 0 benefit most from adversarial generation
        score = state.scores.get(task, 0.0)
        if score == 0.0:
            log_dir = cfg.data_dir.parent / "logs"
            if log_dir.exists() and any(log_dir.glob("bench_*.log")):
                return "adversarial"

        return "topup"


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    from utils.config import load_config
    cfg   = load_config()
    state = AgentState()
    agent = DataAgent()
    agent.run(state, cfg)
