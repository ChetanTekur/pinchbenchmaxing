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

        # ── Compute per-task example counts (score-proportional decay) ────
        target_score = cfg.loop.target_score
        try:
            max_per_task = cfg.loop.max_examples_per_task
        except AttributeError:
            max_per_task = 100
        try:
            total_cap = cfg.loop.total_new_examples_cap
        except AttributeError:
            total_cap = 500

        task_counts = self._compute_per_task_counts(
            state.weak_tasks, state.scores, target_score, max_per_task, total_cap
        )

        self.log(f"\n  Score-proportional generation plan:")
        total_planned = 0
        for task, n in sorted(task_counts.items()):
            score = state.scores.get(task, 0.0)
            self.log(f"    {task}: score={score:.2f} → {n} new examples")
            total_planned += n
        self.log(f"  Total: {total_planned} (cap: {total_cap})")

        # ── Classify adversarial tasks ─────────────────────────────────────
        tasks_adversarial = [t for t in state.weak_tasks
                             if self._needs_adversarial(t, state)]

        # ── Execute: targeted topup (all weak tasks, score-proportional) ───
        if task_counts:
            self.log(f"\n[TARGETED TOPUP] {len(task_counts)} tasks")
            script = str(_PROJECT_ROOT / "targeted_topup.py")

            # Pass each task with its specific count via the per-task mechanism
            # Use the max count as --min-per-task since we call per-task
            for task, n in task_counts.items():
                self.log(f"  {task}: generating {n} examples")
                rc = self.run_cmd(
                    [sys.executable, script, "run",
                     "--diagnosis-file", str(diag_file),
                     "--tasks", task,
                     "--min-per-task", str(n)],
                    check=False,
                )
                if rc not in (0, 2):
                    self.log(f"  WARNING: targeted_topup for {task} exited {rc}")

        # ── Execute: adversarial (score-0 and regenerate tasks) ────────────
        if tasks_adversarial:
            self.log(f"\n[ADVERSARIAL] {len(tasks_adversarial)} tasks")
            log_dir = cfg.data_dir.parent / "logs"
            script = str(_PROJECT_ROOT / "adversarial_gen.py")

            # Adversarial count also scales with gap, capped at 15
            for task in tasks_adversarial:
                score = state.scores.get(task, 0.0)
                gap = max(0, target_score - score)
                n_adv = max(3, min(15, round(15 * gap / target_score)))
                self.log(f"  {task}: {n_adv} adversarial examples")
                rc = self.run_cmd(
                    [sys.executable, script, "run",
                     "--log-dir", str(log_dir),
                     "--tasks", task,
                     "--n-per-task", str(n_adv)],
                    check=False,
                )
                if rc != 0:
                    self.log(f"  WARNING: adversarial for {task} exited {rc} (non-fatal)")

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

    def _compute_per_task_counts(
        self, weak_tasks: list, scores: dict,
        target_score: float, max_per_task: int, total_cap: int,
    ) -> dict[str, int]:
        """
        Compute how many new examples each weak task should get.

        Score-proportional decay:
          n = max_per_task × (gap / target)

        A task at 0% gets max_per_task (100).
        A task at 40% gets ~53.
        A task at 70% gets ~18.
        A task at 85% (target) gets 0.

        Total is capped at total_cap. If the sum exceeds the cap,
        all counts are scaled down proportionally (preserving relative allocation).
        """
        raw_counts = {}
        for task in weak_tasks:
            score = scores.get(task, 0.0)
            gap = max(0, target_score - score)
            n = round(max_per_task * (gap / target_score))
            raw_counts[task] = max(5, n)  # floor of 5 — always worth generating something

        # Apply total cap
        total = sum(raw_counts.values())
        if total > total_cap and total > 0:
            scale = total_cap / total
            raw_counts = {t: max(3, round(n * scale)) for t, n in raw_counts.items()}

        return raw_counts

    def _needs_adversarial(self, task: str, state: AgentState) -> bool:
        """Should this task also get adversarial examples (in addition to topup)?"""
        analysis = state.last_analysis or {}

        # Diagnosis says regenerate → learn from failure transcripts
        for df in analysis.get("data_fixes", []):
            if df.get("task") == task and df.get("action") in ("regenerate", "delete"):
                return True

        # Tasks scoring 0 always benefit from adversarial
        if state.scores.get(task, 0.0) == 0.0:
            return True

        return False


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    from utils.config import load_config
    cfg   = load_config()
    state = AgentState()
    agent = DataAgent()
    agent.run(state, cfg)
