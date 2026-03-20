"""
DataAgent — generates targeted training data using diagnosis from EvalAnalysisAgent.

Uses score-proportional generation: tasks further from target get more examples.
Two strategies per weak task:
  1. Targeted topup — diagnosis-aware meta-prompts with weighted variations
  2. Adversarial — for score-0 tasks, generates from benchmark failure transcripts
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
            self.log("No weak tasks — skipping")
            return state

        self.log(f"Weak tasks ({len(state.weak_tasks)}): {state.weak_tasks}")

        # ── Build per-task diagnosis ───────────────────────────────────────
        diagnosis = self._build_per_task_diagnosis(state)
        diag_file = cfg.data_dir / "current_diagnosis.json"
        diag_file.parent.mkdir(parents=True, exist_ok=True)
        diag_file.write_text(json.dumps(diagnosis, indent=2))

        if diagnosis:
            self.log(f"Diagnosis for {len(diagnosis)} tasks:")
            for task, d in sorted(diagnosis.items()):
                cause = d.get("root_cause", "")[:70]
                action = d.get("data_action", "topup")
                self.log(f"  [{action}] {task}: {cause}")
        else:
            self.log("No diagnosis — using plain topup")

        # ── Compute per-task example counts ────────────────────────────────
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

        self.log(f"Generation plan (score-proportional):")
        total_planned = 0
        for task, n in sorted(task_counts.items()):
            score = state.scores.get(task, 0.0)
            self.log(f"  {task}: score={score:.2f} gap={target_score-score:.2f} → {n} examples")
            total_planned += n
        self.log(f"  Total: {total_planned} (cap: {total_cap})")

        # ── Classify adversarial tasks ─────────────────────────────────────
        tasks_adversarial = [t for t in state.weak_tasks
                             if self._needs_adversarial(t, state)]

        # ── Strategy 1: Targeted topup ─────────────────────────────────────
        if task_counts:
            self.log(f"Running targeted topup for {len(task_counts)} tasks...")
            script = str(_PROJECT_ROOT / "targeted_topup.py")
            for task, n in task_counts.items():
                self.log(f"  [topup] {task}: {n} examples")
                rc = self.run_cmd(
                    [sys.executable, script, "run",
                     "--diagnosis-file", str(diag_file),
                     "--tasks", task,
                     "--min-per-task", str(n)],
                    check=False,
                )
                if rc not in (0, 2):
                    self.log(f"  WARNING: topup for {task} exited {rc}")

        # ── Strategy 2: Adversarial generation ─────────────────────────────
        if tasks_adversarial:
            self.log(f"Running adversarial generation for {len(tasks_adversarial)} tasks...")
            log_dir = cfg.data_dir.parent / "logs"
            script = str(_PROJECT_ROOT / "adversarial_gen.py")

            for task in tasks_adversarial:
                score = state.scores.get(task, 0.0)
                gap = max(0, target_score - score)
                n_adv = max(3, min(15, round(15 * gap / target_score)))
                self.log(f"  [adversarial] {task}: {n_adv} examples (score={score:.2f})")
                rc = self.run_cmd(
                    [sys.executable, script, "run",
                     "--log-dir", str(log_dir),
                     "--tasks", task,
                     "--n-per-task", str(n_adv)],
                    check=False,
                )
                if rc != 0:
                    self.log(f"  WARNING: adversarial for {task} exited {rc}")

        return state

    # ── Helpers ────────────────────────────────────────────────────────────

    def _build_per_task_diagnosis(self, state: AgentState) -> dict:
        result = {}
        analysis = state.last_analysis or {}

        for rc in analysis.get("root_causes", []):
            for task in rc.get("affected_tasks", []):
                result.setdefault(task, {})
                result[task]["root_cause"] = rc.get("cause", "")
                result[task]["fix"] = rc.get("fix", "")
                result[task]["confidence"] = rc.get("confidence", "low")

        for df in analysis.get("data_fixes", []):
            task = df.get("task", "")
            if task:
                result.setdefault(task, {})
                result[task]["data_action"] = df.get("action", "topup")
                result[task]["priority"] = df.get("priority", "medium")
                result[task]["reason"] = df.get("reason", "")

        for task, note in state.failure_analysis.items():
            result.setdefault(task, {})
            result[task].setdefault("root_cause", note)

        return result

    def _compute_per_task_counts(
        self, weak_tasks: list, scores: dict,
        target_score: float, max_per_task: int, total_cap: int,
    ) -> dict[str, int]:
        """Score-proportional: n = max × (gap / target). Capped at total_cap."""
        raw_counts = {}
        for task in weak_tasks:
            score = scores.get(task, 0.0)
            gap = max(0, target_score - score)
            n = round(max_per_task * (gap / target_score))
            raw_counts[task] = max(5, n)

        total = sum(raw_counts.values())
        if total > total_cap and total > 0:
            scale = total_cap / total
            raw_counts = {t: max(3, round(n * scale)) for t, n in raw_counts.items()}

        return raw_counts

    def _needs_adversarial(self, task: str, state: AgentState) -> bool:
        analysis = state.last_analysis or {}
        for df in analysis.get("data_fixes", []):
            if df.get("task") == task and df.get("action") in ("regenerate", "delete"):
                return True
        if state.scores.get(task, 0.0) == 0.0:
            return True
        return False


if __name__ == "__main__":
    from utils.config import load_config
    cfg   = load_config()
    state = AgentState()
    DataAgent().run(state, cfg)
