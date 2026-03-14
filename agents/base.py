"""
Base classes for the AutoResearch agent pipeline.

Each agent has a single responsibility and communicates via AgentState.
Agents are independently runnable and can be developed without touching
the rest of the pipeline.
"""

import os
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# TASK REGISTRY  (single source of truth — matches generate.py / topup.py)
# ─────────────────────────────────────────────────────────────────────────────
TASK_IDS = [
    "task_00_sanity",
    "task_01_calendar",
    "task_02_stock",
    "task_03_blog",
    "task_04_weather",
    "task_05_summary",
    "task_06_events",
    "task_07_email",
    "task_08_memory",
    "task_09_files",
    "task_10_workflow",
    "task_11_config_update",
    "task_12_skill_search",
    "task_13_image_gen",
    "task_14_humanizer",
    "task_15_daily_summary",
    "task_16_email_triage",
    "task_17_email_search",
    "task_18_market_research",
    "task_19_spreadsheet_summary",
    "task_20_eli5_pdf",
    "task_21_openclaw_comprehension",
    "task_22_second_brain",
]


# ─────────────────────────────────────────────────────────────────────────────
# AGENT STATE  (flows between agents; persisted to loop_state.json)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AgentState:
    """
    Shared context that flows through the agent pipeline each iteration.

    scores          — per-task benchmark scores (0.0–1.0) from EvalAgent
    weak_tasks      — tasks below threshold, identified by EvalAgent
    failure_analysis — per-task failure notes; stub now, richer in future
                       (e.g. "model never calls write_file for task_09")
    history         — list of per-iteration summaries for loop_state.json
    iteration       — lifetime iteration counter
    """
    iteration:        int              = 0
    scores:           dict             = field(default_factory=dict)
    weak_tasks:       list             = field(default_factory=list)
    failure_analysis: dict             = field(default_factory=dict)
    history:          list             = field(default_factory=list)

    @property
    def avg_score(self) -> float:
        return sum(self.scores.values()) / len(self.scores) if self.scores else 0.0

    def snapshot(self, status: str) -> dict:
        """Return a history entry for this iteration."""
        return {
            "iteration":    self.iteration,
            "timestamp":    datetime.utcnow().isoformat(),
            "status":       status,
            "avg_score":    round(self.avg_score, 4),
            "scores":       dict(self.scores),
            "weak_tasks":   list(self.weak_tasks),
            "n_weak_tasks": len(self.weak_tasks),
        }

    def to_dict(self) -> dict:
        return {
            "iteration":        self.iteration,
            "scores":           self.scores,
            "weak_tasks":       self.weak_tasks,
            "failure_analysis": self.failure_analysis,
            "history":          self.history,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentState":
        return cls(
            iteration=        d.get("iteration", 0),
            scores=           d.get("scores", {}),
            weak_tasks=       d.get("weak_tasks", []),
            failure_analysis= d.get("failure_analysis", {}),
            history=          d.get("history", []),
        )


# ─────────────────────────────────────────────────────────────────────────────
# AGENT BASE CLASS
# ─────────────────────────────────────────────────────────────────────────────
class Agent(ABC):
    """
    Abstract base for all pipeline agents.

    Each agent:
    - Receives AgentState, does its work, returns updated AgentState
    - Is independently runnable (python -m agents.<name>_agent)
    - Logs under its own name for clear visibility in the orchestrator output
    """
    name: str = "agent"

    @abstractmethod
    def run(self, state: AgentState, cfg) -> AgentState:
        """Execute this agent's responsibility. Return updated state."""
        ...

    # ── Helpers ──────────────────────────────────────────────────────────────

    def log(self, msg: str) -> None:
        print(f"  [{self.name.upper()} AGENT] {msg}", flush=True)

    def run_cmd(self, cmd: list[str], env: dict | None = None,
                check: bool = True) -> int:
        merged = {**os.environ, **(env or {})}
        self.log(f"$ {' '.join(cmd)}")
        result = subprocess.run(cmd, env=merged)
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd)
        return result.returncode

    def __repr__(self) -> str:
        return f"<Agent:{self.name}>"
