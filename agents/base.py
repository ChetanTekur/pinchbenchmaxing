"""
Base classes for the AutoResearch agent pipeline.

Each agent has a single responsibility and communicates via AgentState.
Agents are independently runnable and can be developed without touching
the rest of the pipeline.
"""

import logging
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# FILE LOGGER  (all agents write to the same log file for unified debugging)
# ─────────────────────────────────────────────────────────────────────────────
_log_file = None


def setup_file_logger(log_dir: str | os.PathLike, session_label: str = "") -> None:
    """Initialize the shared file logger. Called once at startup.

    Creates a session-specific log file (e.g. loop_v23_20260330_143000.log)
    and a loop_latest.log symlink for convenience.
    """
    global _log_file
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    label = f"_{session_label}" if session_label else ""
    session_log = log_dir / f"loop{label}_{ts}.log"
    _log_file = open(session_log, "a", buffering=1)

    # Symlink loop_latest.log for convenience
    latest = log_dir / "loop_latest.log"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(session_log.name)
    except OSError:
        pass  # symlink may fail on some filesystems

    _write_log(f"\n{'='*62}")
    _write_log(f"  Session started: {datetime.utcnow().isoformat()}")
    _write_log(f"  Log file: {session_log.name}")
    _write_log(f"{'='*62}")


def _write_log(msg: str) -> None:
    """Write a line to the log file (if open)."""
    if _log_file:
        ts = datetime.utcnow().strftime("%H:%M:%S")
        _log_file.write(f"[{ts}] {msg}\n")


def log_print(msg: str = "") -> None:
    """Print to stdout AND write to log file. Use instead of print() in loop.py."""
    print(msg, flush=True)
    _write_log(msg)


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
class PauseException(Exception):
    """
    Raised when the pipeline should pause for human review.
    The reason is stored in state.pause_reason before raising.
    Exit code 3 — distinct from error (1) and no-new-data (2).
    """
    pass


@dataclass
class AgentState:
    """
    Shared context that flows through the agent pipeline each iteration.

    Versioning
    ----------
    model_version       — incremented by TrainerAgent after each fine-tune (0 = no model yet)
    current_ollama_model — Ollama tag of the model to benchmark/probe (set by TrainerAgent)
    eval_version        — model_version at which eval was last run (avoids redundant evals)
    model_history       — [{version, ollama_name, avg_score, scores, timestamp}]
    best_avg_score      — highest avg_score ever achieved across all versions
    best_version        — model_version that achieved best_avg_score

    Scoring
    -------
    scores          — per-task benchmark scores (0.0–1.0) from EvalAgent
    weak_tasks      — tasks below threshold, identified by EvalAgent
    failure_analysis — per-task failure notes from EvalAgent + EvalAnalysisAgent

    Control
    -------
    pause_reason    — human-readable explanation when PauseException is raised
    history         — per-iteration snapshots for loop_state.json
    iteration       — lifetime iteration counter
    """
    # Scoring
    iteration:           int   = 0
    scores:              dict  = field(default_factory=dict)
    weak_tasks:          list  = field(default_factory=list)
    failure_analysis:    dict  = field(default_factory=dict)
    history:             list  = field(default_factory=list)

    # Model versioning
    model_version:       int   = 0
    current_ollama_model: str  = ""
    eval_version:        int   = -1   # -1 = never evaled
    model_history:       list  = field(default_factory=list)
    best_avg_score:      float = 0.0
    best_version:        int   = 0

    # Control flow
    pause_reason:        str   = ""
    last_analysis:       dict  = field(default_factory=dict)
    model_validated:     bool  = False
    data_gen_version:    int   = -1   # model_version at which data was last generated

    # Orchestrator
    action_history:      list  = field(default_factory=list)  # [{turn, action, result_summary, cost}]
    budget_spent_usd:    float = 0.0
    base_model:          str   = ""   # tracks which HF model this state is for
    scratchpad:          list  = field(default_factory=list)  # [{timestamp, note}] — agent's working memory
    last_data_summary:   dict  = field(default_factory=dict)  # cached inspect_data result
    baseline_task_counts: dict = field(default_factory=dict)  # per-task counts at session start — filter protection
    diagnosis_required:  bool  = False  # set True after benchmark; blocks generation until diagnose runs
    diagnose_count:      int   = 0      # diagnose calls since last benchmark; caps at 2 to prevent analysis paralysis

    @property
    def avg_score(self) -> float:
        return sum(self.scores.values()) / len(self.scores) if self.scores else 0.0

    @property
    def eval_is_current(self) -> bool:
        """True if eval has already been run for the current model version."""
        return self.eval_version == self.model_version and bool(self.scores)

    def record_model(self, version: int, ollama_name: str) -> None:
        """Called by TrainerAgent after registering a new model version."""
        self.model_version       = version
        self.current_ollama_model = ollama_name
        self.eval_version        = -1   # invalidate — eval needed for new model
        self.data_gen_version    = -1   # invalidate — data gen needed for new model

    def record_eval(self, scores: dict) -> None:
        """Called by EvalAgent after a successful eval."""
        self.scores    = scores
        self.eval_version = self.model_version
        avg = self.avg_score
        # Update model history entry for current version
        entry = {
            "version":     self.model_version,
            "ollama_name": self.current_ollama_model,
            "avg_score":   round(avg, 4),
            "scores":      dict(scores),
            "timestamp":   datetime.utcnow().isoformat(),
        }
        # Replace existing entry for this version or append
        self.model_history = [
            h for h in self.model_history if h["version"] != self.model_version
        ]
        self.model_history.append(entry)
        # Track best
        if avg > self.best_avg_score:
            self.best_avg_score = avg
            self.best_version   = self.model_version

    def snapshot(self, status: str) -> dict:
        return {
            "iteration":     self.iteration,
            "timestamp":     datetime.utcnow().isoformat(),
            "status":        status,
            "model_version": self.model_version,
            "ollama_model":  self.current_ollama_model,
            "avg_score":     round(self.avg_score, 4),
            "best_score":    round(self.best_avg_score, 4),
            "scores":        dict(self.scores),
            "weak_tasks":    list(self.weak_tasks),
            "n_weak_tasks":  len(self.weak_tasks),
            "pause_reason":  self.pause_reason,
        }

    def to_dict(self) -> dict:
        return {
            "iteration":           self.iteration,
            "scores":              self.scores,
            "weak_tasks":          self.weak_tasks,
            "failure_analysis":    self.failure_analysis,
            "history":             self.history,
            "model_version":       self.model_version,
            "current_ollama_model": self.current_ollama_model,
            "eval_version":        self.eval_version,
            "model_history":       self.model_history,
            "best_avg_score":      self.best_avg_score,
            "best_version":        self.best_version,
            "pause_reason":        self.pause_reason,
            "last_analysis":       self.last_analysis,
            "model_validated":     self.model_validated,
            "data_gen_version":    self.data_gen_version,
            "action_history":      self.action_history,
            "budget_spent_usd":    self.budget_spent_usd,
            "base_model":          self.base_model,
            "scratchpad":          self.scratchpad,
            "last_data_summary":   self.last_data_summary,
            "baseline_task_counts": self.baseline_task_counts,
            "diagnosis_required":  self.diagnosis_required,
            "diagnose_count":      self.diagnose_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentState":
        return cls(
            iteration=            d.get("iteration", 0),
            scores=               d.get("scores", {}),
            weak_tasks=           d.get("weak_tasks", []),
            failure_analysis=     d.get("failure_analysis", {}),
            history=              d.get("history", []),
            model_version=        d.get("model_version", 0),
            current_ollama_model= d.get("current_ollama_model", ""),
            eval_version=         d.get("eval_version", -1),
            model_history=        d.get("model_history", []),
            best_avg_score=       d.get("best_avg_score", 0.0),
            best_version=         d.get("best_version", 0),
            pause_reason=         d.get("pause_reason", ""),
            last_analysis=        d.get("last_analysis", {}),
            model_validated=      d.get("model_validated", False),
            data_gen_version=     d.get("data_gen_version", -1),
            action_history=       d.get("action_history", []),
            budget_spent_usd=     d.get("budget_spent_usd", 0.0),
            base_model=           d.get("base_model", ""),
            scratchpad=           d.get("scratchpad", []),
            last_data_summary=    d.get("last_data_summary", {}),
            baseline_task_counts= d.get("baseline_task_counts", {}),
            diagnosis_required=   d.get("diagnosis_required", False),
            diagnose_count=       d.get("diagnose_count", 0),
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
        line = f"  [{self.name.upper()}] {msg}"
        print(line, flush=True)
        _write_log(line)

    def run_cmd(self, cmd: list[str], env: dict | None = None,
                check: bool = True) -> int:
        """Run a subprocess, streaming output line-by-line to stdout and log file."""
        merged = {**os.environ, **(env or {})}
        self.log(f"$ {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd, env=merged,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,  # line-buffered
        )
        for line in proc.stdout:
            line = line.rstrip('\n')
            print(f"    {line}", flush=True)
            _write_log(f"    {line}")
        proc.wait()
        if check and proc.returncode != 0:
            self.log(f"Command exited with code {proc.returncode}")
            raise subprocess.CalledProcessError(proc.returncode, cmd)
        return proc.returncode

    def __repr__(self) -> str:
        return f"<Agent:{self.name}>"
