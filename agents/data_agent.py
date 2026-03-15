"""
DataAgent — generates synthetic training data for tasks below target count.

Delegates all deficit logic to topup.py:
- topup.py computes which tasks need more examples (data count vs target)
- topup.py uses per-task EPC (1 for hard tasks, 3 for normal)
- topup.py submits to Claude Batch API and collects results

DataAgent's future role:
- Consume failure_analysis from EvalAgent to generate *targeted* examples
  (e.g. inject specific failure patterns into the meta-prompt)
- Adjust example style based on what the model is getting wrong
"""

import subprocess
import sys
from pathlib import Path

from .base import Agent, AgentState

_PROJECT_ROOT = Path(__file__).parent.parent


class DataAgent(Agent):
    """
    Generates synthetic training data to fill gaps in the dataset.

    Reads weak_tasks and failure_analysis from state (set by EvalAgent)
    but delegates the actual deficit calculation to topup.py, which
    counts existing examples and computes gaps against the target.
    """
    name = "data"

    def run(self, state: AgentState, cfg) -> AgentState:
        if not state.weak_tasks:
            self.log("No weak tasks — skipping data generation.")
            return state

        self.log(f"Weak tasks to address: {state.weak_tasks}")

        if state.failure_analysis:
            self.log("Failure context available (used in future targeted generation):")
            for task, note in state.failure_analysis.items():
                self.log(f"  {task}: {note}")

        self.log("Running topup (deficit auto-computed from data counts)...")
        topup = str(_PROJECT_ROOT / "topup.py")
        rc = self.run_cmd([sys.executable, topup, "run"], check=False)

        if rc == 2:
            self.log("All tasks already at target — no new data needed.")
        elif rc != 0:
            raise subprocess.CalledProcessError(rc, ["topup.py", "run"])

        return state


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    from utils.config import load_config
    cfg   = load_config()
    state = AgentState()
    agent = DataAgent()
    agent.run(state, cfg)
