"""
CuratorAgent — filters low-quality training examples via LLM judge.

Runs llm_judge.py to score every example in train.jsonl and removes
any that score below min_judge_score (default: 3/5).

Future capabilities:
- Rewrite borderline examples (score 2-3) instead of discarding them
- Detect and deduplicate near-identical examples
- Balance the dataset across variation types (happy path vs error recovery)
"""

import sys
from pathlib import Path

from .base import Agent, AgentState

_PROJECT_ROOT = Path(__file__).parent.parent


class CuratorAgent(Agent):
    """
    Quality gate between data generation and training.

    Gate 1: Run llm_judge.py run  — score ALL examples (errors = hard stop)
    Gate 2: Run llm_judge.py filter — remove low-quality examples
    Gate 3: Verify train.jsonl is non-empty after filtering

    Only high-quality examples (judge score >= min_judge_score) make it
    through to the fine-tuning stage.
    """
    name = "curator"

    def run(self, state: AgentState, cfg) -> AgentState:
        min_score = cfg.data.min_judge_score
        scores_file = cfg.data_dir / "scores.json"

        # Gate 1: Score all examples
        judge = str(_PROJECT_ROOT / "llm_judge.py")
        self.log("Running LLM judge on all examples (this may take a while)...")
        self.run_cmd([sys.executable, judge, "run"])

        if not scores_file.exists():
            raise RuntimeError(
                "llm_judge.py run completed but scores.json was not created. "
                "Check ANTHROPIC_API_KEY and llm_judge.py output."
            )
        self.log(f"  Scores written to {scores_file}")

        # Gate 2: Filter by min score
        self.log(f"Filtering examples with min score {min_score}/5...")
        self.run_cmd([sys.executable, judge, "filter", "--min", str(min_score)])

        # Gate 3: Verify output is non-empty
        train_file = cfg.train_file
        if not train_file.exists() or train_file.stat().st_size == 0:
            raise RuntimeError(
                f"train.jsonl is empty after filtering at score >= {min_score}. "
                "Lower min_judge_score or regenerate data."
            )
        n = sum(1 for line in train_file.read_text().splitlines() if line.strip())
        self.log(f"  {n} examples passed quality gate (score >= {min_score}/5)")

        return state


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    from utils.config import load_config
    cfg   = load_config()
    state = AgentState()
    agent = CuratorAgent()
    agent.run(state, cfg)
