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

from .base import Agent, AgentState


class CuratorAgent(Agent):
    """
    Quality gate between data generation and training.

    Only high-quality examples (judge score >= min_judge_score) make it
    through to the fine-tuning stage.
    """
    name = "curator"

    def run(self, state: AgentState, cfg) -> AgentState:
        min_score = cfg.data.min_judge_score
        self.log(f"Filtering examples with LLM judge (min score: {min_score}/5)...")
        self.run_cmd([sys.executable, "llm_judge.py", "filter", "--min", str(min_score)])
        return state


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    from utils.config import load_config
    cfg   = load_config()
    state = AgentState()
    agent = CuratorAgent()
    agent.run(state, cfg)
