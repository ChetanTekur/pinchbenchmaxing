"""
TrainerAgent — fine-tunes the model and registers it with Ollama.

Pipeline:
  prepare → finetune → convert (GGUF) → fix_modelfile → Ollama ready

Future capabilities:
- Hyperparameter search based on previous training loss curves
- Early stopping if eval loss plateaus
- A/B testing between training configurations
"""

import sys
from pathlib import Path

from .base import Agent, AgentState


class TrainerAgent(Agent):
    """
    Owns the full training lifecycle: data prep → LoRA fine-tune →
    GGUF quantization → Ollama model registration.

    After this agent completes, the model is live in Ollama and ready
    for EvalAgent to benchmark.
    """
    name = "trainer"

    # Ordered stages: (label, command)
    _STAGES = [
        ("prepare",       lambda exe, cfg: [exe, "-m", "stages.prepare"]),
        ("finetune",      lambda exe, cfg: [exe, "-m", "stages.finetune"]),
        ("convert",       lambda exe, cfg: [exe, "-m", "stages.convert"]),
        ("fix_modelfile", lambda exe, cfg: ["bash", str(
            Path(__file__).parent.parent / "scripts" / "fix_modelfile.sh"
        )]),
    ]

    def run(self, state: AgentState, cfg) -> AgentState:
        for label, build_cmd in self._STAGES:
            self.log(f"Stage: {label}")
            self.run_cmd(build_cmd(sys.executable, cfg))
        self.log("Model ready in Ollama.")
        return state


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    from utils.config import load_config
    cfg   = load_config()
    state = AgentState()
    agent = TrainerAgent()
    agent.run(state, cfg)
