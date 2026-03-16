"""
TrainerAgent — fine-tunes the model, creates a versioned Ollama registration.

Each run increments model_version and registers:
  {model_name}-v{N}   in Ollama (e.g. qwen35-9b-clawd-v3)

Previous versions remain registered so EvalAnalysisAgent can probe them.

Pipeline: prepare → finetune → convert → register_model (versioned) → verify
"""

import subprocess
import sys
from pathlib import Path

from .base import Agent, AgentState, PauseException


class TrainerAgent(Agent):
    name = "trainer"

    _STAGES = [
        ("prepare",  lambda exe, cfg: [exe, "-m", "stages.prepare"]),
        ("finetune", lambda exe, cfg: [exe, "-m", "stages.finetune"]),
        ("convert",  lambda exe, cfg: [exe, "-m", "stages.convert"]),
    ]

    def run(self, state: AgentState, cfg) -> AgentState:
        # ── Determine next version ────────────────────────────────────────────
        next_version    = state.model_version + 1
        versioned_name  = f"{cfg.ollama_model_name}-v{next_version}"
        self.log(f"Training v{next_version} → will register as '{versioned_name}'")

        # ── Run prepare → finetune → convert ─────────────────────────────────
        for label, build_cmd in self._STAGES:
            self.log(f"Stage: {label}")
            self.run_cmd(build_cmd(sys.executable, cfg))

        # ── Gate: GGUF must exist after convert ───────────────────────────────
        gguf = cfg.gguf_file
        if not gguf.exists():
            raise RuntimeError(
                f"Convert stage completed but GGUF not found at {gguf}. "
                "Check convert stage logs."
            )
        self.log(f"GGUF verified: {gguf} ({gguf.stat().st_size // 1024 // 1024} MB)")

        # ── Register versioned model in Ollama ────────────────────────────────
        self.log(f"Registering '{versioned_name}' in Ollama via register_model.sh ...")
        fix_script = Path(__file__).parent.parent / "scripts" / "register_model.sh"
        self.run_cmd(
            ["bash", str(fix_script)],
            env={"OLLAMA_MODEL": versioned_name},
        )

        # ── Gate: verify model is registered ─────────────────────────────────
        self._verify_ollama_model(versioned_name)

        # ── Record in state ───────────────────────────────────────────────────
        state.record_model(next_version, versioned_name)
        self.log(f"Model v{next_version} ready in Ollama as '{versioned_name}'")
        self.log(f"All versions: {[h['ollama_name'] for h in state.model_history]}")
        return state

    def _verify_ollama_model(self, model_name: str) -> None:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=30
        )
        if model_name not in result.stdout:
            raise RuntimeError(
                f"register_model.sh completed but '{model_name}' not found in "
                f"'ollama list'. Registration may have failed."
            )


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    from utils.config import load_config
    cfg   = load_config()
    state = AgentState()
    agent = TrainerAgent()
    agent.run(state, cfg)
