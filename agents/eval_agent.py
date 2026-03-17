"""
EvalAgent — runs PinchBench against the current model, parses scores.

Skips if eval is already current for this model version (eval_version == model_version).
Gates: benchmark log must exist and contain parseable scores — raises on failure.
"""

import json
import re
import sys
from pathlib import Path

from .base import Agent, AgentState, PauseException, TASK_IDS


class EvalAgent(Agent):
    name = "eval"

    def run(self, state: AgentState, cfg) -> AgentState:
        # ── Gate: skip if already evaled this model version ──────────────────
        if state.eval_is_current:
            self.log(f"Eval already current for v{state.model_version} "
                     f"({state.current_ollama_model}) — skipping.")
            return state

        # ── Determine which model to benchmark ───────────────────────────────
        if not state.current_ollama_model:
            # First-ever run — no fine-tuned model yet, use base for baseline
            ollama_model = f"ollama/{cfg.ollama_model_name}"
            self.log(f"No fine-tuned model yet. Benchmarking base: {ollama_model}")
        else:
            ollama_model = f"ollama/{state.current_ollama_model}"
            self.log(f"Benchmarking v{state.model_version}: {ollama_model}")

        # ── Run benchmark ────────────────────────────────────────────────────
        log_file = self._run_benchmark(cfg, ollama_model, state)

        # ── Parse scores — hard gate ─────────────────────────────────────────
        scores = self._parse_scores(log_file)
        if not scores:
            raise RuntimeError(
                f"Could not parse any task scores from {log_file}. "
                "Check benchmark log for errors."
            )

        # ── Update state ────────────────────────────────────────────────────
        threshold  = cfg.loop.weak_task_threshold
        weak_tasks = [t for t in TASK_IDS if t in scores and scores[t] < threshold]

        prev_avg = state.avg_score
        state.record_eval(scores)
        new_avg  = state.avg_score

        self.log(f"Avg score: {prev_avg:.3f} → {new_avg:.3f}  "
                 f"(best ever: {state.best_avg_score:.3f} at v{state.best_version})")
        self.log(f"Weak tasks ({len(weak_tasks)}): {weak_tasks}")

        state.weak_tasks      = weak_tasks
        state.failure_analysis = self._analyze_failures(scores, weak_tasks, log_file)
        return state

    # ── Benchmark runner ──────────────────────────────────────────────────────

    def _run_benchmark(self, cfg, ollama_model: str, state=None) -> Path:
        script = Path(__file__).parent.parent / "scripts" / "benchmark_run.sh"
        self.run_cmd(["bash", str(script), ollama_model])

        # benchmark_run.sh writes to bench_{safe_name}.log
        safe_name = ollama_model.replace("/", "_").replace(":", "_")
        log_file  = cfg.data_dir.parent / "logs" / f"bench_{safe_name}.log"
        if not log_file.exists():
            raise RuntimeError(f"Benchmark log not found at {log_file}")

        # Copy to a timestamped + versioned file so no iteration overwrites another
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        iteration = state.iteration if state else 0
        version = state.model_version if state else 0
        archive = cfg.data_dir.parent / "logs" / f"bench_v{version}_iter{iteration}_{ts}.log"
        import shutil
        shutil.copy2(str(log_file), str(archive))
        self.log(f"  Log archived → {archive.name}")

        return log_file

    # ── Score parser ──────────────────────────────────────────────────────────

    def _parse_scores(self, log_file: Path) -> dict[str, float]:
        scores: dict[str, float] = {}
        text = log_file.read_text(errors="replace")

        # Format: "Task task_XX_name: 0.75/1.0 (75%)"
        for m in re.finditer(r'Task (task_\w+):\s*([01](?:\.\d+)?)\s*/\s*1\.0', text):
            if m.group(1) in TASK_IDS:
                scores[m.group(1)] = float(m.group(2))

        # Fallback: loose pattern
        if not scores:
            for m in re.finditer(
                r'(task_\d{2}_\w+)["\s:]*\s+([01](?:\.\d+)?)\s*(?:/\s*1\.0)?', text
            ):
                if m.group(1) in TASK_IDS:
                    scores[m.group(1)] = float(m.group(2))

        # Fallback: JSON blobs
        for blob in re.findall(r'\{[^{}]*"task_\d{2}_\w+"[^{}]*\}', text):
            try:
                obj = json.loads(blob)
                for k, v in obj.items():
                    if k in TASK_IDS and isinstance(v, (int, float)):
                        scores[k] = float(v)
            except json.JSONDecodeError:
                pass

        return scores

    # ── Failure analysis ──────────────────────────────────────────────────────

    def _analyze_failures(self, scores: dict, weak_tasks: list,
                          log_file: Path) -> dict[str, str]:
        analysis = {}
        for task in weak_tasks:
            score = scores.get(task, 0.0)
            if score == 0.0:
                analysis[task] = "complete failure — missing tool calls or wrong output"
            elif score < 0.3:
                analysis[task] = f"score={score:.2f} — partial, key criteria missed"
            else:
                analysis[task] = f"score={score:.2f} — close but below threshold"
        return analysis


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    from utils.config import load_config
    cfg   = load_config()
    state = AgentState()
    state = EvalAgent().run(state, cfg)
    print(f"\nScores: {state.scores}")
    print(f"Weak tasks: {state.weak_tasks}")
