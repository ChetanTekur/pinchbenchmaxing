"""
EvalAgent — runs PinchBench, parses scores, identifies weak tasks.

Future capability: analyze *why* tasks fail by inspecting model outputs
and generate targeted failure reports to guide data generation.
"""

import json
import re
import sys
from pathlib import Path

from .base import Agent, AgentState, TASK_IDS


class EvalAgent(Agent):
    """
    Responsibilities:
    1. Run PinchBench against the current model
    2. Parse per-task scores from the benchmark log
    3. Identify weak tasks (score < threshold)
    4. Populate failure_analysis — currently a stub; future: LLM-powered
       inspection of model outputs to identify *specific* failure patterns
       (e.g. "model skips write_file", "wrong filenames", "no error handling")

    Future:
    - Given a natural language task description, generate a new benchmark
      with relevant metrics automatically (eval-agent-as-benchmark-designer)
    """
    name = "eval"

    def run(self, state: AgentState, cfg) -> AgentState:
        self.log("Running PinchBench...")
        log_file = self._run_benchmark(cfg)

        self.log(f"Parsing scores from {log_file} ...")
        scores = self._parse_scores(str(log_file))

        if not scores:
            self.log("WARNING: Could not parse scores. Keeping previous scores.")
            return state

        threshold  = cfg.loop.weak_task_threshold
        weak_tasks = [t for t in TASK_IDS if t in scores and scores[t] < threshold]

        self.log(f"Avg score: {state.avg_score:.3f} → {sum(scores.values())/len(scores):.3f}")
        self.log(f"Weak tasks ({len(weak_tasks)}): {weak_tasks}")

        state.scores          = scores
        state.weak_tasks      = weak_tasks
        state.failure_analysis = self._analyze_failures(scores, weak_tasks, log_file)
        return state

    # ── Benchmark runner ──────────────────────────────────────────────────────

    def _run_benchmark(self, cfg) -> Path:
        script = Path(__file__).parent.parent / "scripts" / "benchmark_run.sh"
        model  = f"ollama/{cfg.model_name}"
        self.run_cmd(["bash", str(script), model])
        safe_name = model.replace("/", "_").replace(":", "_")
        return Path(f"/tmp/bench_{safe_name}.log")

    # ── Score parser ──────────────────────────────────────────────────────────

    def _parse_scores(self, log_path: str) -> dict[str, float]:
        scores: dict[str, float] = {}
        text = Path(log_path).read_text(errors="replace")

        pattern = re.compile(
            r'(task_\d{2}_\w+)["\s:]*\s+([01](?:\.\d+)?)\s*(?:/\s*1\.0)?'
        )
        for m in pattern.finditer(text):
            task_id = m.group(1)
            if task_id in TASK_IDS:
                scores[task_id] = float(m.group(2))

        json_pattern = re.compile(r'\{[^{}]*"task_\d{2}_\w+"[^{}]*\}')
        for blob in json_pattern.findall(text):
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
        """
        Identify WHY each weak task failed.

        Currently a stub — returns score-based notes.

        Future implementation:
        - Read per-task model outputs from the benchmark log
        - Use an LLM to classify failure patterns:
            "model did not call generate_image"
            "wrong output filename"
            "missing error handling in generated code"
        - Return structured failure notes that DataAgent uses to
          generate more targeted training examples
        - Longer term: given a natural language task description,
          generate a full benchmark with relevant grading metrics
        """
        analysis = {}
        for task in weak_tasks:
            score = scores.get(task, 0.0)
            if score == 0.0:
                analysis[task] = "complete failure — model likely missing tool calls or wrong output"
            elif score < 0.3:
                analysis[task] = f"score={score:.2f} — partial completion, key criteria missed"
            else:
                analysis[task] = f"score={score:.2f} — close but below threshold"
        return analysis


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    from utils.config import load_config
    cfg   = load_config()
    state = AgentState()
    agent = EvalAgent()
    state = agent.run(state, cfg)
    print(f"\nScores: {state.scores}")
    print(f"Weak tasks: {state.weak_tasks}")
