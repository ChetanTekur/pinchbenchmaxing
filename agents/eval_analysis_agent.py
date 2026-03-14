"""
EvalAnalysisAgent — iteratively diagnoses benchmark regressions.

Pipeline per round:
  collect_signals → generate_hypotheses → probe_models → refine

Repeats for --rounds iterations then writes a final diagnosis report to
{workspace}/data/eval_analysis_{timestamp}.json

Usage:
  python -m agents.eval_analysis_agent
  python -m agents.eval_analysis_agent --rounds 2
  python -m agents.eval_analysis_agent --models qwen3:8b qwen35-9b-clawd --rounds 3
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
import httpx

from .base import Agent, AgentState

# ── Constants ─────────────────────────────────────────────────────────────────
OLLAMA_URL     = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
ANALYSIS_MODEL = "claude-sonnet-4-6"
MAX_ROUNDS     = 2
PROBE_TIMEOUT  = 90   # seconds — Qwen3.5-9B can be slow

# Known benchmark regression for context (v1 → v2)
KNOWN_REGRESSION = {
    "Research":      {"v1": 1.9, "v2": 1.0, "delta": -0.9,
                      "tasks": ["task_02_stock", "task_06_events", "task_18_market_research"]},
    "Writing":       {"v1": 1.9, "v2": 1.0, "delta": -0.9,
                      "tasks": ["task_03_blog", "task_07_email", "task_14_humanizer"]},
    "Organization":  {"v1": 0.8, "v2": 0.0, "delta": -0.8,
                      "tasks": ["task_22_second_brain"]},
    "Calendar":      {"v1": 1.0, "v2": 0.7, "delta": -0.3,
                      "tasks": ["task_01_calendar"]},
    "Coding":        {"v1": 0.0, "v2": 1.0, "delta": +1.0,
                      "tasks": ["task_04_weather"]},
    "FileOps":       {"v1": 1.7, "v2": 2.1, "delta": +0.4,
                      "tasks": ["task_09_files", "task_10_workflow", "task_11_config_update"]},
    "Memory":        {"v1": 0.1, "v2": 0.0, "delta": -0.1,
                      "tasks": ["task_08_memory", "task_22_second_brain"]},
    "Comprehension": {"v1": 1.0, "v2": 1.0, "delta": 0.0,
                      "tasks": ["task_20_eli5_pdf", "task_21_openclaw_comprehension"]},
}

# Clawd tool definitions (Ollama native format)
CLAWD_TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read the contents of a file",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string", "description": "File path"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write content to a file",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List files in a directory",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_bash",
        "description": "Run a bash command and return stdout",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web and return results",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "write_memory",
        "description": "Store a fact in long-term memory",
        "parameters": {"type": "object",
                       "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                       "required": ["key", "value"]}}},
    {"type": "function", "function": {
        "name": "read_memory",
        "description": "Retrieve a fact from long-term memory",
        "parameters": {"type": "object",
                       "properties": {"key": {"type": "string"}},
                       "required": ["key"]}}},
    {"type": "function", "function": {
        "name": "draft_email",
        "description": "Draft and send an email",
        "parameters": {"type": "object",
                       "properties": {"to": {"type": "string"},
                                      "subject": {"type": "string"},
                                      "body": {"type": "string"}},
                       "required": ["to", "subject", "body"]}}},
    {"type": "function", "function": {
        "name": "generate_image",
        "description": "Generate an image from a text prompt",
        "parameters": {"type": "object",
                       "properties": {"prompt": {"type": "string"},
                                      "filename": {"type": "string"}},
                       "required": ["prompt", "filename"]}}},
    {"type": "function", "function": {
        "name": "create_calendar_event",
        "description": "Create a calendar event and save as .ics",
        "parameters": {"type": "object",
                       "properties": {"title": {"type": "string"},
                                      "date": {"type": "string"},
                                      "time": {"type": "string"},
                                      "attendees": {"type": "array", "items": {"type": "string"}},
                                      "filename": {"type": "string"}},
                       "required": ["title", "date", "time", "filename"]}}},
]

TOOLS_BY_NAME = {t["function"]["name"]: t for t in CLAWD_TOOLS}


# ─────────────────────────────────────────────────────────────────────────────
class EvalAnalysisAgent(Agent):
    """
    Diagnoses benchmark regressions by collecting signals, forming hypotheses,
    probing base + fine-tuned models, and iteratively refining understanding.
    """
    name = "eval_analysis"

    def run(self, state: AgentState, cfg) -> AgentState:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--rounds", type=int, default=MAX_ROUNDS)
        parser.add_argument("--models", nargs="+",
                            default=["qwen3:8b", cfg.ollama_model_name])
        args, _ = parser.parse_known_args()

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            self.log("ERROR: ANTHROPIC_API_KEY not set")
            return state

        client = anthropic.Anthropic(api_key=api_key)
        models = args.models

        self.log(f"Starting eval analysis — {args.rounds} rounds")
        self.log(f"Models to probe: {models}")

        # ── Phase 1: Collect signals ──────────────────────────────────────────
        signals = self._collect_signals(cfg)

        # ── Phase 2-4: Hypothesize → Probe → Refine ──────────────────────────
        hypotheses: list[dict] = []
        all_probes: list[dict] = []

        for round_num in range(1, args.rounds + 1):
            self.log(f"\n{'─'*55}")
            self.log(f"Round {round_num}/{args.rounds}: hypothesis generation")
            self.log(f"{'─'*55}")

            hypotheses = self._generate_hypotheses(
                client, signals, hypotheses, all_probes, round_num
            )
            if not hypotheses:
                self.log("No hypotheses produced — stopping early")
                break

            self.log(f"\nRound {round_num}: probing models")
            round_probes = self._probe_models(hypotheses, models)
            all_probes.extend(round_probes)

        # ── Phase 5: Final diagnosis ──────────────────────────────────────────
        report = self._final_diagnosis(client, signals, hypotheses, all_probes)

        # Save
        out_path = cfg.data_dir / f"eval_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        self.log(f"\nReport saved → {out_path}")

        self._print_summary(report["diagnosis"])

        # Populate failure_analysis in state for DataAgent to use
        for rc in report["diagnosis"].get("root_causes", []):
            for task in rc.get("affected_tasks", []):
                state.failure_analysis[task] = rc.get("cause", "")

        return state

    # ── Signal Collection ─────────────────────────────────────────────────────

    def _collect_signals(self, cfg) -> dict[str, Any]:
        self.log("Collecting signals...")
        signals = {
            "dataset":    self._sig_dataset(cfg),
            "validation": self._sig_validation(cfg),
            "judge":      self._sig_judge(cfg),
            "benchmark":  self._sig_benchmark_logs(),
            "training":   self._sig_training(cfg),
            "regression": KNOWN_REGRESSION,
        }
        self.log(f"  Dataset   : {signals['dataset']['total']} examples, "
                 f"{signals['dataset']['task_count']} tasks")
        self.log(f"  Validation: {signals['validation']['issue_count']} issues found")
        self.log(f"  Judge     : {signals['judge']['scored']} examples scored "
                 f"(avg {signals['judge']['avg'] or 'N/A'})")
        self.log(f"  Bench logs: {len(signals['benchmark']['runs'])} found")
        return signals

    def _sig_dataset(self, cfg) -> dict:
        try:
            out = subprocess.check_output(
                [sys.executable, "inspect_data.py", "stats"],
                stderr=subprocess.DEVNULL, timeout=30, text=True
            )
            by_task: dict[str, int] = {}
            for line in out.splitlines():
                m = re.match(r'\s+(task_\w+)\s+(\d+)', line)
                if m:
                    by_task[m.group(1)] = int(m.group(2))
            return {"by_task": by_task, "total": sum(by_task.values()),
                    "task_count": len(by_task), "raw": out}
        except Exception as e:
            return {"by_task": {}, "total": 0, "task_count": 0, "error": str(e)}

    def _sig_validation(self, cfg) -> dict:
        try:
            out = subprocess.check_output(
                [sys.executable, "inspect_data.py", "validate"],
                stderr=subprocess.DEVNULL, timeout=90, text=True
            )
            by_type: dict[str, int] = {}
            for line in out.splitlines():
                m = re.match(r'\s+(\d+)×\s+(.+)', line)
                if m:
                    by_type[m.group(2).strip()] = int(m.group(1))
            # Also capture per-task issue details
            task_issues: dict[str, list] = defaultdict(list)
            current_task = None
            for line in out.splitlines():
                tm = re.search(r'task_\w+', line)
                if tm and "Task:" in line:
                    current_task = tm.group()
                elif current_task and "✗" in line:
                    task_issues[current_task].append(line.strip().lstrip("✗").strip())
            return {"by_type": by_type, "task_issues": dict(task_issues),
                    "issue_count": sum(by_type.values()), "raw": out}
        except Exception as e:
            return {"by_type": {}, "task_issues": {}, "issue_count": 0, "error": str(e)}

    def _sig_judge(self, cfg) -> dict:
        scores_file = cfg.data_dir / "scores.json"
        if not scores_file.exists():
            return {"scored": 0, "avg": None, "by_task": {}}
        try:
            scores = json.loads(scores_file.read_text())
            by_task: dict[str, list] = defaultdict(list)
            for v in scores.values():
                s = v.get("score", 0)
                if s > 0:
                    by_task[v.get("task_id", "unknown")].append(s)
            flat = [s for lst in by_task.values() for s in lst]
            by_task_avg = {t: round(sum(s) / len(s), 2) for t, s in by_task.items() if s}
            file_age_h = (time.time() - scores_file.stat().st_mtime) / 3600
            return {
                "scored": len(flat),
                "avg": round(sum(flat) / len(flat), 2) if flat else None,
                "by_task": by_task_avg,
                "file_age_hours": round(file_age_h, 1),
                "stale": file_age_h > 12,
            }
        except Exception as e:
            return {"scored": 0, "avg": None, "by_task": {}, "error": str(e)}

    def _sig_benchmark_logs(self) -> dict:
        """Parse all bench_*.log files in /tmp/."""
        runs = []
        for log_file in sorted(Path("/tmp").glob("bench_*.log"),
                                key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                text = log_file.read_text(errors="replace")
                run: dict[str, Any] = {
                    "file": log_file.name,
                    "mtime": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat(),
                    "scores": {},
                    "errors": [],
                }
                # Parse task scores  — format: "Task task_XX_name: 0.75/1.0 (75%)"
                for m in re.finditer(
                    r'Task (task_\w+):\s*([01](?:\.\d+)?)\s*/\s*1\.0', text
                ):
                    run["scores"][m.group(1)] = float(m.group(2))
                # Errors
                for line in text.splitlines():
                    if re.search(r'\bERROR\b|error.*:', line, re.IGNORECASE):
                        run["errors"].append(line.strip()[:200])
                if run["scores"]:
                    runs.append(run)
            except Exception:
                pass
        return {"runs": runs[:5]}  # keep last 5

    def _sig_training(self, cfg) -> dict:
        for log_path in [Path("/tmp/finetune.log"), cfg.workspace / "finetune.log"]:
            if log_path.exists():
                text = log_path.read_text(errors="replace")
                losses = [float(x) for x in
                          re.findall(r"'loss':\s*([0-9.]+)", text)]
                return {
                    "log_path": str(log_path),
                    "tail": "\n".join(text.splitlines()[-80:]),
                    "losses": losses[-30:],
                    "final_loss": losses[-1] if losses else None,
                }
        return {"log_path": None, "tail": "", "losses": [], "final_loss": None}

    # ── Hypothesis Generation ─────────────────────────────────────────────────

    def _generate_hypotheses(
        self,
        client: anthropic.Anthropic,
        signals: dict,
        prev: list[dict],
        probes: list[dict],
        round_num: int,
    ) -> list[dict]:
        if round_num == 1:
            prompt = self._prompt_initial(signals)
        else:
            prompt = self._prompt_refine(signals, prev, probes)

        try:
            resp = client.messages.create(
                model=ANALYSIS_MODEL,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            # Extract JSON array
            m = re.search(r'\[\s*\{.*\}\s*\]', raw, re.DOTALL)
            if not m:
                self.log("WARNING: could not parse JSON from response")
                self.log(f"  Snippet: {raw[:300]}")
                return prev
            hypotheses = json.loads(m.group())
            self.log(f"  {len(hypotheses)} hypotheses:")
            for h in hypotheses:
                self.log(f"    [{h.get('confidence','?'):6}] {h['id']}: "
                         f"{h['hypothesis'][:70]}")
            return hypotheses
        except Exception as e:
            self.log(f"ERROR in hypothesis generation: {e}")
            return prev

    def _prompt_initial(self, signals: dict) -> str:
        bench_txt = ""
        for run in signals["benchmark"]["runs"][:2]:
            bench_txt += f"\n  File: {run['file']}  ({run['mtime'][:10]})\n"
            for task, score in sorted(run["scores"].items()):
                bench_txt += f"    {task}: {score:.2f}\n"
            if run["errors"][:2]:
                bench_txt += f"    Errors: {run['errors'][:2]}\n"

        val_issues = json.dumps(signals["validation"]["by_type"], indent=4)
        task_issues = json.dumps(signals["validation"]["task_issues"], indent=4)
        judge_by_task = json.dumps(signals["judge"]["by_task"], indent=4)
        dataset_by_task = json.dumps(signals["dataset"]["by_task"], indent=4)

        return f"""You are a machine learning diagnostician analyzing a fine-tuning regression.

## Context
Model: Qwen3.5-9B fine-tuned on synthetic data to act as "Clawd", an AI agent
Framework: OpenClaw (tool-calling agent framework)
Benchmark: PinchBench — 23 tasks, each scored 0.0–1.0

## Known Regression (v1 → v2, overall 54% → 45%)
{json.dumps(KNOWN_REGRESSION, indent=2)}

## Dataset counts per task (training examples)
{dataset_by_task}

## Structural validation issues in the dataset
By issue type: {val_issues}
By task:       {task_issues}

## LLM judge scores per task (scale 1–5, from scores.json)
Scored: {signals['judge']['scored']} examples | Average: {signals['judge']['avg']}
File age: {signals['judge']['file_age_hours']} hours (stale={signals['judge']['stale']})
Per task: {judge_by_task}

## Training loss (last 30 steps)
{signals['training']['losses']}
Final loss: {signals['training']['final_loss']}

## Recent benchmark logs
{bench_txt or '  (none found in /tmp/)'}

## Your task
Generate 4–6 specific, testable hypotheses explaining the regression.
For each hypothesis, specify 1–2 test prompts to run against:
- "base"      : qwen3:8b   (untuned — reveals baseline tool-calling behavior)
- "finetuned" : qwen35-9b-clawd (our model — what did training change?)

Hypotheses to consider:
- Dataset imbalance (tasks 09–22 have 2× more examples than 00–08)
- Validation issues causing bad examples to persist in training
- Tool call format corruption (model using wrong tool names or args)
- Catastrophic forgetting on tasks with fewer examples
- Specific task patterns learned incorrectly (e.g. research vs. retrieval)
- Judge scores stale — new topup examples never re-judged

Return ONLY a valid JSON array, no explanation outside it:
[
  {{
    "id": "h1",
    "hypothesis": "...",
    "confidence": "high|medium|low",
    "affected_tasks": ["task_XX", ...],
    "evidence": "which signals support this",
    "test_prompts": [
      {{
        "id": "p1",
        "description": "what we're testing",
        "system": "You are Clawd, an AI agent. Use tools to complete tasks.",
        "user": "< the actual prompt to send to the model >",
        "tools_needed": ["write_file", "web_search"],
        "expected_base": "what the untuned model should do",
        "expected_finetuned": "what our fine-tuned model should do if working correctly",
        "regression_signal": "what response would confirm this hypothesis"
      }}
    ]
  }}
]"""

    def _prompt_refine(
        self, signals: dict, prev: list[dict], probes: list[dict]
    ) -> str:
        return f"""You are refining hypotheses about a fine-tuning regression based on model probe results.

## Previous Hypotheses
{json.dumps(prev, indent=2)}

## Model Probe Results
{json.dumps(probes, indent=2)}

## Your task
1. Mark each hypothesis as confirmed / refuted / unclear based on probe results
2. Generate REFINED or NEW hypotheses to close remaining gaps
3. Propose new targeted test prompts for anything still unclear

Return ONLY a valid JSON array:
[
  {{
    "id": "hN",
    "status": "confirmed|refuted|unclear|new",
    "hypothesis": "...",
    "confidence": "high|medium|low",
    "affected_tasks": [...],
    "evidence": "...",
    "test_prompts": [...]
  }}
]"""

    # ── Model Probing ─────────────────────────────────────────────────────────

    def _probe_models(self, hypotheses: list[dict], models: list[str]) -> list[dict]:
        results = []
        for hyp in hypotheses:
            for p in hyp.get("test_prompts", []):
                self.log(f"  Probe [{hyp['id']}/{p['id']}] {p['description'][:60]}")
                probe: dict[str, Any] = {
                    "hypothesis_id": hyp["id"],
                    "prompt_id":     p["id"],
                    "description":   p["description"],
                    "expected_base":       p.get("expected_base", ""),
                    "expected_finetuned":  p.get("expected_finetuned", ""),
                    "regression_signal":   p.get("regression_signal", ""),
                    "responses": {},
                }
                tools = self._get_tools(p.get("tools_needed", []))
                for model in models:
                    self.log(f"    → {model} ...", )
                    resp = self._query_ollama(
                        model=model,
                        system=p.get("system", "You are Clawd, an AI agent."),
                        user=p["user"],
                        tools=tools,
                    )
                    probe["responses"][model] = resp
                    # Quick summary
                    if resp.get("error"):
                        self.log(f"      ✗ {resp['error'][:80]}")
                    elif resp.get("tool_calls"):
                        calls = [tc["name"] for tc in resp["tool_calls"]]
                        self.log(f"      Tools: {calls}")
                    else:
                        snippet = (resp.get("content") or "")[:80]
                        self.log(f"      Text: {snippet}")
                results.append(probe)
        return results

    def _query_ollama(
        self,
        model: str,
        system: str,
        user: str,
        tools: list[dict],
    ) -> dict:
        payload: dict[str, Any] = {
            "model":   model,
            "stream":  False,
            "messages": [
                {"role": "system",  "content": system},
                {"role": "user",    "content": user},
            ],
        }
        if tools:
            payload["tools"] = tools

        try:
            r = httpx.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
                timeout=PROBE_TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
            msg  = data.get("message", {})
            tcs  = msg.get("tool_calls") or []
            return {
                "content":    msg.get("content", ""),
                "tool_calls": [
                    {"name": tc["function"]["name"],
                     "arguments": tc["function"].get("arguments", {})}
                    for tc in tcs
                ],
                "done_reason":    data.get("done_reason", ""),
                "duration_ms":    data.get("total_duration", 0) // 1_000_000,
            }
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}",
                    "content": "", "tool_calls": []}
        except Exception as e:
            return {"error": str(e), "content": "", "tool_calls": []}

    def _get_tools(self, names: list[str]) -> list[dict]:
        if not names:
            return CLAWD_TOOLS
        return [TOOLS_BY_NAME[n] for n in names if n in TOOLS_BY_NAME]

    # ── Final Diagnosis ───────────────────────────────────────────────────────

    def _final_diagnosis(
        self,
        client: anthropic.Anthropic,
        signals: dict,
        hypotheses: list[dict],
        probes: list[dict],
    ) -> dict:
        self.log("\nGenerating final diagnosis...")

        prompt = f"""You are producing a final regression diagnosis for a fine-tuned LLM.

## All Hypotheses (with statuses)
{json.dumps(hypotheses, indent=2)}

## All Probe Results
{json.dumps(probes, indent=2)}

## Validation Issues
{json.dumps(signals['validation']['by_type'], indent=2)}

## Dataset Imbalance
{json.dumps(signals['dataset']['by_task'], indent=2)}

## Judge Score Staleness
Scored: {signals['judge']['scored']}, Age: {signals['judge']['file_age_hours']}h

Produce a concise, actionable diagnosis. Return ONLY valid JSON:
{{
  "summary": "2-3 sentence plain-English summary of why the model regressed",
  "root_causes": [
    {{
      "rank": 1,
      "cause": "...",
      "confidence": "high|medium|low",
      "affected_tasks": ["task_XX", ...],
      "evidence": "...",
      "fix": "concrete action to fix this"
    }}
  ],
  "data_fixes": [
    {{
      "task": "task_XX",
      "action": "delete|regenerate|topup|rejudge",
      "reason": "...",
      "priority": "high|medium|low"
    }}
  ],
  "training_changes": ["..."],
  "v3_watchpoints": ["..."],
  "rerun_judge": true
}}"""

        try:
            resp = client.messages.create(
                model=ANALYSIS_MODEL,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            diagnosis = json.loads(m.group()) if m else {"summary": raw,
                                                          "root_causes": [],
                                                          "data_fixes": [],
                                                          "training_changes": [],
                                                          "v3_watchpoints": []}
        except Exception as e:
            diagnosis = {"summary": f"Diagnosis error: {e}", "root_causes": [],
                         "data_fixes": [], "training_changes": [], "v3_watchpoints": []}

        return {
            "timestamp":  datetime.now().isoformat(),
            "signals_summary": {
                "dataset_total":   signals["dataset"]["total"],
                "validation_issues": signals["validation"]["issue_count"],
                "judge_scored":    signals["judge"]["scored"],
                "judge_stale":     signals["judge"].get("stale", False),
                "bench_runs":      len(signals["benchmark"]["runs"]),
            },
            "hypotheses":  hypotheses,
            "probe_results": probes,
            "diagnosis":   diagnosis,
        }

    def _print_summary(self, diag: dict) -> None:
        print(f"\n{'═'*62}")
        print(f"  EVAL ANALYSIS DIAGNOSIS")
        print(f"{'═'*62}")
        print(f"\n  {diag.get('summary', '')}")

        print(f"\n  ROOT CAUSES:")
        for rc in diag.get("root_causes", []):
            conf = rc.get("confidence", "?")
            print(f"  [{rc.get('rank','?')}] [{conf:6}] {rc.get('cause', '')}")
            print(f"          Tasks:    {rc.get('affected_tasks', [])}")
            print(f"          Fix:      {rc.get('fix', '')}")

        print(f"\n  DATA FIXES:")
        for fix in diag.get("data_fixes", []):
            print(f"  [{fix.get('priority','?'):6}] {fix.get('task','?')}: "
                  f"{fix.get('action','?')} — {fix.get('reason','')}")

        print(f"\n  TRAINING CHANGES:")
        for c in diag.get("training_changes", []):
            print(f"    • {c}")

        print(f"\n  WATCH IN V3:")
        for w in diag.get("v3_watchpoints", []):
            print(f"    • {w}")

        if diag.get("rerun_judge"):
            print(f"\n  ⚠  Recommendation: re-run llm_judge.py run (scores are stale)")

        print(f"\n{'═'*62}")


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    from utils.config import load_config
    cfg   = load_config()
    state = AgentState()
    agent = EvalAnalysisAgent()
    agent.run(state, cfg)
