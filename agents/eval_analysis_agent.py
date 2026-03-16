"""
EvalAnalysisAgent — iteratively diagnoses regressions by probing all model versions.

Discovers all registered Ollama models matching our model naming pattern,
then probes them with targeted prompts to understand how behavior changed
across versions. Uses Claude to form and refine hypotheses.

Pipeline per round:
  collect_signals → generate_hypotheses → probe_all_versions → refine

Usage:
  python -m agents.eval_analysis_agent
  python -m agents.eval_analysis_agent --rounds 2
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
from utils.config import load_config as _load_config

# ── Constants ─────────────────────────────────────────────────────────────────
OLLAMA_URL     = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
ANALYSIS_MODEL = "claude-sonnet-4-6"
MAX_ROUNDS     = 2
PROBE_TIMEOUT  = 90

# Clawd tool definitions (Ollama native format)
CLAWD_TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read file contents",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Write content to a file",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "list_files", "description": "List directory contents",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "web_search", "description": "Search the web",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "write_memory", "description": "Store a fact in memory",
        "parameters": {"type": "object",
                       "properties": {"key": {"type": "string"},
                                      "value": {"type": "string"}},
                       "required": ["key", "value"]}}},
    {"type": "function", "function": {
        "name": "read_memory", "description": "Retrieve a fact from memory",
        "parameters": {"type": "object",
                       "properties": {"key": {"type": "string"}},
                       "required": ["key"]}}},
    {"type": "function", "function": {
        "name": "draft_email", "description": "Draft and send an email",
        "parameters": {"type": "object",
                       "properties": {"to": {"type": "string"},
                                      "subject": {"type": "string"},
                                      "body": {"type": "string"}},
                       "required": ["to", "subject", "body"]}}},
    {"type": "function", "function": {
        "name": "create_calendar_event", "description": "Create a .ics calendar event",
        "parameters": {"type": "object",
                       "properties": {"title": {"type": "string"},
                                      "date": {"type": "string"},
                                      "time": {"type": "string"},
                                      "attendees": {"type": "array",
                                                    "items": {"type": "string"}},
                                      "filename": {"type": "string"}},
                       "required": ["title", "date", "time", "filename"]}}},
    {"type": "function", "function": {
        "name": "generate_image", "description": "Generate an image from a prompt",
        "parameters": {"type": "object",
                       "properties": {"prompt": {"type": "string"},
                                      "filename": {"type": "string"}},
                       "required": ["prompt", "filename"]}}},
]
TOOLS_BY_NAME = {t["function"]["name"]: t for t in CLAWD_TOOLS}


# ─────────────────────────────────────────────────────────────────────────────
class EvalAnalysisAgent(Agent):
    """
    Diagnoses regressions by probing ALL registered model versions via Ollama,
    comparing tool-calling behavior version by version.
    """
    name = "eval_analysis"

    def run(self, state: AgentState, cfg) -> AgentState:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--rounds", type=int, default=MAX_ROUNDS)
        args, _ = parser.parse_known_args()

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            self.log("ANTHROPIC_API_KEY not set — skipping analysis (non-fatal)")
            return state

        client = anthropic.Anthropic(api_key=api_key)

        # ── Discover all available model versions ─────────────────────────────
        models = self._discover_models(cfg, state)
        if not models:
            self.log("No model versions found in Ollama — skipping analysis")
            return state

        self.log(f"Models to probe ({len(models)}): {[m['label'] for m in models]}")

        # ── Collect signals ───────────────────────────────────────────────────
        signals = self._collect_signals(cfg, state, models)

        # ── Hypothesis → Probe loop ───────────────────────────────────────────
        hypotheses: list[dict] = []
        all_probes: list[dict] = []

        for rnd in range(1, args.rounds + 1):
            self.log(f"\n── Analysis round {rnd}/{args.rounds} ──")
            hypotheses = self._generate_hypotheses(
                client, signals, state, hypotheses, all_probes, rnd
            )
            if not hypotheses:
                self.log("No hypotheses — stopping")
                break
            round_probes = self._probe_models(hypotheses, models)
            all_probes.extend(round_probes)

        # ── Final diagnosis ───────────────────────────────────────────────────
        report = self._final_diagnosis(client, signals, state, hypotheses, all_probes)

        out = cfg.data_dir / f"eval_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        self.log(f"Report → {out}")

        self._print_summary(report["diagnosis"])

        # Populate failure_analysis for DataAgent
        for rc in report["diagnosis"].get("root_causes", []):
            for task in rc.get("affected_tasks", []):
                state.failure_analysis[task] = rc.get("cause", "")

        state.last_analysis = report["diagnosis"]
        return state

    # ── Model Discovery ───────────────────────────────────────────────────────

    def _discover_models(self, cfg, state: AgentState) -> list[dict]:
        """
        Query `ollama list` and return all models relevant to this project:
        - All versions of our fine-tuned model (name matches {model_name}-vN)
        - Any legacy names tracked in model_history
        - The base Ollama model if available (qwen3:8b or similar)

        Returns list of dicts: {ollama_name, label, version (int or 'base')}
        """
        try:
            result = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, timeout=20
            )
            available = set()
            for line in result.stdout.splitlines()[1:]:  # skip header
                name = line.split()[0] if line.split() else ""
                if name:
                    available.add(name.split(":")[0])  # strip :latest tag
                    available.add(name)
        except Exception as e:
            self.log(f"Could not query ollama list: {e}")
            return []

        models = []
        base_name = cfg.model_name  # e.g. qwen35-9b-clawd

        # Add all registered versions of our model (from state history + ollama list)
        seen_versions = set()
        for entry in sorted(state.model_history, key=lambda h: h["version"]):
            oname = entry["ollama_name"]
            tag   = oname + ":latest"
            if oname in available or tag in available:
                v = entry["version"]
                seen_versions.add(v)
                models.append({
                    "ollama_name": oname,
                    "label":       f"v{v} ({oname})",
                    "version":     v,
                    "avg_score":   entry.get("avg_score"),
                })

        # Also scan ollama list for versioned names not in history (e.g. from manual runs)
        version_pattern = re.compile(rf'^{re.escape(base_name)}-v(\d+)(?::.*)?$')
        for name in sorted(available):
            m = version_pattern.match(name)
            if m:
                v = int(m.group(1))
                if v not in seen_versions:
                    seen_versions.add(v)
                    models.append({
                        "ollama_name": name.split(":")[0],
                        "label":       f"v{v} ({name})",
                        "version":     v,
                        "avg_score":   None,
                    })

        # Legacy names (e.g. qwen35-9b-gguf-claw from before versioning)
        legacy_names = ["qwen35-9b-gguf-claw", base_name]
        for legacy in legacy_names:
            tag = legacy + ":latest"
            if (legacy in available or tag in available) and \
               not any(m["ollama_name"] == legacy for m in models):
                models.append({
                    "ollama_name": legacy,
                    "label":       f"legacy ({legacy})",
                    "version":     "legacy",
                    "avg_score":   None,
                })

        # Sort by version (legacy last)
        models.sort(key=lambda m: m["version"] if isinstance(m["version"], int) else 9999)
        return models

    # ── Signal Collection ─────────────────────────────────────────────────────

    def _collect_signals(self, cfg, state: AgentState, models: list[dict]) -> dict:
        self.log("Collecting signals...")
        signals = {
            "model_history":   state.model_history,
            "current_version": state.model_version,
            "best_version":    state.best_version,
            "best_avg_score":  state.best_avg_score,
            "dataset":         self._sig_dataset(cfg),
            "validation":      self._sig_validation(cfg),
            "judge":           self._sig_judge(cfg),
            "benchmark_logs":  self._sig_benchmark_logs(cfg),
            "training":        self._sig_training(cfg),
        }
        self.log(f"  Dataset    : {signals['dataset']['total']} examples")
        self.log(f"  Validation : {signals['validation']['issue_count']} issues")
        self.log(f"  Judge      : {signals['judge']['scored']} scored "
                 f"(age {signals['judge'].get('file_age_hours','?')}h)")
        self.log(f"  Bench logs : {len(signals['benchmark_logs']['runs'])} found")
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
                    "task_count": len(by_task)}
        except Exception as e:
            return {"by_task": {}, "total": 0, "task_count": 0, "error": str(e)}

    def _sig_validation(self, cfg) -> dict:
        try:
            out = subprocess.check_output(
                [sys.executable, "inspect_data.py", "validate"],
                stderr=subprocess.DEVNULL, timeout=90, text=True
            )
            by_type: dict[str, int] = {}
            task_issues: dict[str, list] = defaultdict(list)
            current_task = None
            for line in out.splitlines():
                m = re.match(r'\s+(\d+)×\s+(.+)', line)
                if m:
                    by_type[m.group(2).strip()] = int(m.group(1))
                if "Task:" in line:
                    tm = re.search(r'task_\w+', line)
                    current_task = tm.group() if tm else None
                elif current_task and "✗" in line:
                    task_issues[current_task].append(
                        line.strip().lstrip("✗").strip()
                    )
            return {"by_type": by_type, "task_issues": dict(task_issues),
                    "issue_count": sum(by_type.values())}
        except Exception as e:
            return {"by_type": {}, "task_issues": {}, "issue_count": 0,
                    "error": str(e)}

    def _sig_judge(self, cfg) -> dict:
        scores_file = cfg.data_dir / "scores.json"
        if not scores_file.exists():
            return {"scored": 0, "avg": None, "by_task": {}, "file_age_hours": None}
        try:
            scores  = json.loads(scores_file.read_text())
            by_task: dict[str, list] = defaultdict(list)
            for v in scores.values():
                s = v.get("score", 0)
                if s > 0:
                    by_task[v.get("task_id", "unknown")].append(s)
            flat = [s for lst in by_task.values() for s in lst]
            age  = round((time.time() - scores_file.stat().st_mtime) / 3600, 1)
            return {
                "scored":          len(flat),
                "avg":             round(sum(flat) / len(flat), 2) if flat else None,
                "by_task":         {t: round(sum(s) / len(s), 2)
                                    for t, s in by_task.items() if s},
                "file_age_hours":  age,
                "stale":           age > 12,
            }
        except Exception as e:
            return {"scored": 0, "avg": None, "by_task": {}, "error": str(e)}

    def _sig_benchmark_logs(self, cfg) -> dict:
        runs = []
        logs_dir = cfg.data_dir.parent / "logs"
        for log_file in sorted(logs_dir.glob("bench_*.log"),
                                key=lambda p: p.stat().st_mtime, reverse=True)[:5]:
            try:
                text = log_file.read_text(errors="replace")
                run: dict[str, Any] = {
                    "file":   log_file.name,
                    "mtime":  datetime.fromtimestamp(log_file.stat().st_mtime).isoformat(),
                    "scores": {},
                    "errors": [],
                }
                for m in re.finditer(
                    r'Task (task_\w+):\s*([01](?:\.\d+)?)\s*/\s*1\.0', text
                ):
                    run["scores"][m.group(1)] = float(m.group(2))
                for line in text.splitlines():
                    if re.search(r'\bERROR\b', line):
                        run["errors"].append(line.strip()[:200])
                if run["scores"]:
                    runs.append(run)
            except Exception:
                pass
        return {"runs": runs}

    def _sig_training(self, cfg) -> dict:
        for p in [Path("/tmp/finetune.log"), cfg.workspace / "finetune.log"]:
            if p.exists():
                text  = p.read_text(errors="replace")
                losses = [float(x) for x in re.findall(r"'loss':\s*([0-9.]+)", text)]
                return {"tail": "\n".join(text.splitlines()[-60:]),
                        "losses": losses[-30:],
                        "final_loss": losses[-1] if losses else None}
        return {"tail": "", "losses": [], "final_loss": None}

    # ── Hypothesis Generation ─────────────────────────────────────────────────

    def _generate_hypotheses(
        self, client, signals, state, prev, probes, rnd
    ) -> list[dict]:
        prompt = (self._prompt_initial(signals, state)
                  if rnd == 1 else
                  self._prompt_refine(signals, prev, probes))
        try:
            resp = client.messages.create(
                model=ANALYSIS_MODEL, max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = resp.content[0].text.strip()
            m = re.search(r'\[\s*\{.*\}\s*\]', raw, re.DOTALL)
            if not m:
                self.log(f"Could not parse hypotheses: {raw[:200]}")
                return prev
            hyps = json.loads(m.group())
            self.log(f"  {len(hyps)} hypotheses generated:")
            for h in hyps:
                self.log(f"    [{h.get('confidence','?'):6}] {h['id']}: "
                         f"{h['hypothesis'][:70]}")
            return hyps
        except Exception as e:
            self.log(f"Hypothesis generation error: {e}")
            return prev

    def _prompt_initial(self, signals: dict, state: AgentState) -> str:
        # Build version score table
        version_table = ""
        for entry in sorted(state.model_history, key=lambda h: h["version"]):
            avg = entry.get("avg_score", "?")
            version_table += f"  v{entry['version']} ({entry['ollama_name']}): avg={avg}\n"
            for task, score in sorted(entry.get("scores", {}).items()):
                version_table += f"    {task}: {score:.2f}\n"

        bench_txt = ""
        for run in signals["benchmark_logs"]["runs"][:3]:
            bench_txt += f"\n  {run['file']} ({run['mtime'][:10]}):\n"
            for task, sc in sorted(run["scores"].items()):
                bench_txt += f"    {task}: {sc:.2f}\n"
            if run["errors"][:2]:
                bench_txt += f"    ERRORS: {run['errors'][:2]}\n"

        return f"""You are diagnosing regressions in a fine-tuned LLM agent (Clawd on OpenClaw/PinchBench).

## Model Version History (benchmark scores per task)
{version_table or "  (no version history yet)"}

Best version: v{state.best_version} (avg={state.best_avg_score:.3f})
Current version: v{state.model_version}

## Dataset distribution (examples per task)
{json.dumps(signals['dataset']['by_task'], indent=2)}

## Structural validation issues
By type: {json.dumps(signals['validation']['by_type'], indent=2)}
By task: {json.dumps(signals['validation']['task_issues'], indent=2)}

## LLM judge scores (1–5 scale, from scores.json)
Scored: {signals['judge']['scored']} | Avg: {signals['judge']['avg']}
Age: {signals['judge'].get('file_age_hours','?')}h (stale={signals['judge'].get('stale','?')})
Per task: {json.dumps(signals['judge']['by_task'], indent=2)}

## Training loss
{signals['training']['losses']}  (final: {signals['training']['final_loss']})

## Benchmark logs
{bench_txt or "  (none found)"}

## Your task
Generate 4–6 specific, testable hypotheses about what is causing regressions.
Focus on score CHANGES across versions (not just current performance).

For each hypothesis, provide 1–2 test prompts. These will be run against EVERY
available model version to compare behavior. Pick prompts that would show
clear behavioral differences between versions.

For each hypothesis set evidence_type:
- "data_only"   — fully evaluable from dataset/judge/validation signals alone.
                  Do NOT add test_prompts (no Ollama call needed).
- "behavioral"  — requires comparing model outputs across versions.
                  Add 1–2 targeted test_prompts.

Only mark "behavioral" when you genuinely cannot tell from data alone.
This keeps probing fast: we only call Ollama when necessary, and only
against the regression boundary pair (best version vs first regressed version).

Return ONLY a valid JSON array:
[
  {{
    "id": "h1",
    "hypothesis": "...",
    "confidence": "high|medium|low",
    "evidence_type": "data_only|behavioral",
    "affected_tasks": ["task_XX", ...],
    "evidence": "which signals support this and how scores changed",
    "test_prompts": [
      {{
        "id": "p1",
        "description": "what behavioral difference we expect to see",
        "system": "You are Clawd, an AI agent. Use tools to complete tasks.",
        "user": "< the actual task prompt >",
        "tools_needed": ["write_file", "web_search"],
        "regression_signal": "what a regressed model does vs a good model"
      }}
    ]
  }}
]"""

    def _prompt_refine(self, signals, prev, probes) -> str:
        return f"""Refine hypotheses based on model probe results.

## Previous Hypotheses
{json.dumps(prev, indent=2)}

## Probe Results (all versions compared)
{json.dumps(probes, indent=2)}

Mark each hypothesis confirmed/refuted/unclear.
Generate refined or new hypotheses for remaining gaps.
Add new test prompts only where needed.

Return ONLY a valid JSON array with status field added:
[{{"id":"hN","status":"confirmed|refuted|unclear|new","hypothesis":"...","confidence":"...","affected_tasks":[...],"evidence":"...","test_prompts":[...]}}]"""

    # ── Model Probing ─────────────────────────────────────────────────────────

    def _select_probe_models(
        self, hypothesis: dict, all_models: list[dict]
    ) -> list[dict]:
        """
        Return the minimal set of models needed to test this hypothesis.

        Strategy: probe only the REGRESSION BOUNDARY — the last good version
        and the first regressed version. This is always the most informative
        pair regardless of total model count.

        If the hypothesis has evidence_type == "data_only" (evaluable purely
        from dataset signals), return empty list → skip probing entirely.

        Falls back to all models when there's no score history to identify
        a boundary (early runs with <2 versions).
        """
        if hypothesis.get("evidence_type") == "data_only":
            return []

        # Sort by version number, filter to those with known scores
        scored = [m for m in all_models
                  if isinstance(m.get("version"), int) and m.get("avg_score") is not None]
        scored.sort(key=lambda m: m["version"])

        if len(scored) < 2:
            # Not enough history — probe everything we have
            return all_models

        # Find the regression boundary: last version with score >= best,
        # first version with score < best (i.e. where it dropped)
        best_score = max(m["avg_score"] for m in scored)
        regression_threshold = best_score - 0.03  # 3pt drop counts as regression

        last_good  = None
        first_bad  = None
        for m in scored:
            if m["avg_score"] >= regression_threshold:
                last_good = m
            elif last_good is not None and first_bad is None:
                first_bad = m

        if last_good and first_bad:
            chosen = [last_good, first_bad]
            self.log(f"    Boundary: {last_good['label']} (good) vs "
                     f"{first_bad['label']} (regressed)")
            # Also include current version if it's different
            current = [m for m in scored if m == scored[-1] and m not in chosen]
            return chosen + current

        # No clear boundary (monotonically improving or all same) — probe all
        return all_models

    def _probe_models(self, hypotheses: list[dict], models: list[dict]) -> list[dict]:
        results = []
        for hyp in hypotheses:
            prompts = hyp.get("test_prompts", [])
            if not prompts:
                continue

            probe_models = self._select_probe_models(hyp, models)
            if not probe_models:
                self.log(f"  [{hyp['id']}] data_only — no Ollama probe needed")
                continue

            for p in prompts:
                self.log(f"  Probe [{hyp['id']}/{p['id']}]: {p['description'][:55]}"
                         f"  ({len(probe_models)} models)")
                probe: dict[str, Any] = {
                    "hypothesis_id":     hyp["id"],
                    "prompt_id":         p["id"],
                    "description":       p["description"],
                    "regression_signal": p.get("regression_signal", ""),
                    "models_probed":     [m["label"] for m in probe_models],
                    "responses":         {},
                }
                tools = self._get_tools(p.get("tools_needed", []))
                for model in probe_models:
                    self.log(f"    → {model['label']}")
                    resp = self._query_ollama(
                        model=model["ollama_name"],
                        system=p.get("system", "You are Clawd, an AI agent."),
                        user=p["user"],
                        tools=tools,
                    )
                    probe["responses"][model["label"]] = resp
                    if resp.get("error"):
                        self.log(f"      ✗ {resp['error'][:80]}")
                    elif resp.get("tool_calls"):
                        self.log(f"      Tools: {[t['name'] for t in resp['tool_calls']]}")
                    else:
                        self.log(f"      Text: {(resp.get('content') or '')[:80]}")
                results.append(probe)
        return results

    def _query_ollama(self, model, system, user, tools) -> dict:
        payload: dict[str, Any] = {
            "model":  model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }
        if tools:
            payload["tools"] = tools
        try:
            r = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=PROBE_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            msg  = data.get("message", {})
            tcs  = msg.get("tool_calls") or []
            return {
                "content":    msg.get("content", ""),
                "tool_calls": [{"name": tc["function"]["name"],
                                "arguments": tc["function"].get("arguments", {})}
                               for tc in tcs],
                "done_reason": data.get("done_reason", ""),
                "duration_ms": data.get("total_duration", 0) // 1_000_000,
            }
        except Exception as e:
            return {"error": str(e), "content": "", "tool_calls": []}

    def _get_tools(self, names: list[str]) -> list[dict]:
        return [TOOLS_BY_NAME[n] for n in names if n in TOOLS_BY_NAME] or CLAWD_TOOLS

    # ── JSON Extraction Helpers ───────────────────────────────────────────────

    @staticmethod
    def _extract_json_object(text: str) -> dict | None:
        """Robustly extract the outermost JSON object from a Claude response."""
        # 1. Strip markdown code fences
        text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
        text = re.sub(r'\s*```$', '', text.strip(), flags=re.MULTILINE)
        text = text.strip()

        # 2. Try direct parse first (clean responses)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 3. Find outermost {} by tracking brace depth
        start = text.find('{')
        if start == -1:
            return None
        depth = 0
        in_str = False
        escape = False
        for i, ch in enumerate(text[start:], start):
            if escape:
                escape = False
                continue
            if ch == '\\' and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    # ── Final Diagnosis ───────────────────────────────────────────────────────

    def _final_diagnosis(self, client, signals, state, hypotheses, probes) -> dict:
        self.log("Generating final diagnosis...")
        prompt = f"""Final regression diagnosis for Clawd fine-tuning project.

## Confirmed/Unclear Hypotheses
{json.dumps([h for h in hypotheses if h.get('status') != 'refuted'], indent=2)}

## Probe Results Summary
{json.dumps(probes, indent=2)}

## Model Version Scores
{json.dumps(state.model_history, indent=2)}

## Validation Issues
{json.dumps(signals['validation']['by_type'], indent=2)}

## Judge Score Staleness
Scored: {signals['judge']['scored']}, Age: {signals['judge'].get('file_age_hours','?')}h

Produce an actionable diagnosis. Return ONLY valid JSON:
{{
  "summary": "2-3 sentences on root cause of regression",
  "root_causes": [
    {{"rank":1,"cause":"...","confidence":"high|medium|low",
      "affected_tasks":[...],"evidence":"...","fix":"concrete action"}}
  ],
  "data_fixes": [
    {{"task":"task_XX","action":"delete|regenerate|topup|rejudge",
      "reason":"...","priority":"high|medium|low"}}
  ],
  "training_changes": ["..."],
  "v_next_watchpoints": ["what to verify after next training run"],
  "rerun_judge": true
}}"""
        try:
            resp = client.messages.create(
                model=ANALYSIS_MODEL, max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = resp.content[0].text.strip()
            diag = self._extract_json_object(raw)
            if diag is None:
                diag = {"summary": raw, "root_causes": [], "data_fixes": [],
                        "training_changes": [], "v_next_watchpoints": []}
        except Exception as e:
            diag = {"summary": f"Error: {e}", "root_causes": [], "data_fixes": [],
                    "training_changes": [], "v_next_watchpoints": []}

        return {
            "timestamp":  datetime.now().isoformat(),
            "model_history": state.model_history,
            "hypotheses": hypotheses,
            "probes":     probes,
            "diagnosis":  diag,
        }

    def _print_summary(self, diag: dict) -> None:
        print(f"\n{'═'*62}")
        print(f"  DIAGNOSIS")
        print(f"{'═'*62}")
        print(f"\n  {diag.get('summary','')}")
        print(f"\n  ROOT CAUSES:")
        for rc in diag.get("root_causes", []):
            print(f"  [{rc.get('rank','?')}][{rc.get('confidence','?'):6}] "
                  f"{rc.get('cause','')}")
            print(f"          Fix: {rc.get('fix','')}")
        print(f"\n  DATA FIXES:")
        for fix in diag.get("data_fixes", []):
            print(f"  [{fix.get('priority','?'):6}] {fix.get('task','?')}: "
                  f"{fix.get('action','?')} — {fix.get('reason','')}")
        print(f"\n  TRAINING CHANGES:")
        for c in diag.get("training_changes", []):
            print(f"    • {c}")
        print(f"\n  WATCH NEXT VERSION:")
        for w in diag.get("v_next_watchpoints", []):
            print(f"    • {w}")
        if diag.get("rerun_judge"):
            print(f"\n  ⚠  Re-run llm_judge.py run (scores may be stale)")
        print(f"{'═'*62}")


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg   = _load_config()
    state = AgentState()
    EvalAnalysisAgent().run(state, cfg)
