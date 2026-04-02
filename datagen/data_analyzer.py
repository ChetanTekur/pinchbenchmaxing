#!/usr/bin/env python3
"""
Smart data quality analyzer — replaces hard-coded count caps with signal-based decisions.

Uses three signals per task to decide what action the orchestrator should take:
  1. Benchmark score  — how the model actually performs (0-100%)
  2. Judge score      — average LLM judge quality rating (1-5)
  3. Example count    — how many training examples currently exist

Decision matrix:
  LEAVE_ALONE    — benchmark HIGH (>=80%) regardless of judge score
  GENERATE       — benchmark LOW/MID + count LOW (<30)
  REGENERATE     — benchmark LOW + judge LOW (<3.5) — bad quality data, replace it
  ADVERSARIAL    — benchmark LOW + judge HIGH + count HIGH (>=50) — data is fine but model fails
  TRIM           — benchmark MID + count HIGH (>100) — too much data risks forgetting
  INVESTIGATE    — task regressed from previous version
  INFRASTRUCTURE — benchmark 0% + count HIGH (>=50) + known infra-dependent task

Usage:
  python -m datagen.data_analyzer                    # analyze all tasks
  python -m datagen.data_analyzer --task task_12     # single task
  python -m datagen.data_analyzer --json             # machine-readable output

Importable:
  from datagen.data_analyzer import get_task_recommendation
  rec = get_task_recommendation("task_12_skill_search", cfg, state)
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from utils.config import load_config, Config
from agents.base import TASK_IDS


# ── Recommendation enum ─────────────────────────────────────────────────────

class Recommendation(str, Enum):
    LEAVE_ALONE    = "LEAVE_ALONE"
    GENERATE       = "GENERATE"
    REGENERATE     = "REGENERATE"
    ADVERSARIAL    = "ADVERSARIAL"
    TRIM           = "TRIM"
    INVESTIGATE    = "INVESTIGATE"
    INFRASTRUCTURE = "INFRASTRUCTURE"

    def __str__(self):
        return self.value


# ── Thresholds ───────────────────────────────────────────────────────────────

BENCH_HIGH    = 0.80   # benchmark score >= this is "passing"
BENCH_MID_LO  = 0.30   # benchmark score at or below this is "failing"
BENCH_PROTECTED = 0.70 # tasks at or above this have gold floor protection
JUDGE_HIGH    = 4.5     # judge score >= this means data quality is good
JUDGE_LOW     = 3.5     # judge score below this means data quality is bad
COUNT_HIGH    = 50      # enough examples that more won't help
COUNT_LOW     = 30      # too few examples to learn the task
COUNT_BLOAT   = 100     # so many examples it may cause forgetting of other tasks
MAX_ADD_PER_CYCLE = 20  # max new examples per task per improvement cycle




# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class TaskSignals:
    task_id: str
    bench_score: float       # 0.0 - 1.0
    judge_score: float       # 1.0 - 5.0 (0.0 = no scores)
    example_count: int
    prev_bench_score: float  # -1.0 = no previous data
    regressed: bool


@dataclass
class TaskRecommendation:
    task_id: str
    action: Recommendation
    reason: str
    signals: TaskSignals

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "action": str(self.action),
            "reason": self.reason,
            "bench_score": self.signals.bench_score,
            "judge_score": self.signals.judge_score,
            "example_count": self.signals.example_count,
            "prev_bench_score": self.signals.prev_bench_score,
            "regressed": self.signals.regressed,
        }


# ── Signal loading ───────────────────────────────────────────────────────────

def _load_example_counts(train_file: Path) -> dict[str, int]:
    """Count training examples per task from train.jsonl."""
    counts: Counter = Counter()
    if not train_file.exists():
        return dict(counts)
    for line in train_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            counts[rec.get("task_id", "unknown")] += 1
        except json.JSONDecodeError:
            continue
    return dict(counts)


def _load_judge_scores(scores_file: Path) -> dict[str, float]:
    """Load average judge score per task from scores.json."""
    if not scores_file.exists():
        return {}
    raw = json.loads(scores_file.read_text())
    from collections import defaultdict
    task_scores = defaultdict(list)
    for key, data in raw.items():
        task_id = data.get("task_id", key.split("::")[0] if "::" in key else "unknown")
        s = data.get("score", 0)
        if s > 0:
            task_scores[task_id].append(s)
    return {t: sum(ss) / len(ss) for t, ss in task_scores.items() if ss}


def _load_benchmark_scores(state_dict: dict) -> dict[str, float]:
    """Extract per-task benchmark scores from loop state."""
    return {k: float(v) for k, v in state_dict.get("scores", {}).items()}


def _load_previous_scores(data_dir: Path, current_version: int) -> dict[str, float]:
    """Load benchmark scores from the previous version's snapshot in model_history."""
    state_file = data_dir / "loop_state.json"
    if not state_file.exists():
        return {}

    state = json.loads(state_file.read_text())
    history = state.get("model_history", [])
    if not history:
        return {}

    # Find the entry just before current_version
    prev_entries = [h for h in history if h.get("version", 0) < current_version]
    if not prev_entries:
        return {}

    prev_entry = max(prev_entries, key=lambda h: h.get("version", 0))
    return {k: float(v) for k, v in prev_entry.get("scores", {}).items()}


def _load_state_file(data_dir: Path) -> dict:
    """Load loop_state.json as raw dict."""
    state_file = data_dir / "loop_state.json"
    if state_file.exists():
        return json.loads(state_file.read_text())
    return {}


# ── Core decision logic ─────────────────────────────────────────────────────

def _decide(signals: TaskSignals) -> tuple[Recommendation, str]:
    """
    Pure decision function. No I/O — just logic on signals.

    Priority order:
      1. Regression detection (highest priority — something broke)
      2. Infrastructure issues (don't waste budget on infra problems)
      3. Benchmark HIGH → leave alone
      4. Benchmark LOW + judge LOW → regenerate
      5. Benchmark LOW + count LOW → generate more
      6. Benchmark LOW + judge HIGH + count HIGH → adversarial or infra
      7. Benchmark MID + count HIGH → trim
      8. Benchmark MID + count LOW → generate
      9. Benchmark MID + judge HIGH → leave for training to improve
    """
    bench = signals.bench_score
    judge = signals.judge_score
    count = signals.example_count

    # -1. HARD FLOOR — if below training minimum, always generate regardless of score
    HARD_FLOOR = 30
    if count < HARD_FLOOR:
        return (
            Recommendation.GENERATE,
            f"Below training minimum: {count} examples (<{HARD_FLOOR}). "
            f"Must generate to reach minimum, bench={bench:.0%}."
        )

    # 0. HARD CEILING — but allow regeneration if task is failing badly
    HARD_CEILING = 80
    if count >= HARD_CEILING:
        if bench <= BENCH_MID_LO and count > COUNT_BLOAT:
            return (
                Recommendation.TRIM,
                f"Hard ceiling: {count} examples (>{HARD_CEILING}) but bench={bench:.0%}. "
                f"Trim bad examples to ~{COUNT_HIGH}, then regenerate."
            )
        if bench <= BENCH_MID_LO:
            # Task has lots of data but still fails — data quality is the problem
            return (
                Recommendation.REGENERATE,
                f"Hard ceiling ({count} examples) but bench only {bench:.0%}. "
                f"Data quality is bad — trim and regenerate, don't add more."
            )
        return (
            Recommendation.LEAVE_ALONE,
            f"Hard ceiling: {count} examples (>={HARD_CEILING}). No more generation."
        )

    # 1. Regression detection
    if signals.regressed and signals.prev_bench_score > 0:
        drop = signals.prev_bench_score - bench
        return (
            Recommendation.INVESTIGATE,
            f"Regressed {drop:.0%} (was {signals.prev_bench_score:.0%}, now {bench:.0%}). "
            f"Check if rebalancing or new data harmed this task."
        )

    # 2. Benchmark HIGH — working well, don't touch
    if bench >= BENCH_HIGH:
        return (
            Recommendation.LEAVE_ALONE,
            f"Benchmark {bench:.0%} >= {BENCH_HIGH:.0%} threshold. Task is passing."
        )

    # 3. Benchmark ZERO with lots of good data
    # Only mark as INFRASTRUCTURE for tasks that truly can't work without
    # external services (web search, image gen). Tasks that need Python
    # libraries (pandas, pdfplumber) are data problems, not infra problems.
    TRULY_INFRA = {"task_02_stock", "task_04_weather", "task_06_events",
                   "task_13_image_gen", "task_18_market_research"}
    if bench == 0.0 and count >= COUNT_HIGH and judge >= JUDGE_HIGH:
        if signals.task_id in TRULY_INFRA:
            return (
                Recommendation.INFRASTRUCTURE,
                f"Scores 0% with {count} examples (judge avg {judge:.1f}). "
                f"This task requires external services -- likely an infrastructure issue."
            )
        return (
            Recommendation.REGENERATE,
            f"Scores 0% with {count} examples (judge {judge:.1f}). "
            f"Data may teach wrong approach. Regenerate with diagnosis context."
        )

    # 4. Benchmark LOW + judge LOW → data quality problem
    if bench <= BENCH_MID_LO and judge > 0 and judge < JUDGE_LOW:
        return (
            Recommendation.REGENERATE,
            f"Benchmark {bench:.0%} with low judge score ({judge:.1f}). "
            f"Data quality is poor — regenerate with higher quality threshold."
        )

    # 5. Benchmark LOW + count LOW → not enough data
    if bench <= BENCH_MID_LO and count < COUNT_LOW:
        return (
            Recommendation.GENERATE,
            f"Benchmark {bench:.0%} with only {count} examples (need >= {COUNT_LOW}). "
            f"Generate more training data."
        )

    # 6. Benchmark LOW + enough data + good judge → adversarial
    if bench <= BENCH_MID_LO and count >= COUNT_HIGH and (judge >= JUDGE_HIGH or judge == 0):
        return (
            Recommendation.ADVERSARIAL,
            f"Benchmark {bench:.0%} despite {count} examples"
            + (f" (judge {judge:.1f})" if judge > 0 else "")
            + ". Standard data isn't helping — try adversarial generation."
        )

    # 7. Benchmark LOW + moderate data → generate to fill up
    if bench <= BENCH_MID_LO:
        return (
            Recommendation.GENERATE,
            f"Benchmark {bench:.0%} with {count} examples. "
            f"Need more diverse training data."
        )

    # ── MID-range benchmark (20-80%) ──

    # 8. MID + bloated count → trim to prevent forgetting
    if bench < BENCH_HIGH and count > COUNT_BLOAT:
        return (
            Recommendation.TRIM,
            f"Benchmark {bench:.0%} with {count} examples (> {COUNT_BLOAT} cap). "
            f"Excess data may cause forgetting of other tasks. Trim to ~{COUNT_HIGH}."
        )

    # 9. MID + too few examples → generate
    if bench < BENCH_HIGH and count < COUNT_LOW:
        return (
            Recommendation.GENERATE,
            f"Benchmark {bench:.0%} with only {count} examples. "
            f"More data could push this over the threshold."
        )

    # 10. MID + reasonable data → leave alone for now (training may improve it)
    return (
        Recommendation.LEAVE_ALONE,
        f"Benchmark {bench:.0%} with {count} examples"
        + (f" (judge {judge:.1f})" if judge > 0 else "")
        + ". In the improvement zone — training iterations should help."
    )


# ── Public API ───────────────────────────────────────────────────────────────

def analyze_all(cfg: Config, state_dict: Optional[dict] = None) -> list[TaskRecommendation]:
    """
    Analyze all tasks and return recommendations.

    Args:
        cfg: Config object (for file paths)
        state_dict: Raw loop_state.json dict. If None, loaded from disk.

    Returns:
        List of TaskRecommendation for all 23 tasks.
    """
    data_dir = cfg.data_dir

    # Load signals
    counts = _load_example_counts(cfg.train_file)
    judge_scores = _load_judge_scores(data_dir / "scores.json")

    if state_dict is None:
        state_dict = _load_state_file(data_dir)

    bench_scores = _load_benchmark_scores(state_dict)
    current_version = state_dict.get("model_version", 0)
    prev_scores = _load_previous_scores(data_dir, current_version)

    recommendations = []
    for task_id in TASK_IDS:
        bench = bench_scores.get(task_id, 0.0)
        prev_bench = prev_scores.get(task_id, -1.0)
        judge = judge_scores.get(task_id, 0.0)
        count = counts.get(task_id, 0)

        # Detect regression: score dropped by >= 20 percentage points
        regressed = (
            prev_bench >= 0
            and bench < prev_bench
            and (prev_bench - bench) >= 0.20
        )

        signals = TaskSignals(
            task_id=task_id,
            bench_score=bench,
            judge_score=round(judge, 2),
            example_count=count,
            prev_bench_score=round(prev_bench, 2),
            regressed=regressed,
        )

        action, reason = _decide(signals)
        recommendations.append(TaskRecommendation(
            task_id=task_id,
            action=action,
            reason=reason,
            signals=signals,
        ))

    return recommendations


def get_task_recommendation(
    task_id: str,
    cfg: Config,
    state_dict: Optional[dict] = None,
) -> str:
    """
    Get recommendation for a single task. Returns the action string.

    Usage from orchestrator:
        from datagen.data_analyzer import get_task_recommendation
        rec = get_task_recommendation("task_12_skill_search", cfg, state.to_dict())
        # Returns: "GENERATE", "LEAVE_ALONE", "ADVERSARIAL", etc.
    """
    all_recs = analyze_all(cfg, state_dict)
    for rec in all_recs:
        if rec.task_id == task_id:
            return str(rec.action)
    return str(Recommendation.GENERATE)  # unknown task → generate by default


def get_tasks_needing_action(
    cfg: Config,
    state_dict: Optional[dict] = None,
    actions: Optional[list[str]] = None,
) -> list[str]:
    """
    Return task IDs that need a specific action (or any action other than LEAVE_ALONE).

    Usage from orchestrator:
        tasks = get_tasks_needing_action(cfg, state.to_dict(), ["GENERATE", "REGENERATE"])
    """
    all_recs = analyze_all(cfg, state_dict)
    if actions is None:
        return [r.task_id for r in all_recs if r.action != Recommendation.LEAVE_ALONE]
    action_set = {a.upper() for a in actions}
    return [r.task_id for r in all_recs if str(r.action) in action_set]


# ── Report generation ────────────────────────────────────────────────────────

def _print_report(recommendations: list[TaskRecommendation]) -> None:
    """Print a human-readable analysis report."""
    # Group by action
    by_action: dict[str, list[TaskRecommendation]] = {}
    for rec in recommendations:
        by_action.setdefault(str(rec.action), []).append(rec)

    total = len(recommendations)
    print(f"\n{'='*74}")
    print(f"  DATA QUALITY ANALYZER — {total} tasks analyzed")
    print(f"{'='*74}")

    # Summary counts
    print(f"\n  Action Summary:")
    for action in Recommendation:
        recs = by_action.get(str(action), [])
        if recs:
            tasks_str = ", ".join(r.task_id.replace("task_", "t") for r in recs)
            print(f"    {str(action):<16} {len(recs):>2} tasks  [{tasks_str}]")

    # Detailed per-task table
    print(f"\n  {'Task':<35} {'Bench':>6} {'Judge':>6} {'Count':>6}  {'Action':<16}")
    print(f"  {'─'*35} {'─'*6} {'─'*6} {'─'*6}  {'─'*16}")

    for rec in recommendations:
        s = rec.signals
        bench_str = f"{s.bench_score:.0%}" if s.bench_score >= 0 else "n/a"
        judge_str = f"{s.judge_score:.1f}" if s.judge_score > 0 else "n/a"

        # Markers for notable conditions
        markers = ""
        if s.regressed:
            markers += " [REGRESSED]"
        if rec.action == Recommendation.INFRASTRUCTURE:
            markers += " [INFRA]"

        print(
            f"  {s.task_id:<35} {bench_str:>6} {judge_str:>6} {s.example_count:>6}  "
            f"{str(rec.action):<16}{markers}"
        )

    # Detailed reasons for non-LEAVE_ALONE tasks
    actionable = [r for r in recommendations if r.action != Recommendation.LEAVE_ALONE]
    if actionable:
        print(f"\n  Actionable Details:")
        print(f"  {'─'*70}")
        for rec in actionable:
            print(f"  {rec.task_id}")
            print(f"    -> {rec.reason}")

    # High-level advice
    gen_count = len(by_action.get("GENERATE", []))
    adv_count = len(by_action.get("ADVERSARIAL", []))
    infra_count = len(by_action.get("INFRASTRUCTURE", []))
    regen_count = len(by_action.get("REGENERATE", []))
    trim_count = len(by_action.get("TRIM", []))

    print(f"\n  Suggested Next Steps:")
    if infra_count:
        tasks = [r.task_id for r in by_action.get("INFRASTRUCTURE", [])]
        print(f"    1. Fix infrastructure for {infra_count} tasks: {tasks}")
        print(f"       (Don't generate data — fix the setup first)")
    if gen_count:
        print(f"    {'2' if infra_count else '1'}. Generate data for {gen_count} tasks")
    if regen_count:
        print(f"    {'3' if infra_count else '2'}. Regenerate {regen_count} tasks with quality issues")
    if adv_count:
        print(f"    -> Generate adversarial examples for {adv_count} stuck tasks")
    if trim_count:
        print(f"    -> Trim {trim_count} bloated tasks before next training run")

    print(f"\n{'='*74}\n")


def save_report(recommendations: list[TaskRecommendation], output_file: Path) -> None:
    """Save machine-readable report to JSON."""
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_tasks": len(recommendations),
        "summary": {},
        "tasks": {},
    }

    for action in Recommendation:
        recs = [r for r in recommendations if r.action == action]
        report["summary"][str(action)] = [r.task_id for r in recs]

    for rec in recommendations:
        report["tasks"][rec.task_id] = rec.to_dict()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2))
    print(f"  Report saved: {output_file}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze data quality and recommend per-task actions."
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="Analyze a single task (e.g. task_12_skill_search)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output machine-readable JSON instead of human report",
    )
    args = parser.parse_args()

    cfg = load_config()
    recommendations = analyze_all(cfg)

    if args.task:
        # Fuzzy match: allow "task_12" to match "task_12_skill_search"
        matched = [r for r in recommendations if r.task_id.startswith(args.task)]
        if not matched:
            print(f"No task found matching '{args.task}'")
            print(f"Available: {', '.join(TASK_IDS)}")
            return
        recommendations = matched

    if args.json:
        report = {r.task_id: r.to_dict() for r in recommendations}
        print(json.dumps(report, indent=2))
    else:
        _print_report(recommendations)

    # Always save report
    report_file = cfg.data_dir / "analyzer_report.json"
    save_report(
        recommendations if not args.task else analyze_all(cfg),
        report_file,
    )


if __name__ == "__main__":
    main()
