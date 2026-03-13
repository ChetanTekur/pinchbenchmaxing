#!/usr/bin/env python3
"""
LLM-based quality judge for generated training examples.
Uses Claude to score each example against the task's grading criteria.

Usage:
  python llm_judge.py run              # score all examples (saves scores.json)
  python llm_judge.py report           # show score distribution (needs scores.json)
  python llm_judge.py filter --min 3   # remove examples scoring below 3/5
  python llm_judge.py sample --bad     # show low-scoring examples for inspection

Cost: ~$2 for 834 examples with claude-sonnet-4-5
"""

import os, sys, json, time, argparse, random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import anthropic

TRAIN_FILE  = Path("/workspace/synthbench/data/train.jsonl")
VAL_FILE    = Path("/workspace/synthbench/data/val.jsonl")
SCORES_FILE = Path("/workspace/synthbench/data/scores.json")

JUDGE_MODEL = "claude-sonnet-4-5"
MIN_SCORE   = 3   # default threshold (out of 5)

# ─────────────────────────────────────────────────────────────────────────────
# TASK GRADING CRITERIA  (same as in generate.py — used as judge rubric)
# ─────────────────────────────────────────────────────────────────────────────
TASK_CRITERIA = {
    "task_00_sanity": ["Agent responded with any greeting or confirmation"],
    "task_01_calendar": [
        "An .ics file is created", "Date is next Tuesday",
        "Time is 3:00 PM", "Attendee john@example.com is included",
        "Summary contains 'Project Sync'", "Description mentions roadmap",
    ],
    "task_02_stock": [
        "stock_report.txt is created", "Contains 'AAPL'",
        "Contains a numeric price value", "Contains a date reference",
        "Market summary is at least 50 characters",
    ],
    "task_03_blog": [
        "blog_post.md is created", "400-600 words",
        "Proper markdown structure with headings", "At least 4 distinct advantages",
        "Content is specific to software developers",
    ],
    "task_04_weather": [
        "weather.py is created", "Valid Python syntax",
        "Uses an HTTP library", "References San Francisco",
        "Has try/except error handling", "Has print statement",
    ],
    "task_05_summary": [
        "summary_output.txt created", "Exactly 3 paragraphs",
        "150-250 words", "Covers AI in healthcare overview",
        "Mentions imaging, drug discovery, predictive analytics",
        "Addresses privacy, bias, explainability",
    ],
    "task_06_events": [
        "events.md created", "Exactly 5 conference entries",
        "Each entry has name, dates, location, and URL",
        "Events appear to be real tech conferences",
    ],
    "task_07_email": [
        "email_draft.txt created", "Professional polite tone",
        "Clearly declines the meeting", "Gives scheduling conflict reason",
        "Offers to reschedule", "Has greeting and closing",
    ],
    "task_08_memory": [
        "answer.txt created", "Contains 'June 1, 2024'",
        "Agent called read_file on project_notes.txt",
        "No hallucinated information",
    ],
    "task_09_files": [
        "src/datautils/ directory created", "tests/ directory created",
        "__init__.py created with content", "test_datautils.py with a test",
        "pyproject.toml with 'datautils' and version 0.1.0", "README.md created",
    ],
    "task_10_workflow": [
        "config.json was read", "api_client.py created with valid Python",
        "Uses HTTP library and handles JSON", "Has error handling",
        "NOTES.md created with documentation",
    ],
    "task_11_config_update": [
        "localhost replaced with prod-db.example.com",
        "DB names updated to prod", "Logging changed to warn",
        "API endpoint updated to https://api.example.com",
        "Files remain valid JSON/YAML",
    ],
    "task_12_skill_search": [
        "search_skills was called", "install_skill was called",
        "An output file was created", "Task completed end-to-end",
    ],
    "task_13_image_gen": [
        "generate_image was called",
        "Prompt mentions robot, coffee shop, and book",
        "Output saved as robot_cafe.png",
    ],
    "task_14_humanizer": [
        "ai_blog.txt was read", "humanized_blog.txt created",
        "Removed robotic phrases like 'In today's world'",
        "Uses contractions and natural language",
        "Original meaning preserved",
    ],
    "task_15_daily_summary": [
        "All 5 source files read", "executive_briefing.md created",
        "500-800 words", "Key data from each source represented",
        "Synthesizes connections between documents",
    ],
    "task_16_email_triage": [
        "All 13 emails read", "triage_report.md created",
        "Outage email marked P0", "Contract email P0 or P1",
        "Spam marked P4", "Sorted by priority",
    ],
    "task_17_email_search": [
        "Emails searched and filtered",
        "Budget ($340K→$410K) covered",
        "Tech stack mentioned (PostgreSQL, FastAPI, React, Kafka)",
        "Pipeline revenue figures included",
    ],
    "task_18_market_research": [
        "market_research.md created", "At least 5 competitors covered",
        "Comparison table present", "Pricing information present",
        "Market trends section present", "web_search tool used",
    ],
    "task_19_spreadsheet_summary": [
        "Both files read",
        "Revenue $119,900 correct", "Profit $47,960 correct",
        "Top region: East", "Top product: Widget B",
        "Top employee: Alice Chen",
    ],
    "task_20_eli5_pdf": [
        "GPT4.pdf read", "eli5_summary.txt created",
        "200-400 words", "No technical jargon",
        "Uses analogies and simple language",
    ],
    "task_21_openclaw_comprehension": [
        "openclaw_report.pdf read", "8 answers in answer.txt",
        "Answer 1: 5,705", "Answer 2: 2,999",
        "Answer 5: SKILL.md", "Answer 6: Typed WebSocket API",
    ],
    "task_22_second_brain": [
        "memory/MEMORY.md created", "All 5 facts stored",
        "Rust as programming language", "Dr. Elena Vasquez as mentor",
        "NeonDB project stored", "'purple elephant sunrise' stored",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# JUDGE PROMPT
# ─────────────────────────────────────────────────────────────────────────────
def build_judge_prompt(rec: dict) -> str:
    task_id  = rec.get("task_id", "unknown")
    messages = rec.get("messages", [])
    criteria = TASK_CRITERIA.get(task_id, ["Agent completed the task correctly"])

    criteria_str = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(criteria))

    # Build a readable transcript (skip system prompt — too long)
    transcript_parts = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            continue
        content = msg["content"]
        if len(content) > 1500:
            content = content[:1500] + f"\n[... {len(msg['content'])-1500} chars truncated ...]"
        transcript_parts.append(f"[{role.upper()}]\n{content}")

    transcript = "\n\n".join(transcript_parts)

    return f"""\
You are evaluating a synthetic training example for fine-tuning an AI agent.

## Task: {task_id}

## Grading Criteria (what a correct response must satisfy)
{criteria_str}

## Conversation Transcript
{transcript}

## Your Job
Score this training example on a scale of 1–5:

5 = Excellent: satisfies ALL criteria, realistic tool calls, clean completion
4 = Good: satisfies most criteria, minor gaps or slightly unrealistic details
3 = Acceptable: satisfies core criteria but missing some secondary ones
2 = Poor: satisfies fewer than half the criteria or has significant issues
1 = Bad: fails the task, no relevant tool calls, wrong output, or nonsensical

Also flag any specific problems you see (wrong tool names, missing required files,
incorrect expected values, truncated response, etc.)

Respond in this exact JSON format (no other text):
{{
  "score": <1-5>,
  "criteria_met": [<list of criterion numbers that are satisfied, e.g. [1,2,4]>],
  "issues": [<list of specific problems, or empty list if none>],
  "reasoning": "<one sentence summary>"
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────────────────────
def load_records() -> list[dict]:
    records = []
    for path in [TRAIN_FILE, VAL_FILE]:
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    rec["_source"] = path.name
                    records.append(rec)
    return records


def score_record(client: anthropic.Anthropic, rec: dict) -> dict:
    prompt = build_judge_prompt(rec)
    try:
        resp = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(raw)
        return {
            "score":        result.get("score", 0),
            "criteria_met": result.get("criteria_met", []),
            "issues":       result.get("issues", []),
            "reasoning":    result.get("reasoning", ""),
            "error":        None,
        }
    except Exception as e:
        return {"score": 0, "criteria_met": [], "issues": [], "reasoning": "", "error": str(e)}


def cmd_run():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client  = anthropic.Anthropic(api_key=api_key)
    records = load_records()

    # Resume if scores.json already exists
    existing = {}
    if SCORES_FILE.exists():
        existing = json.loads(SCORES_FILE.read_text())
        print(f"Resuming — {len(existing)} scores already saved, "
              f"{len(records)-len(existing)} remaining")

    scores = dict(existing)
    errors = 0

    for i, rec in enumerate(tqdm(records, desc="Judging")):
        # Use task_id + user message as a stable key
        user_msgs = [m for m in rec.get("messages", []) if m["role"] == "user"]
        key = rec.get("task_id", "?") + "|" + (user_msgs[0]["content"][:80] if user_msgs else str(i))

        if key in scores:
            continue  # already scored

        result = score_record(client, rec)
        result["task_id"] = rec.get("task_id")
        result["source"]  = rec.get("_source")
        scores[key] = result

        if result["error"]:
            errors += 1

        # Save after every 10 records so we can resume on crash
        if i % 10 == 0:
            SCORES_FILE.write_text(json.dumps(scores, indent=2))

        # Gentle rate limiting
        time.sleep(0.3)

    SCORES_FILE.write_text(json.dumps(scores, indent=2))
    print(f"\n✓ Scored {len(scores)} examples  |  errors: {errors}")
    print(f"  Saved to: {SCORES_FILE}")
    print(f"\nNext: python llm_judge.py report")


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────
def cmd_report():
    if not SCORES_FILE.exists():
        print("No scores.json found. Run: python llm_judge.py run")
        sys.exit(1)

    scores     = json.loads(SCORES_FILE.read_text())
    all_scores = [v["score"] for v in scores.values() if v["score"] > 0]
    by_task    = defaultdict(list)
    for v in scores.values():
        by_task[v.get("task_id", "unknown")].append(v["score"])

    dist = defaultdict(int)
    for s in all_scores:
        dist[s] += 1

    print(f"\n{'═'*55}")
    print(f"  LLM JUDGE REPORT  ({len(all_scores)} examples scored)")
    print(f"{'═'*55}")
    print(f"  Average score:  {sum(all_scores)/len(all_scores):.2f} / 5")
    print(f"  Score ≥ 3:      {sum(1 for s in all_scores if s >= 3)} "
          f"({100*sum(1 for s in all_scores if s >= 3)/len(all_scores):.1f}%)")
    print(f"  Score ≥ 4:      {sum(1 for s in all_scores if s >= 4)} "
          f"({100*sum(1 for s in all_scores if s >= 4)/len(all_scores):.1f}%)")
    print()
    print(f"  Score distribution:")
    for s in [5, 4, 3, 2, 1]:
        bar = "█" * dist[s]
        print(f"    {s}/5  {dist[s]:>4}  {bar}")

    print(f"\n  Per-task averages:")
    print(f"  {'Task':<40} {'Avg':>5}  {'Min':>4}  {'Count':>6}")
    print(f"  {'─'*40} {'─'*5}  {'─'*4}  {'─'*6}")
    for task_id in sorted(by_task):
        task_scores = [s for s in by_task[task_id] if s > 0]
        if task_scores:
            avg = sum(task_scores) / len(task_scores)
            mn  = min(task_scores)
            flag = "  ⚠" if avg < 3.0 else ""
            print(f"  {task_id:<40} {avg:>5.2f}  {mn:>4}  {len(task_scores):>6}{flag}")

    # Common issues
    all_issues = []
    for v in scores.values():
        all_issues.extend(v.get("issues", []))
    if all_issues:
        issue_counts = defaultdict(int)
        for issue in all_issues:
            issue_counts[issue[:80]] += 1
        print(f"\n  Top issues reported:")
        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    {count:>4}×  {issue}")

    print(f"\nTo remove low-scoring examples: python llm_judge.py filter --min 3")


# ─────────────────────────────────────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────────────────────────────────────
def cmd_filter(min_score: int):
    if not SCORES_FILE.exists():
        print("No scores.json found. Run: python llm_judge.py run")
        sys.exit(1)

    scores  = json.loads(SCORES_FILE.read_text())
    records = load_records()

    kept = 0
    removed = 0
    by_task_kept = defaultdict(list)

    for rec in records:
        user_msgs = [m for m in rec.get("messages", []) if m["role"] == "user"]
        key = rec.get("task_id", "?") + "|" + (user_msgs[0]["content"][:80] if user_msgs else "")

        score_data = scores.get(key, {})
        score = score_data.get("score", 5)  # default keep if not scored

        if score >= min_score:
            by_task_kept[rec.get("task_id")].append(rec)
            kept += 1
        else:
            removed += 1

    # Re-split train/val
    VAL_PER_TASK = 2
    train_out, val_out = [], []
    for task_id, exs in by_task_kept.items():
        random.shuffle(exs)
        val_cut = min(VAL_PER_TASK, len(exs))
        val_out.extend(exs[:val_cut])
        train_out.extend(exs[val_cut:])

    with open(TRAIN_FILE, "w") as f:
        for r in train_out:
            r.pop("_source", None)
            f.write(json.dumps(r) + "\n")
    with open(VAL_FILE, "w") as f:
        for r in val_out:
            r.pop("_source", None)
            f.write(json.dumps(r) + "\n")

    print(f"\n✓ Filtered at score ≥ {min_score}")
    print(f"  Kept:    {kept}")
    print(f"  Removed: {removed}")
    print(f"  Train:   {len(train_out)}  →  {TRAIN_FILE}")
    print(f"  Val:     {len(val_out)}   →  {VAL_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE BAD
# ─────────────────────────────────────────────────────────────────────────────
def cmd_sample_bad():
    if not SCORES_FILE.exists():
        print("No scores.json found. Run: python llm_judge.py run")
        sys.exit(1)

    scores = json.loads(SCORES_FILE.read_text())
    bad    = [(k, v) for k, v in scores.items() if v.get("score", 5) <= 2]
    bad.sort(key=lambda x: x[1].get("score", 5))

    print(f"\nLow-scoring examples (score ≤ 2): {len(bad)} total\n")
    for key, data in bad[:10]:
        task_id = data.get("task_id", "?")
        score   = data.get("score", "?")
        reason  = data.get("reasoning", "")
        issues  = data.get("issues", [])
        print(f"  [{score}/5] {task_id}")
        print(f"          {reason}")
        for issue in issues:
            print(f"          ✗ {issue}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("run")
    sub.add_parser("report")
    sub.add_parser("sample").add_argument("--bad", action="store_true")

    p_filter = sub.add_parser("filter")
    p_filter.add_argument("--min", type=int, default=MIN_SCORE,
                          dest="min_score", help="Minimum score to keep (default 3)")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        cmd_run()
    elif args.command == "report":
        cmd_report()
    elif args.command == "filter":
        cmd_filter(args.min_score)
    elif args.command == "sample":
        cmd_sample_bad()
