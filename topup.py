#!/usr/bin/env python3
"""
Targeted top-up script — regenerates examples only for underrepresented tasks.
Reads current dataset, calculates gaps, submits a new batch for deficit tasks only,
then appends collected results to existing train/val files.

Usage:
  python topup.py count              # show current counts vs target
  python topup.py submit             # submit batch for deficit tasks
  python topup.py status             # check batch status
  python topup.py collect            # collect and APPEND to existing train/val
  python topup.py run                # submit → poll → collect in one shot
"""

import os, sys, json, time, random, argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import anthropic
from utils.config import load_config

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
_cfg       = load_config()
DATA_DIR   = _cfg.data_dir
RAW_DIR    = DATA_DIR / "raw_topup"
BATCH_FILE = DATA_DIR / "topup_batch_id.txt"
TRAIN_FILE = _cfg.train_file
VAL_FILE   = _cfg.val_file

MODEL             = "claude-sonnet-4-5"
EXAMPLES_PER_CALL = int(os.environ.get("EXAMPLES_PER_CALL", "5"))
TARGET_PER_TASK   = _cfg.data.examples_per_task  # from config.yaml
VAL_PER_TASK      = 2

# ─────────────────────────────────────────────────────────────────────────────
# OPENCLAW SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────
OPENCLAW_SYSTEM = """\
You are Clawd, an autonomous AI agent powered by OpenClaw. You help users \
accomplish real-world tasks by using tools. Be direct and competent — \
start with action, not explanation. Get things done.

## Available Tools

read_file(path: str) -> str
  Read the contents of a file.

write_file(path: str, content: str) -> dict
  Write content to a file (creates parent dirs automatically).
  Returns: {"status": "success", "path": "..."}

create_directory(path: str) -> dict
  Create a directory and any needed parents.

list_files(directory: str = ".") -> list
  List files in a directory.

run_bash(command: str) -> dict
  Execute a shell command.
  Returns: {"stdout": "...", "stderr": "...", "exit_code": 0}

run_python(code: str) -> dict
  Execute Python code.
  Returns: {"output": "...", "error": null}

web_search(query: str, num_results: int = 5) -> list
  Search the web. Returns [{title, url, snippet}, ...]

fetch_url(url: str) -> str
  Fetch the text content of a URL.

create_calendar_event(title: str, date: str, time: str,
                      attendees: list = [], description: str = "") -> dict
  Create a calendar event. Date: YYYY-MM-DD, Time: HH:MM.

draft_email(to: str, subject: str, body: str, cc: str = "") -> dict
  Draft an email.

search_emails(query: str, folder: str = "inbox") -> list
  Search emails.

read_email(email_id: str) -> dict
  Read a full email by ID.

generate_image(prompt: str, filename: str) -> dict
  Generate an image using AI and save it to the workspace.
  Returns: {"status": "success", "path": "...", "size": "1024x1024"}
  NOTE: This is the ONLY way to create image files. Always use this for any image task.

read_memory(key: str = None) -> str
  Read from persistent memory.

write_memory(key: str, value: str) -> dict
  Write a key-value pair to persistent memory.

search_skills(query: str) -> list
  Search ClawHub for installable skills.

install_skill(name: str) -> dict
  Install a skill from ClawHub.

## Format

Use tool calls like this:
<tool_call>
{"name": "tool_name", "arguments": {"arg": "value"}}
</tool_call>

## Rules
- Always use RELATIVE paths for file operations (e.g. "report.txt" not "/workspace/tasks/report.txt")
- The `apply_patch` tool does NOT exist — always use `write_file` to create or update files
- To generate an image, you MUST call `generate_image` — never write a placeholder file
- When a tool fails, try an alternative approach — never give up after one error
- Confirm task completion with a brief summary at the end
"""

# ─────────────────────────────────────────────────────────────────────────────
# TASK DEFINITIONS (only deficit tasks included — no point shipping the rest)
# ─────────────────────────────────────────────────────────────────────────────
TASKS = {
    "task_09_files": {
        "name": "Python Project Structure",
        "prompt": (
            "Create a standard Python project structure for a library named "
            "'datautils'. Include: src/datautils/__init__.py with content, "
            "tests/test_datautils.py with sample test code, pyproject.toml "
            "with name 'datautils' and version 0.1.0, and README.md."
        ),
        "grading": [
            "src/datautils/ directory created",
            "tests/ directory created",
            "src/datautils/__init__.py created with substantive content",
            "tests/test_datautils.py created with at least one test",
            "pyproject.toml created containing 'datautils'",
            "pyproject.toml contains version '0.1.0'",
            "README.md created with title and description",
        ],
        "tools_needed": ["create_directory", "write_file"],
        "complexity": "medium",
    },
    "task_10_workflow": {
        "name": "Workflow Automation",
        "prompt": (
            "Read config.json, extract the API endpoint URL, write a Python script "
            "api_client.py that calls that endpoint with proper error handling and "
            "JSON parsing, then create NOTES.md documenting the process."
        ),
        "grading": [
            "config.json was read (visible in tool calls)",
            "api_client.py created with valid Python",
            "api_client.py uses an HTTP library and handles JSON",
            "api_client.py has try/except error handling",
            "NOTES.md created with clear documentation",
            "Documentation explains what the script does and how to use it",
        ],
        "tools_needed": ["read_file", "write_file"],
        "complexity": "hard",
    },
    "task_15_daily_summary": {
        "name": "Executive Briefing Synthesis",
        "prompt": (
            "Synthesize these 5 research documents into a 500–800 word executive "
            "briefing saved as executive_briefing.md: market_analysis.txt, "
            "competitor_intelligence.txt, customer_feedback.txt, "
            "product_updates.txt, industry_news.txt.\n\n"
            "Key facts to synthesize:\n"
            "- market_analysis.txt: S&P 500 at 5,842.31 (+1.2%), tech sector +2.1%\n"
            "- competitor_intelligence.txt: NexusAI launched at $99/user/month\n"
            "- customer_feedback.txt: 247 support tickets, top issue = API rate limiting\n"
            "- product_updates.txt: real-time collaboration shipped; DB migration Saturday 2am-6am EST\n"
            "- industry_news.txt: EU AI Act enforcement begins March 1, 2026\n"
            "- Cross-document insight: MegaCorp ($450K ARR) is evaluating competitors — churn risk."
        ),
        "grading": [
            "All 5 source files read",
            "executive_briefing.md created",
            "Word count 500–800",
            "Key data from each source represented",
            "Synthesizes cross-document connections (churn risk + competitor launch)",
            "Executive-appropriate structure and language",
        ],
        "tools_needed": ["read_file", "write_file"],
        "complexity": "hard",
    },
    "task_16_email_triage": {
        "name": "Email Triage",
        "prompt": (
            "Read all emails in the emails/ directory (13 total) and create "
            "triage_report.md with priority (P0–P4), category, and recommended "
            "action for each email. Sort by priority descending.\n\n"
            "Key correct assignments:\n"
            "- Production outage email → P0\n"
            "- $2M enterprise contract email → P0 or P1\n"
            "- Latency alert (linked to outage) → P0\n"
            "- Flash sale promotional spam → P4\n"
            "Report must be sorted by priority with a summary section."
        ),
        "grading": [
            "All 13 emails read",
            "triage_report.md created",
            "Production outage email correctly marked P0",
            "$2M contract email marked P0 or P1",
            "Spam/promotional emails marked P4",
            "Report sorted by priority",
            "Each entry has priority, category, and action",
        ],
        "tools_needed": ["list_files", "read_file", "write_file"],
        "complexity": "hard",
    },
    "task_17_email_search": {
        "name": "Email Search and Synthesis",
        "prompt": (
            "Search 12 email files in emails/ for content related to 'Project Alpha'. "
            "Filter out the 2 unrelated emails, then synthesize a thematic summary "
            "covering: timeline, budget, tech stack, and sales pipeline.\n\n"
            "Key facts present in the emails:\n"
            "- Tech stack: PostgreSQL/TimescaleDB, FastAPI, React, Kafka, Flink, Redis\n"
            "- Budget: $340K original → $410K after infrastructure assessment\n"
            "- Timeline: beta slipped April 21 → May 6; GA May 12 → May 27\n"
            "- Security: 5 issue categories including critical cross-tenant isolation gaps\n"
            "- Pipeline: 5 enterprise prospects = $1.85M immediate, $2.8M total ARR"
        ),
        "grading": [
            "Email files searched and relevant ones identified",
            "2 unrelated emails excluded from the summary",
            "Timeline covered (beta slipped April→May, GA to May 27)",
            "Budget covered ($340K original, $410K after assessment)",
            "Tech stack mentioned (PostgreSQL, FastAPI, React, Kafka)",
            "Pipeline/revenue figures included ($1.85M immediate ARR)",
        ],
        "tools_needed": ["list_files", "read_file", "write_file"],
        "complexity": "hard",
    },
    "task_18_market_research": {
        "name": "Competitive Market Research",
        "prompt": (
            "Create a competitive analysis of the enterprise observability/APM sector "
            "as market_research.md. Include: ≥5 competitors (Datadog, New Relic, "
            "Dynatrace, Splunk, Grafana Labs), pricing, differentiators, a comparison "
            "table, and current market trends (AI/ML integration, OpenTelemetry adoption, "
            "cloud-native consolidation)."
        ),
        "grading": [
            "market_research.md created",
            "At least 5 competitors covered",
            "Comparison table present",
            "Pricing information present for major players",
            "Market trends section covering AI/ML, OpenTelemetry, cloud-native",
            "web_search tool used for current data",
        ],
        "tools_needed": ["web_search", "write_file"],
        "complexity": "hard",
    },
    "task_21_openclaw_comprehension": {
        "name": "OpenClaw Report Comprehension",
        "prompt": (
            "Read openclaw_report.pdf and answer these questions in answer.txt "
            "(one answer per line, in order):\n"
            "1. How many community-built skills before filtering?\n"
            "2. How many skills after filtering?\n"
            "3. Largest skill category? (format: 'Category Name: count')\n"
            "4. Second-largest skill category? (same format)\n"
            "5. What filename defines an OpenClaw skill?\n"
            "6. What type of API does the OpenClaw gateway expose?\n"
            "7. What date was the skills registry data collected?\n"
            "8. How many new benchmark tasks does the paper propose?"
        ),
        "grading": [
            "openclaw_report.pdf read",
            "answer.txt created with 8 lines",
            "Q1 answer: 5,705",
            "Q2 answer: 2,999",
            "Q3 answer: AI & LLMs: 287",
            "Q4 answer: Search & Research: 253",
            "Q5 answer: SKILL.md",
            "Q6 answer: Typed WebSocket API",
            "Q7 answer: February 7, 2026",
            "Q8 answer: 6",
        ],
        "tools_needed": ["read_file", "write_file"],
        "complexity": "hard",
    },
    "task_13_image_gen": {
        "name": "Image Generation",
        "prompt": (
            "Generate an image of 'a friendly robot sitting in a cozy coffee shop, "
            "reading a book' and save it as robot_cafe.png."
        ),
        "grading": [
            "generate_image tool was called",
            "Prompt contains robot, coffee shop/café, and book",
            "Output saved as robot_cafe.png",
            "Task completed successfully",
        ],
        "tools_needed": ["generate_image"],
        "complexity": "easy",
    },
    "task_20_eli5_pdf": {
        "name": "ELI5 PDF Summary",
        "prompt": (
            "Read GPT4.pdf (the OpenAI GPT-4 Technical Report) and write a "
            "200–400 word ELI5 explanation to eli5_summary.txt. Use simple language "
            "and child-friendly analogies. Do not use jargon like 'multimodal', "
            "'transformer', 'RLHF', or 'benchmarks'."
        ),
        "grading": [
            "GPT4.pdf read",
            "eli5_summary.txt created",
            "Word count 200–400",
            "No technical jargon present",
            "Uses analogies and simple comparisons",
            "Accurate to the source material",
        ],
        "tools_needed": ["read_file", "write_file"],
        "complexity": "medium",
    },
    "task_22_second_brain": {
        "name": "Memory and Knowledge Management",
        "prompt": (
            "Store these user facts in memory/MEMORY.md and confirm they're saved:\n"
            "- Programming language: Rust\n"
            "- Started learning: January 15, 2024\n"
            "- Mentor: Dr. Elena Vasquez (Stanford)\n"
            "- Project: NeonDB (distributed key-value store)\n"
            "- Team phrase: 'purple elephant sunrise'"
        ),
        "grading": [
            "memory/ directory created",
            "memory/MEMORY.md created",
            "Rust as programming language stored",
            "January 15, 2024 as learning start date stored",
            "Dr. Elena Vasquez as mentor stored",
            "NeonDB project stored",
            "Team phrase 'purple elephant sunrise' stored",
            "Agent confirms all facts were saved",
        ],
        "tools_needed": ["create_directory", "write_file", "write_memory"],
        "complexity": "medium",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# VARIATION TYPES
# ─────────────────────────────────────────────────────────────────────────────
VARIATION_CONFIGS = [
    {
        "id": "happy_formal",
        "user_style": "formal and precise",
        "scenario": "Straightforward task. User provides all needed information. Agent completes in the most direct way. No errors.",
        "has_error": False,
    },
    {
        "id": "happy_casual",
        "user_style": "casual and conversational",
        "scenario": "Straightforward task. User is casual. Agent adapts and completes cleanly.",
        "has_error": False,
    },
    {
        "id": "vague_input",
        "user_style": "vague, leaving out one or two details",
        "scenario": "User omits a detail. Agent makes a smart assumption, mentions it briefly, then completes the task.",
        "has_error": False,
    },
    {
        "id": "error_recovery",
        "user_style": "normal",
        "scenario": "One tool call fails with a realistic error. Agent reads the error, tries an alternative, successfully completes the task.",
        "has_error": True,
    },
    {
        "id": "multi_tool_chain",
        "user_style": "detailed with multiple requirements",
        "scenario": "Task requires chaining 3+ tools. Agent plans the sequence, uses output from each step in the next.",
        "has_error": False,
    },
    {
        "id": "terse",
        "user_style": "extremely brief, one sentence",
        "scenario": "User gives a minimal request. Agent infers intent correctly and completes the task.",
        "has_error": False,
    },
    {
        "id": "detailed_user",
        "user_style": "verbose with very specific instructions",
        "scenario": "User over-specifies exactly how they want it done. Agent follows every specification precisely.",
        "has_error": False,
    },
    {
        "id": "one_clarification",
        "user_style": "ambiguous on exactly one critical detail",
        "scenario": "User's request has one genuinely ambiguous detail. Agent asks ONE targeted question, gets an answer, then completes the task.",
        "has_error": False,
    },
    {
        "id": "self_correction",
        "user_style": "normal",
        "scenario": "Agent makes a first attempt, inspects its own output, notices it is incomplete, corrects it, delivers final correct output.",
        "has_error": False,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def count_existing() -> dict[str, int]:
    counts = defaultdict(int)
    for path in [TRAIN_FILE, VAL_FILE]:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                counts[rec.get("task_id", "unknown")] += 1
            except json.JSONDecodeError:
                counts["__corrupt__"] += 1
    return counts


def compute_deficits(target: int = TARGET_PER_TASK, only_tasks: list | None = None) -> dict[str, int]:
    """Returns {task_id: n_additional_examples_needed} for deficit tasks only.
    If only_tasks is provided, restrict to those task IDs regardless of existing count."""
    counts = count_existing()
    deficits = {}
    task_ids = only_tasks if only_tasks else list(TASKS.keys())
    for task_id in task_ids:
        if task_id not in TASKS:
            continue
        current = counts.get(task_id, 0)
        # When targeting specific tasks, always add examples even if at target
        needed = target - current if not only_tasks else max(target - current, EXAMPLES_PER_CALL)
        if needed > 0:
            deficits[task_id] = needed
    return deficits


def build_meta_prompt(task_id: str, task: dict, variation: dict) -> str:
    tools_hint = (
        f"Key tools for this task: {', '.join(task['tools_needed'])}"
        if task["tools_needed"] else "This task may not need tool calls."
    )
    grading_list = "\n".join(f"  - {g}" for g in task["grading"])
    error_note = (
        "\n⚠️  ERROR RECOVERY REQUIRED: Include exactly one tool call that returns "
        "an error. Agent must read the error and try a different approach.\n"
        if variation["has_error"] else ""
    )

    return f"""\
You are generating synthetic fine-tuning data for training an LLM to act as \
an OpenClaw AI agent called Clawd.

## Task Being Tested
Name: {task["name"]}
ID: {task_id}
Complexity: {task["complexity"]}

Original task prompt:
\"\"\"{task["prompt"]}\"\"\"

## Grading Criteria (ALL must be satisfied)
{grading_list}

## Variation Type: {variation["id"]}
User style: {variation["user_style"]}
Scenario: {variation["scenario"]}
{error_note}

## Your Job
Generate {EXAMPLES_PER_CALL} diverse, complete agent conversation examples.

Each example MUST follow this exact JSON structure:
{{
  "user_message": "<user request, phrased per the variation style>",
  "turns": [
    {{
      "role": "assistant",
      "content": "<action or thinking — include <tool_call> tags when calling tools>"
    }},
    {{
      "role": "tool_result",
      "content": "<realistic JSON result from the tool>"
    }},
    ... more turns ...
    {{
      "role": "assistant",
      "content": "<final confirmation summarising what was done>"
    }}
  ]
}}

{tools_hint}

Tool call format (inside assistant content):
<tool_call>
{{"name": "tool_name", "arguments": {{"arg": "value"}}}}
</tool_call>

## Critical Rules
1. user_message must vary meaningfully across the {EXAMPLES_PER_CALL} examples.
2. Tool arguments must be realistic and specific to the task.
3. tool_result content must be plausible — use real-looking paths, timestamps, data.
4. For tasks with specific expected values (exact figures, filenames, dates), those \
exact values MUST appear in the tool results and final response.
5. The final assistant turn must confirm completion and satisfy ALL grading criteria.
6. Do NOT skip required tool calls — if the task needs a file created, call write_file.
7. For multi-document tasks, show ALL required files being read individually.
8. JSON ESCAPING IS MANDATORY: Inside any JSON string value, you MUST escape:
   - Double quotes as \\\"  (e.g. content with "quotes" → \\"quotes\\")
   - Backslashes as \\\\
   - Newlines as \\n
   Failure to escape will produce invalid JSON that cannot be parsed.
   Use single quotes inside code strings where possible to reduce escaping.

Return ONLY a valid JSON array of {EXAMPLES_PER_CALL} objects. No markdown, no preamble.
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMMANDS
# ─────────────────────────────────────────────────────────────────────────────
def cmd_count():
    counts   = count_existing()
    deficits = compute_deficits()

    # ── Diagnostic: show resolved paths so path issues are visible ────────────
    total_known = sum(v for k, v in counts.items() if k != "__corrupt__")
    print(f"\n  Data paths:")
    for path in [TRAIN_FILE, VAL_FILE]:
        size = f"{path.stat().st_size:,} bytes" if path.exists() else "NOT FOUND"
        print(f"    {path}  [{size}]")
    print(f"  Total records loaded: {total_known}")
    if counts.get("__corrupt__", 0):
        print(f"  WARNING: {counts['__corrupt__']} corrupt lines skipped")
    # Show any task_ids present in the data but not in TASKS dict
    unknown_ids = {k: v for k, v in counts.items() if k not in TASKS and k not in ("__corrupt__", "unknown")}
    if unknown_ids:
        top = sorted(unknown_ids.items(), key=lambda x: -x[1])[:5]
        print(f"  Tasks in data not shown below (top 5): {dict(top)}")

    print(f"\n{'═'*55}")
    print(f"  TASK COUNTS vs TARGET ({TARGET_PER_TASK})")
    print(f"{'═'*55}")
    print(f"  {'Task':<40} {'Have':>5}  {'Need':>5}  {'Gap':>5}")
    print(f"  {'─'*40} {'─'*5}  {'─'*5}  {'─'*5}")

    for task_id in TASKS:
        current = counts.get(task_id, 0)
        gap     = max(0, TARGET_PER_TASK - current)
        flag    = "  ⚠" if gap > 0 else "  ✓"
        print(f"  {task_id:<40} {current:>5}  {TARGET_PER_TASK:>5}  {gap:>5}{flag}")

    total_new = sum(deficits.values())
    batches   = sum((n + EXAMPLES_PER_CALL - 1) // EXAMPLES_PER_CALL for n in deficits.values())
    print(f"\n  New examples needed: ~{total_new}")
    print(f"  Batch requests:       {batches}")
    print(f"  Est. cost:           ~${batches * EXAMPLES_PER_CALL * 0.015:.2f}")


def cmd_submit(only_tasks: list | None = None):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    deficits = compute_deficits(only_tasks=only_tasks)
    if not deficits:
        print("All tasks are at or above target. Nothing to do.")
        sys.exit(0)

    print(f"Deficit tasks: {list(deficits.keys())}")

    # Build requests — use all variation types, enough to cover the deficit
    requests = []
    for task_id, needed in deficits.items():
        task = TASKS[task_id]
        # How many batch calls do we need?
        n_calls = (needed + EXAMPLES_PER_CALL - 1) // EXAMPLES_PER_CALL
        # Cycle through variations to maximise diversity
        for i in range(n_calls):
            variation = VARIATION_CONFIGS[i % len(VARIATION_CONFIGS)]
            custom_id = f"topup__{task_id}__{variation['id']}__{i:03d}"
            prompt    = build_meta_prompt(task_id, task, variation)
            # Scale token budget: 1 example ≈ 3000 tokens for complex tasks
            max_tok = min(16000, max(8192, EXAMPLES_PER_CALL * 3500))
            requests.append({
                "custom_id": custom_id,
                "params": {
                    "model": MODEL,
                    "max_tokens": max_tok,
                    "messages": [{"role": "user", "content": prompt}],
                },
            })

    client = anthropic.Anthropic(api_key=api_key)
    batch  = client.messages.batches.create(requests=requests)
    BATCH_FILE.write_text(batch.id)

    print(f"\n✓ Submitted {len(requests)} requests × {EXAMPLES_PER_CALL} = "
          f"~{len(requests) * EXAMPLES_PER_CALL} new examples")
    print(f"  Batch ID: {batch.id}  (saved to {BATCH_FILE})")
    print(f"\n  python topup.py status    # check progress")
    print(f"  python topup.py collect   # when done")


def cmd_status():
    if not BATCH_FILE.exists():
        print("No topup batch found. Run: python topup.py submit")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client  = anthropic.Anthropic(api_key=api_key)
    batch   = client.messages.batches.retrieve(BATCH_FILE.read_text().strip())
    counts  = batch.request_counts

    print(f"Status: {batch.processing_status}")
    print(f"  Processing: {counts.processing}")
    print(f"  Succeeded:  {counts.succeeded}")
    print(f"  Errored:    {counts.errored}")
    if batch.processing_status == "ended":
        print("\n✓ Ready. Run: python topup.py collect")


def parse_example(raw: dict, task_id: str) -> dict | None:
    try:
        user_msg = raw.get("user_message") or raw.get("user") or raw.get("prompt")
        turns    = raw.get("turns") or raw.get("conversation") or []
        if not user_msg or not turns:
            return None

        messages = [{"role": "system", "content": OPENCLAW_SYSTEM}]
        messages.append({"role": "user", "content": str(user_msg)})
        for turn in turns:
            role = turn.get("role", "")
            if role == "assistant":
                messages.append({"role": "assistant", "content": str(turn["content"])})
            elif role in ("tool_result", "tool", "function"):
                messages.append({"role": "tool", "content": str(turn["content"])})

        return {"task_id": task_id, "messages": messages} if len(messages) >= 3 else None
    except Exception:
        return None


def extract_json_array(text: str):
    """Extract and merge ALL JSON arrays from a response.

    Claude sometimes returns multiple separate JSON arrays (one per example)
    when examples are long. This merges all of them into a single list.
    """
    import re
    text = text.strip()
    merged = []

    # Find all ```json ... ``` or ``` ... ``` code blocks
    blocks = re.findall(r'```(?:json)?\s*\n([\s\S]*?)\n```', text)
    for block in blocks:
        try:
            result = json.loads(block.strip())
            if isinstance(result, list):
                merged.extend(result)
            elif isinstance(result, dict):
                merged.append(result)
        except json.JSONDecodeError:
            pass

    if merged:
        return merged

    # No code blocks — try finding all top-level JSON arrays in the text
    for m in re.finditer(r'\[[\s\S]*?\](?=\s*(?:\[|$|```))', text):
        try:
            result = json.loads(m.group(0))
            if isinstance(result, list):
                merged.extend(result)
        except json.JSONDecodeError:
            pass

    if merged:
        return merged

    # Last resort — parse the whole thing
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else [result]
    except json.JSONDecodeError:
        pass

    return None


def cmd_collect():
    if not BATCH_FILE.exists():
        print("No topup batch found. Run: python topup.py submit")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client  = anthropic.Anthropic(api_key=api_key)
    batch   = client.messages.batches.retrieve(BATCH_FILE.read_text().strip())

    if batch.processing_status != "ended":
        print(f"Not ready yet: {batch.processing_status}")
        sys.exit(1)

    new_examples: dict[str, list] = defaultdict(list)
    errors = parse_failures = 0

    for result in tqdm(client.messages.batches.results(batch.id), desc="Collecting"):
        parts   = result.custom_id.split("__")
        task_id = parts[1] if len(parts) >= 2 else "unknown"

        if result.result.type != "succeeded":
            errors += 1
            continue

        raw_text = result.result.message.content[0].text
        (RAW_DIR / f"{result.custom_id}.json").write_text(raw_text)

        examples = extract_json_array(raw_text)
        if examples is None:
            parse_failures += 1
            continue

        for ex in examples:
            parsed = parse_example(ex, task_id)
            if parsed:
                new_examples[task_id].append(parsed)

    # Read existing dataset
    existing_train = []
    existing_val   = []
    if TRAIN_FILE.exists():
        existing_train = [json.loads(l) for l in TRAIN_FILE.read_text().splitlines() if l.strip()]
    if VAL_FILE.exists():
        existing_val   = [json.loads(l) for l in VAL_FILE.read_text().splitlines() if l.strip()]

    # Count existing per task to decide val allocation
    existing_val_counts = defaultdict(int)
    for r in existing_val:
        existing_val_counts[r["task_id"]] += 1

    new_train, new_val = [], []
    per_task_added: dict[str, int] = defaultdict(int)
    for task_id, examples in new_examples.items():
        random.shuffle(examples)
        # Only add to val if this task is still short of VAL_PER_TASK
        val_deficit = max(0, VAL_PER_TASK - existing_val_counts[task_id])
        t_val   = examples[:val_deficit]
        t_train = examples[val_deficit:]
        new_val.extend(t_val)
        new_train.extend(t_train)
        per_task_added[task_id] = len(t_train) + len(t_val)

    # Write merged dataset
    with open(TRAIN_FILE, "w") as f:
        for r in existing_train + new_train:
            f.write(json.dumps(r) + "\n")
    with open(VAL_FILE, "w") as f:
        for r in existing_val + new_val:
            f.write(json.dumps(r) + "\n")

    # Count final totals per task for logging
    final_counts: dict[str, int] = defaultdict(int)
    for r in existing_train + new_train + existing_val + new_val:
        final_counts[r["task_id"]] += 1

    print(f"\n{'─'*50}")
    print(f"✓ Top-up complete")
    print(f"  API errors:     {errors}")
    print(f"  Parse failures: {parse_failures}")
    print(f"\n  Per-task additions:")
    for task_id in sorted(set(list(new_examples.keys()) + list(per_task_added.keys()))):
        added = per_task_added.get(task_id, 0)
        total = final_counts.get(task_id, 0)
        print(f"    {task_id:<40}  +{added:>3}  (total: {total})")
    print(f"\n  New train:      {len(new_train)}")
    print(f"  New val:        {len(new_val)}")
    print(f"  Total train:    {len(existing_train) + len(new_train)}")
    print(f"  Total val:      {len(existing_val)   + len(new_val)}")
    print(f"\nRun: python topup.py count  to verify gaps are closed")
    return len(new_train) + len(new_val)


def cmd_run(only_tasks: list | None = None):
    cmd_submit(only_tasks=only_tasks)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client  = anthropic.Anthropic(api_key=api_key)
    batch_id = BATCH_FILE.read_text().strip()
    print("\nPolling every 2 minutes...")
    while True:
        time.sleep(120)
        batch  = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  [{time.strftime('%H:%M')}] processing={counts.processing} "
              f"succeeded={counts.succeeded} errored={counts.errored}")
        if batch.processing_status == "ended":
            break
    return cmd_collect()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["count", "submit", "status", "collect", "run"],
        help="count: show gaps | submit: send batch | status: check | collect: append | run: all",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task IDs to target (e.g. task_09_files,task_13_image_gen). "
             "Overrides deficit calculation — always generates examples for these tasks.",
    )
    args = parser.parse_args()

    only_tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else None

    if args.command in ("submit", "run"):
        n_new = {"submit": cmd_submit, "run": cmd_run}[args.command](only_tasks=only_tasks)
        if args.command == "run" and n_new == 0:
            print("\n[topup] ERROR: 0 new examples collected — aborting.", file=sys.stderr)
            sys.exit(2)
    else:
        {"count": cmd_count, "status": cmd_status, "collect": cmd_collect}[args.command]()
