#!/usr/bin/env python3
"""
PinchBench Fine-Tuning Data Generator
Generates ~1035 synthetic OpenClaw agent traces via Claude Batch API

Usage:
  python generate.py submit    # Build and submit batch to Claude API
  python generate.py status    # Check batch processing status
  python generate.py collect   # Collect results and write train/val JSONL
  python generate.py run       # submit → wait → collect in one shot
"""

import os, sys, json, time, random, argparse
from pathlib import Path
from tqdm import tqdm
import anthropic

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("/workspace/data")
RAW_DIR    = DATA_DIR / "raw"
BATCH_FILE = DATA_DIR / "batch_id.txt"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE   = DATA_DIR / "val.jsonl"

MODEL              = "claude-sonnet-4-5"
EXAMPLES_PER_CALL  = 5    # Claude generates N examples per batch request
VARIATIONS         = 9    # variation types × tasks = total batch requests
                          # 23 tasks × 9 variations × 5 examples = 1,035 total

VAL_PER_TASK = 2          # 2 examples per task held out for validation = 46 val

# ─────────────────────────────────────────────────────────────────────────────
# OPENCLAW TOOL SCHEMA  (injected into every training example's system prompt)
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
  Returns: {"status": "success", "event_id": "...", "ics_path": "..."}

draft_email(to: str, subject: str, body: str, cc: str = "") -> dict
  Draft an email. Returns: {"status": "drafted", "draft_id": "..."}

search_emails(query: str, folder: str = "inbox") -> list
  Search emails. Returns [{id, from, subject, date, snippet}, ...]

read_email(email_id: str) -> dict
  Read a full email by ID.

generate_image(prompt: str, filename: str) -> dict
  Generate an image and save to workspace.
  Returns: {"status": "success", "path": "...", "size": "1024x1024"}

read_memory(key: str = None) -> str
  Read from persistent memory. None returns all stored memory.

write_memory(key: str, value: str) -> dict
  Write a key-value pair to persistent memory.

search_skills(query: str) -> list
  Search ClawHub for installable skills.
  Returns [{name, description, author, installs}, ...]

install_skill(name: str) -> dict
  Install a skill from ClawHub.
  Returns: {"status": "installed", "skill": "...", "tools_added": [...]}

## Format

Use tool calls like this:
<tool_call>
{"name": "tool_name", "arguments": {"arg1": "value1", "arg2": "value2"}}
</tool_call>

Tool results will be returned as:
<tool_result>
{"status": "success", ...}
</tool_result>

## Rules
- Working directory for file tasks: /workspace/tasks/
- When a tool fails, adapt — try an alternative approach
- Confirm task completion with a brief summary at the end
- One task at a time; stay focused on what was asked
"""

# ─────────────────────────────────────────────────────────────────────────────
# TASKS  (23 PinchBench tasks with exact grading criteria)
# ─────────────────────────────────────────────────────────────────────────────
TASKS = [
    {
        "id": "task_00_sanity",
        "name": "Sanity Check",
        "prompt": "Say 'Hello, I'm ready!' to confirm you can respond.",
        "grading": ["Agent responded with any greeting or confirmation message"],
        "tools_needed": [],
        "complexity": "trivial",
    },
    {
        "id": "task_01_calendar",
        "name": "Calendar Event Creation",
        "prompt": (
            "Schedule a meeting called 'Project Sync' for next Tuesday at 3:00 PM "
            "with attendee john@example.com and include a note about Q1 roadmap "
            "discussion. Save it as an .ics (iCalendar) file in the workspace."
        ),
        "grading": [
            "An .ics file is created in the workspace",
            "Date is next Tuesday (dynamically calculated)",
            "Start time is 3:00 PM (DTSTART contains T15)",
            "Attendee john@example.com is present in the file",
            "Summary/title contains 'Project Sync'",
            "Description mentions roadmap",
        ],
        "tools_needed": ["create_calendar_event", "write_file"],
        "complexity": "medium",
    },
    {
        "id": "task_02_stock",
        "name": "Stock Price Research",
        "prompt": (
            "Research Apple's (AAPL) current stock price and create stock_report.txt "
            "with the current price, the date, and a brief market summary."
        ),
        "grading": [
            "stock_report.txt is created in the workspace",
            "File contains 'AAPL'",
            "File contains a numeric price value (e.g. $185.42)",
            "File contains a date reference",
            "Market summary is at least 50 characters",
            "File has 3 or more lines",
        ],
        "tools_needed": ["web_search", "fetch_url", "write_file"],
        "complexity": "medium",
    },
    {
        "id": "task_03_blog",
        "name": "Blog Post Writing",
        "prompt": (
            "Write a ~500-word markdown blog post about the advantages of remote work "
            "for software developers. Save it as blog_post.md."
        ),
        "grading": [
            "blog_post.md is created",
            "Word count is between 400 and 600",
            "Has proper markdown structure: intro, headings, conclusion",
            "Mentions at least 4 distinct advantages",
            "Content is specific to software developers (not generic)",
        ],
        "tools_needed": ["write_file"],
        "complexity": "medium",
    },
    {
        "id": "task_04_weather",
        "name": "Weather Script Creation",
        "prompt": (
            "Write a Python script weather.py that fetches and displays weather data "
            "for San Francisco using the wttr.in API "
            "(https://wttr.in/San_Francisco?format=j1)."
        ),
        "grading": [
            "weather.py is created",
            "Valid Python syntax (parseable by AST)",
            "Uses an HTTP library (requests, urllib, or http.client)",
            "References 'San Francisco' or the wttr.in URL",
            "Has try/except error handling",
            "Has a print statement for output",
            "Has a function definition or if __name__ == '__main__' block",
        ],
        "tools_needed": ["write_file"],
        "complexity": "medium",
    },
    {
        "id": "task_05_summary",
        "name": "Document Summarization",
        "prompt": (
            "Read summary_source.txt (an article about AI in healthcare) and write "
            "a 3-paragraph summary (150–250 words) to summary_output.txt. "
            "Paragraph 1: main topic/overview. Paragraph 2: key applications "
            "(imaging, drug discovery, predictive analytics). Paragraph 3: "
            "challenges and future (privacy, bias, explainability)."
        ),
        "grading": [
            "summary_output.txt is created",
            "Has exactly 3 paragraphs",
            "Total word count 150–250",
            "Covers AI in healthcare overview",
            "Mentions key applications (imaging, drug discovery, predictive analytics)",
            "Addresses challenges (privacy, algorithmic bias, explainability)",
        ],
        "tools_needed": ["read_file", "write_file"],
        "complexity": "medium",
    },
    {
        "id": "task_06_events",
        "name": "Technology Events Research",
        "prompt": (
            "Find 5 legitimate upcoming technology conferences and compile them into "
            "events.md. For each include: name, dates, location (city/country), "
            "and website URL."
        ),
        "grading": [
            "events.md is created",
            "Contains exactly 5 entries",
            "Each entry has name, dates, location, and URL",
            "Events are real, verifiable technology conferences",
            "Consistent markdown formatting with headings, lists, or links",
        ],
        "tools_needed": ["web_search", "write_file"],
        "complexity": "medium",
    },
    {
        "id": "task_07_email",
        "name": "Email Drafting",
        "prompt": (
            "Draft a professional email declining a meeting due to scheduling "
            "conflicts. Save as email_draft.txt. Include: greeting, clear decline, "
            "reason (scheduling conflict), and offer to reschedule."
        ),
        "grading": [
            "email_draft.txt is created",
            "Professional and polite tone throughout",
            "Meeting is clearly declined",
            "Scheduling conflict is given as the reason",
            "Offer to reschedule is included",
            "Has a greeting and a closing/signature",
            "Length is 50–150 words",
        ],
        "tools_needed": ["write_file"],
        "complexity": "easy",
    },
    {
        "id": "task_08_memory",
        "name": "Memory and Context Retrieval",
        "prompt": (
            "Read project_notes.txt and extract the beta release deadline. "
            "Save the answer to answer.txt."
        ),
        "grading": [
            "answer.txt is created",
            "Contains the correct date: June 1, 2024",
            "Answer is clear and unambiguous",
            "Agent called read_file on project_notes.txt (visible in tool calls)",
            "No hallucinated or fabricated information",
        ],
        "tools_needed": ["read_file", "write_file"],
        "complexity": "easy",
    },
    {
        "id": "task_09_files",
        "name": "Python Project Structure",
        "prompt": (
            "Create a standard Python project structure for a library named "
            "'datautils'. Include: src/datautils/__init__.py with content, "
            "tests/test_datautils.py with sample test code, pyproject.toml "
            "with name 'datautils' and version 0.1.0, and README.md."
        ),
        "grading": [
            "src/datautils/ directory exists",
            "tests/ directory exists",
            "src/datautils/__init__.py created with substantive content",
            "tests/test_datautils.py created with at least one test",
            "pyproject.toml created containing 'datautils'",
            "pyproject.toml contains version '0.1.0'",
            "README.md created with title and description",
        ],
        "tools_needed": ["create_directory", "write_file"],
        "complexity": "medium",
    },
    {
        "id": "task_10_workflow",
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
    {
        "id": "task_11_config_update",
        "name": "Production Configuration Update",
        "prompt": (
            "Update the production config files config/settings.json and "
            "config/database.yml: replace all 'localhost' with "
            "'prod-db.example.com', rename DB from *_dev and *_test to *_prod, "
            "change logging level from 'debug' to 'warn', update API endpoint "
            "to 'https://api.example.com'."
        ),
        "grading": [
            "localhost replaced with prod-db.example.com in both files",
            "DB names updated to prod variants",
            "Logging level changed from debug to warn",
            "API endpoint updated to https://api.example.com",
            "Both files remain valid JSON/YAML after edits",
        ],
        "tools_needed": ["read_file", "write_file"],
        "complexity": "medium",
    },
    {
        "id": "task_12_skill_search",
        "name": "Skill Search and Discovery",
        "prompt": (
            "Search ClawHub for a data visualization skill, install the best "
            "result, and use it to create a simple chart saved to chart_output.png."
        ),
        "grading": [
            "search_skills tool was called",
            "install_skill tool was called with a relevant skill name",
            "An output file (chart or image) was created",
            "Task completed end-to-end with the installed skill",
        ],
        "tools_needed": ["search_skills", "install_skill", "write_file"],
        "complexity": "hard",
    },
    {
        "id": "task_13_image_gen",
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
    {
        "id": "task_14_humanizer",
        "name": "Text Humanization",
        "prompt": (
            "Read ai_blog.txt and rewrite it to sound more natural and human. "
            "Remove: 'In today's world', 'Furthermore', 'Moreover', hedging phrases, "
            "overly formal language without contractions, and generic conclusions. "
            "Save the result to humanized_blog.txt."
        ),
        "grading": [
            "ai_blog.txt was read",
            "humanized_blog.txt was created",
            "Robotic opener phrases removed",
            "Contractions and natural language used",
            "Original meaning and content preserved",
            "More conversational and engaging tone",
        ],
        "tools_needed": ["read_file", "write_file", "install_skill"],
        "complexity": "medium",
    },
    {
        "id": "task_15_daily_summary",
        "name": "Executive Briefing Synthesis",
        "prompt": (
            "Synthesize these 5 research documents into a 500–800 word executive "
            "briefing saved as executive_briefing.md: market_analysis.txt, "
            "competitor_intelligence.txt, customer_feedback.txt, "
            "product_updates.txt, industry_news.txt."
        ),
        "grading": [
            "All 5 source files read",
            "executive_briefing.md created",
            "Word count 500–800",
            "Key data from each source represented (S&P, NexusAI, API issues, "
            "collab feature, EU AI Act)",
            "Synthesizes cross-document connections (e.g. churn risk + competitor)",
            "Executive-appropriate structure and language",
        ],
        "tools_needed": ["read_file", "write_file"],
        "complexity": "hard",
    },
    {
        "id": "task_16_email_triage",
        "name": "Email Triage",
        "prompt": (
            "Read all emails in the emails/ directory (13 total) and create "
            "triage_report.md with priority (P0–P4), category, and recommended "
            "action for each email. Sort by priority descending."
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
    {
        "id": "task_17_email_search",
        "name": "Email Search and Synthesis",
        "prompt": (
            "Search 12 email files in emails/ for content related to 'Project Alpha'. "
            "Filter out the 2 unrelated emails, then synthesize a thematic summary "
            "covering: timeline, budget, tech stack, and sales pipeline."
        ),
        "grading": [
            "Email files searched and relevant ones identified",
            "2 unrelated emails excluded from the summary",
            "Timeline covered (beta slipped April→May, GA slipped to May 27)",
            "Budget covered ($340K original, $410K after assessment)",
            "Tech stack mentioned (PostgreSQL, FastAPI, React, Kafka)",
            "Pipeline/revenue figures included ($1.85M immediate ARR)",
        ],
        "tools_needed": ["list_files", "read_file", "write_file"],
        "complexity": "hard",
    },
    {
        "id": "task_18_market_research",
        "name": "Competitive Market Research",
        "prompt": (
            "Create a competitive analysis of the enterprise observability/APM sector "
            "as market_research.md. Include: ≥5 competitors (Datadog, New Relic, "
            "Dynatrace, Splunk, Grafana Labs), pricing, differentiators, a comparison "
            "table, and current market trends."
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
    {
        "id": "task_19_spreadsheet_summary",
        "name": "Spreadsheet Data Analysis",
        "prompt": (
            "Read sales.csv and data.xlsx. Produce a markdown summary report "
            "with: total revenue, total profit, top region, top product, "
            "top department, and top employee."
        ),
        "grading": [
            "Both files read",
            "Total revenue: $119,900 correctly stated",
            "Total profit: $47,960 correctly stated",
            "Top region: East",
            "Top product: Widget B",
            "Top employee: Alice Chen",
        ],
        "tools_needed": ["read_file", "run_python", "write_file"],
        "complexity": "medium",
    },
    {
        "id": "task_20_eli5_pdf",
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
    {
        "id": "task_21_openclaw_comprehension",
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
    {
        "id": "task_22_second_brain",
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
]

# ─────────────────────────────────────────────────────────────────────────────
# VARIATION TYPES  (9 types × 5 examples each = 45 per task)
# ─────────────────────────────────────────────────────────────────────────────
VARIATION_CONFIGS = [
    {
        "id": "happy_formal",
        "user_style": "formal and precise",
        "scenario": (
            "Straightforward task. User provides all needed information upfront. "
            "Agent completes the task in the most direct way. No errors occur."
        ),
        "has_error": False,
    },
    {
        "id": "happy_casual",
        "user_style": "casual and conversational, may use abbreviations or slang",
        "scenario": (
            "Straightforward task. User is casual (e.g. 'hey can you...', 'just '). "
            "Agent adapts to the tone and completes the task cleanly."
        ),
        "has_error": False,
    },
    {
        "id": "vague_input",
        "user_style": "vague, leaving out one or two details",
        "scenario": (
            "User omits a specific detail (e.g. doesn't say which file, or when). "
            "Agent makes a smart, reasonable assumption and proceeds — "
            "mentioning the assumption briefly, then completing the task."
        ),
        "has_error": False,
    },
    {
        "id": "error_recovery",
        "user_style": "normal",
        "scenario": (
            "One tool call fails with a realistic error (file not found, network "
            "timeout, permission denied). The tool_result shows the error. "
            "Agent reads the error, tries an alternative approach, and successfully "
            "completes the task."
        ),
        "has_error": True,
    },
    {
        "id": "multi_tool_chain",
        "user_style": "detailed with multiple requirements",
        "scenario": (
            "Task requires chaining 3 or more tools. Agent plans the sequence "
            "explicitly, uses output from each step in the next, and delivers "
            "a complete result."
        ),
        "has_error": False,
    },
    {
        "id": "terse",
        "user_style": "extremely brief, one sentence or less",
        "scenario": (
            "User gives a minimal request (e.g. 'schedule the usual tuesday meeting'). "
            "Agent correctly infers intent from minimal context and completes the task."
        ),
        "has_error": False,
    },
    {
        "id": "detailed_user",
        "user_style": "verbose with very specific step-by-step instructions",
        "scenario": (
            "User over-specifies exactly how they want the task done. "
            "Agent follows every specification precisely without cutting corners."
        ),
        "has_error": False,
    },
    {
        "id": "one_clarification",
        "user_style": "ambiguous on exactly one critical detail",
        "scenario": (
            "User's request has one genuinely ambiguous detail. Agent asks ONE "
            "targeted clarifying question, user answers, then agent immediately "
            "completes the task."
        ),
        "has_error": False,
    },
    {
        "id": "self_correction",
        "user_style": "normal",
        "scenario": (
            "Agent makes a first attempt, inspects its own output (e.g. reads the "
            "file it just wrote), notices it's incomplete or slightly wrong, "
            "corrects it, and delivers the final correct output."
        ),
        "has_error": False,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# META-PROMPT  (sent to Claude Sonnet to generate each batch of 5 examples)
# ─────────────────────────────────────────────────────────────────────────────
def build_meta_prompt(task: dict, variation: dict) -> str:
    tools_hint = (
        f"Key tools for this task: {', '.join(task['tools_needed'])}"
        if task["tools_needed"]
        else "This task may not need tool calls."
    )

    grading_list = "\n".join(f"  - {g}" for g in task["grading"])

    error_note = ""
    if variation["has_error"]:
        error_note = (
            "\n⚠️  ERROR RECOVERY REQUIRED: Include exactly one tool call that "
            "returns an error in its tool_result. The agent must read the error "
            "and try a different approach to complete the task."
        )

    return f"""\
You are generating synthetic fine-tuning data for training an LLM to act as \
an OpenClaw AI agent called Clawd.

## Task Being Tested
Name: {task["name"]}
ID: {task["id"]}
Complexity: {task["complexity"]}

Original task prompt (what a real user will ask):
\"\"\"{task["prompt"]}\"\"\"

## Grading Criteria (the agent MUST satisfy all of these)
{grading_list}

## Variation Type: {variation["id"]}
User style: {variation["user_style"]}
Scenario: {variation["scenario"]}
{error_note}

## Your Job
Generate {EXAMPLES_PER_CALL} diverse, complete agent conversation examples for \
this task+variation combination.

Each example must follow this EXACT JSON structure:
{{
  "user_message": "<the user's request, rephrased per the variation style>",
  "turns": [
    {{
      "role": "assistant",
      "content": "<assistant thinking or action — include <tool_call> tags when calling a tool>"
    }},
    {{
      "role": "tool_result",
      "content": "<realistic JSON result from the tool>"
    }},
    ... more turns as needed ...
    {{
      "role": "assistant",
      "content": "<final confirmation message summarising what was done>"
    }}
  ]
}}

{tools_hint}

Tool call format (inside assistant content):
<tool_call>
{{"name": "tool_name", "arguments": {{"arg": "value"}}}}
</tool_call>

## Rules
1. user_message must vary meaningfully across the 5 examples (different words, \
different details provided, different tone per the variation style).
2. Tool call arguments must be realistic and task-appropriate.
3. tool_result content must be plausible (real-looking file paths, timestamps, \
data values). For tasks with specific expected values (e.g. task_08 with the \
June 1, 2024 deadline, task_21 with exact counts), the tool results must \
contain those exact values.
4. The final assistant turn must confirm task completion and satisfy ALL grading \
criteria listed above.
5. Do NOT skip tool calls — if the task requires creating a file, actually call \
write_file.

Return ONLY a valid JSON array of {EXAMPLES_PER_CALL} objects. No markdown, \
no explanation, just the JSON array.
"""


# ─────────────────────────────────────────────────────────────────────────────
# BATCH SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────
def build_requests() -> list[dict]:
    """Build all batch request objects (23 tasks × 9 variations = 207 requests)."""
    requests = []
    for task in TASKS:
        for variation in VARIATION_CONFIGS:
            custom_id = f"{task['id']}__{variation['id']}"
            prompt = build_meta_prompt(task, variation)
            requests.append({
                "custom_id": custom_id,
                "params": {
                    "model": MODEL,
                    "max_tokens": 8192,
                    "messages": [{"role": "user", "content": prompt}],
                },
            })
    return requests


def cmd_submit():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic(api_key=api_key)
    requests = build_requests()

    print(f"Submitting {len(requests)} requests × {EXAMPLES_PER_CALL} examples "
          f"= ~{len(requests) * EXAMPLES_PER_CALL} total training examples")
    print(f"Model: {MODEL}  |  Estimated cost: ~$10–15 via Batch API")

    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id

    BATCH_FILE.write_text(batch_id)
    print(f"\n✓ Batch submitted: {batch_id}")
    print(f"  Saved to: {BATCH_FILE}")
    print(f"\nBatch processing typically takes 30–60 minutes.")
    print(f"Run: python generate.py status   (to check)")
    print(f"Run: python generate.py collect  (when complete)")


# ─────────────────────────────────────────────────────────────────────────────
# STATUS CHECK
# ─────────────────────────────────────────────────────────────────────────────
def cmd_status():
    if not BATCH_FILE.exists():
        print("No batch ID found. Run: python generate.py submit")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    batch_id = BATCH_FILE.read_text().strip()

    batch = client.messages.batches.retrieve(batch_id)
    counts = batch.request_counts

    print(f"Batch: {batch_id}")
    print(f"Status: {batch.processing_status}")
    print(f"  Processing: {counts.processing}")
    print(f"  Succeeded:  {counts.succeeded}")
    print(f"  Errored:    {counts.errored}")
    print(f"  Canceled:   {counts.canceled}")

    if batch.processing_status == "ended":
        print("\n✓ Batch complete. Run: python generate.py collect")


# ─────────────────────────────────────────────────────────────────────────────
# COLLECT + FORMAT
# ─────────────────────────────────────────────────────────────────────────────
def parse_example(raw_example: dict, task_id: str, system_prompt: str) -> dict | None:
    """Convert a raw generated example into a chat-format training record."""
    try:
        user_msg = raw_example["user_message"]
        turns = raw_example["turns"]

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": user_msg})

        for turn in turns:
            role = turn["role"]
            content = turn["content"]
            if role in ("assistant", "tool_result"):
                # Normalize tool_result → tool for standard chat format
                messages.append({
                    "role": "assistant" if role == "assistant" else "tool",
                    "content": content,
                })

        return {"task_id": task_id, "messages": messages}
    except (KeyError, TypeError):
        return None


def cmd_collect():
    if not BATCH_FILE.exists():
        print("No batch ID found. Run: python generate.py submit")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    batch_id = BATCH_FILE.read_text().strip()

    # Check status first
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        print(f"Batch not complete yet. Status: {batch.processing_status}")
        print("Check again with: python generate.py status")
        sys.exit(1)

    print("Collecting results...")
    all_examples: dict[str, list] = {}  # task_id → [examples]
    errors = 0
    parse_failures = 0

    for result in tqdm(client.messages.batches.results(batch_id)):
        custom_id = result.custom_id
        task_id = custom_id.split("__")[0]

        if result.result.type != "succeeded":
            errors += 1
            continue

        raw_text = result.result.message.content[0].text

        # Save raw response
        raw_file = RAW_DIR / f"{custom_id}.json"
        raw_file.write_text(raw_text)

        # Parse the JSON array Claude returned
        try:
            # Strip markdown code fences if present
            text = raw_text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            examples = json.loads(text)
        except json.JSONDecodeError:
            parse_failures += 1
            continue

        if task_id not in all_examples:
            all_examples[task_id] = []

        for ex in examples:
            parsed = parse_example(ex, task_id, OPENCLAW_SYSTEM)
            if parsed:
                all_examples[task_id].append(parsed)

    # Split train / val per task
    train_records = []
    val_records = []

    for task_id, examples in all_examples.items():
        random.shuffle(examples)
        val_cut = min(VAL_PER_TASK, len(examples))
        val_records.extend(examples[:val_cut])
        train_records.extend(examples[val_cut:])

    # Write JSONL
    with open(TRAIN_FILE, "w") as f:
        for rec in train_records:
            f.write(json.dumps(rec) + "\n")

    with open(VAL_FILE, "w") as f:
        for rec in val_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n{'─'*50}")
    print(f"✓ Collection complete")
    print(f"  API errors:      {errors}")
    print(f"  Parse failures:  {parse_failures}")
    print(f"  Train examples:  {len(train_records)}  →  {TRAIN_FILE}")
    print(f"  Val examples:    {len(val_records)}   →  {VAL_FILE}")
    print(f"\nNext step: python finetune.py")


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL  (submit → poll → collect)
# ─────────────────────────────────────────────────────────────────────────────
def cmd_run():
    cmd_submit()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    batch_id = BATCH_FILE.read_text().strip()

    print("\nPolling batch status every 2 minutes...")
    while True:
        time.sleep(120)
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  [{time.strftime('%H:%M')}] "
              f"processing={counts.processing}  "
              f"succeeded={counts.succeeded}  "
              f"errored={counts.errored}")
        if batch.processing_status == "ended":
            break

    print("\nBatch complete. Collecting...")
    cmd_collect()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PinchBench data generator")
    parser.add_argument(
        "command",
        choices=["submit", "status", "collect", "run"],
        help="submit: send batch | status: check progress | collect: save JSONL | run: all three",
    )
    args = parser.parse_args()

    commands = {
        "submit":  cmd_submit,
        "status":  cmd_status,
        "collect": cmd_collect,
        "run":     cmd_run,
    }
    commands[args.command]()
