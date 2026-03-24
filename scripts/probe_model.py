#!/usr/bin/env python3
"""
Probe the current Ollama model with benchmark-like prompts.

Sends prompts similar to failing tasks and shows the model's raw response
to understand WHY it's failing — does it use wrong tools, stop early,
hallucinate tools, or produce wrong output?

Usage:
  python scripts/probe_model.py                           # probe all failing tasks
  python scripts/probe_model.py --task task_01_calendar   # probe specific task
  python scripts/probe_model.py --model qwen3.5:9b        # probe base model for comparison
"""

import argparse
import json
import sys
import httpx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import load_config

cfg = load_config()

# Simplified prompts that mimic what PinchBench sends
PROBE_PROMPTS = {
    "task_01_calendar": (
        "Schedule a meeting called 'Project Sync' for next Tuesday at 3:00 PM "
        "with attendee john@example.com and include a note about Q1 roadmap "
        "discussion. Save it as an .ics file."
    ),
    "task_08_memory": (
        "Read the file notes.md in your workspace and tell me: "
        "what is the beta release deadline? Write your answer to answer.txt."
    ),
    "task_10_workflow": (
        "Read config.json, extract the API endpoint URL, write a Python script "
        "api_client.py that calls that endpoint with error handling, "
        "then create NOTES.md documenting the process."
    ),
    "task_12_skill_search": (
        "Update the config files config/settings.json and config/database.yml "
        "for production: replace localhost with prod-db.example.com, rename "
        "databases to myapp_prod, change log level to warn, update API endpoint "
        "to https://api.example.com."
    ),
    "task_15_daily_summary": (
        "Read all files in the research/ directory and synthesize them into "
        "a 500-800 word executive briefing saved as daily_briefing.md."
    ),
    "task_20_eli5_pdf": (
        "Read GPT4.pdf and write an ELI5 summary (200-400 words) that a "
        "5-year-old could understand. Save it to eli5_summary.txt. "
        "Use simple words and everyday analogies, no technical jargon."
    ),
    "task_21_openclaw_comprehension": (
        "Read openclaw_report.pdf and answer these 8 questions, one per line "
        "in answer.txt: 1) Total community skills before filtering, "
        "2) Skills after filtering, 3) Largest category with count, "
        "4) Second largest category with count, 5) Filename defining a skill, "
        "6) API type exposed by gateway, 7) Date data was collected, "
        "8) Number of new benchmark tasks proposed."
    ),
}


def call_ollama(model, prompt, system_prompt=None):
    """Call Ollama chat API and return the response."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        resp = httpx.post(
            "http://127.0.0.1:11434/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "think": False,  # disable thinking mode
                "options": {"num_predict": 4096, "temperature": 0.7},
            },
            timeout=120,
        )
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        thinking = data.get("message", {}).get("thinking", "")
        if thinking:
            print(f"  [THINKING MODE DETECTED: {len(thinking)} chars of thinking]")
            print(f"  Thinking preview: {thinking[:200]}...")
        if not content:
            # Dump raw response for debugging
            print(f"  [RAW RESPONSE KEYS: {list(data.keys())}]")
            msg = data.get("message", {})
            print(f"  [MESSAGE KEYS: {list(msg.keys())}]")
            print(f"  [MESSAGE ROLE: {msg.get('role')}]")
            print(f"  [CONTENT LENGTH: {len(msg.get('content', ''))}]")
            return "(no response)"
        return content
    except Exception as e:
        return f"ERROR: {e}"


def analyze_response(response):
    """Analyze what the model did in its response."""
    import re
    analysis = {
        "length": len(response),
        "has_tool_calls": "<tool_call>" in response,
        "tool_names": [],
        "has_thinking": "<think>" in response.lower() or "thinking" in response[:200].lower(),
        "stops_early": len(response) < 200 and "<tool_call>" not in response,
    }

    for match in re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response, re.DOTALL):
        try:
            obj = json.loads(match)
            analysis["tool_names"].append(obj.get("name", "?"))
        except json.JSONDecodeError:
            analysis["tool_names"].append("PARSE_ERROR")

    # Check for hallucinated tools
    valid_tools = {
        "read_file", "write_file", "create_directory", "list_files",
        "run_bash", "run_python", "web_search", "fetch_url",
        "create_calendar_event", "draft_email", "search_emails", "read_email",
        "generate_image", "read_memory", "write_memory",
        "search_skills", "install_skill",
    }
    analysis["hallucinated_tools"] = [t for t in analysis["tool_names"] if t not in valid_tools]

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Probe model with benchmark-like prompts")
    parser.add_argument("--task", type=str, default=None, help="Specific task to probe")
    parser.add_argument("--model", type=str, default=None, help="Ollama model name")
    parser.add_argument("--no-system", action="store_true", help="Skip system prompt")
    args = parser.parse_args()

    model = args.model or cfg.ollama_model_name + ":latest"

    # System prompt from our training data
    from utils.prompts import OPENCLAW_SYSTEM
    system_prompt = None if args.no_system else OPENCLAW_SYSTEM

    tasks = {args.task: PROBE_PROMPTS[args.task]} if args.task else PROBE_PROMPTS

    print(f"\n{'='*70}")
    print(f"  MODEL PROBE — {model}")
    print(f"  System prompt: {'OFF' if args.no_system else 'ON'}")
    print(f"  Tasks: {len(tasks)}")
    print(f"{'='*70}")

    for task_id, prompt in tasks.items():
        print(f"\n{'─'*70}")
        print(f"  {task_id}")
        print(f"{'─'*70}")
        print(f"  Prompt: {prompt[:150]}...")

        response = call_ollama(model, prompt, system_prompt)
        analysis = analyze_response(response)

        print(f"\n  Response ({analysis['length']} chars):")
        # Show first 500 chars
        for line in response[:500].splitlines():
            print(f"    {line}")
        if len(response) > 500:
            print(f"    ... ({len(response) - 500} more chars)")

        print(f"\n  Analysis:")
        print(f"    Tool calls: {analysis['tool_names'] or 'NONE'}")
        if analysis["hallucinated_tools"]:
            print(f"    ⚠ HALLUCINATED TOOLS: {analysis['hallucinated_tools']}")
        if analysis["has_thinking"]:
            print(f"    ⚠ Thinking mode leaked")
        if analysis["stops_early"]:
            print(f"    ⚠ Very short response — may have stopped early")
        if not analysis["has_tool_calls"]:
            print(f"    ⚠ NO TOOL CALLS — model responded with text only")

    print(f"\n{'='*70}")
    print(f"  PROBE COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
