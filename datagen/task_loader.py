#!/usr/bin/env python3
"""
Load PinchBench task definitions from the actual benchmark .md files.

This is the SINGLE SOURCE OF TRUTH for what each task expects.
No hardcoded task definitions — everything comes from the benchmark repo.

Usage:
    from datagen.task_loader import load_tasks, load_task
    tasks = load_tasks()           # all 23 tasks
    task = load_task("task_12")    # single task

    # Each task dict has:
    # - id, name, category, grading_type
    # - raw_content (full .md file)
    # - prompt_section (extracted prompt/objective)
    # - grading_section (extracted grading criteria)
    # - expected_files (files the agent must create)
    # - expected_values (specific values that must appear)
"""

import os
import re
from pathlib import Path
from utils.config import load_config


def _find_tasks_dir() -> Path:
    """Find the PinchBench tasks directory."""
    cfg = load_config()
    # Primary: workspace/skill/tasks/
    skill_dir = cfg.workspace / "skill" / "tasks"
    if skill_dir.exists():
        return skill_dir

    # Fallback: check PBM_WORKSPACE env
    ws = os.environ.get("PBM_WORKSPACE", "/workspace/synthbench")
    fallback = Path(ws) / "skill" / "tasks"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"PinchBench tasks not found at {skill_dir} or {fallback}. "
        f"Clone pinchbench/skill to the workspace first."
    )


def _parse_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter from a .md file."""
    if not content.startswith("---"):
        return {}
    end = content.find("---", 3)
    if end == -1:
        return {}
    frontmatter = content[3:end].strip()
    result = {}
    for line in frontmatter.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            result[key.strip()] = val.strip()
    return result


def _extract_expected_files(content: str) -> list[str]:
    """Extract expected output filenames from task description."""
    files = []
    # Look for explicit file mentions in common patterns
    patterns = [
        r'(?:Output|File|Save|Write|Create).*?[`"]([a-zA-Z0-9_./\-]+\.[a-z]{1,5})[`"]',
        r'(?:save|write|create).*?(?:as|to|called)\s+[`"]?([a-zA-Z0-9_./\-]+\.[a-z]{1,5})[`"]?',
        r'\*\*(?:Output|File)\*\*:\s*`([^`]+)`',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            fname = match.group(1)
            if fname not in files and not fname.startswith("http"):
                files.append(fname)
    return files


def _extract_expected_values(content: str) -> list[str]:
    """Extract specific values that must appear in the output."""
    values = []
    # Look for explicit expected values in tables or lists
    # Pattern: | number | expected_answer |
    for match in re.finditer(r'\|\s*\d+\s*\|\s*(.+?)\s*\|', content):
        val = match.group(1).strip()
        if val and val != "Expected Answer" and not val.startswith("---"):
            values.append(val)
    # Pattern: specific numbers/dates called out
    for match in re.finditer(r'(?:expected|must|should|correct).*?[`"]([^`"]+)[`"]', content, re.IGNORECASE):
        val = match.group(1).strip()
        if val and len(val) < 100:
            values.append(val)
    return values


def _extract_sections(content: str) -> dict:
    """Extract key sections from the markdown content."""
    sections = {}
    current_heading = None
    current_content = []

    for line in content.splitlines():
        heading_match = re.match(r'^#{1,3}\s+(.+)', line)
        if heading_match:
            if current_heading:
                sections[current_heading.lower()] = "\n".join(current_content).strip()
            current_heading = heading_match.group(1)
            current_content = []
        else:
            current_content.append(line)

    if current_heading:
        sections[current_heading.lower()] = "\n".join(current_content).strip()

    return sections


def load_task(task_id: str) -> dict:
    """Load a single task definition from its .md file."""
    tasks_dir = _find_tasks_dir()

    # Find the file — handle naming variations
    candidates = list(tasks_dir.glob(f"{task_id}*.md"))
    if not candidates:
        # Try partial match
        candidates = [f for f in tasks_dir.glob("task_*.md")
                      if task_id in f.stem]
    if not candidates:
        raise FileNotFoundError(f"No task file found for {task_id} in {tasks_dir}")

    task_file = candidates[0]
    content = task_file.read_text()
    frontmatter = _parse_frontmatter(content)
    sections = _extract_sections(content)

    # Remove frontmatter from content for section parsing
    body = content
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            body = content[end + 3:].strip()

    return {
        "id": frontmatter.get("id", task_id),
        "name": frontmatter.get("name", ""),
        "category": frontmatter.get("category", ""),
        "grading_type": frontmatter.get("grading_type", ""),
        "timeout": int(frontmatter.get("timeout", "180").replace("s", "")),
        "raw_content": body,
        "sections": sections,
        "expected_files": _extract_expected_files(body),
        "expected_values": _extract_expected_values(body),
        "source_file": str(task_file),
    }


def load_tasks() -> dict[str, dict]:
    """Load all task definitions from the benchmark tasks directory."""
    tasks_dir = _find_tasks_dir()
    tasks = {}
    for task_file in sorted(tasks_dir.glob("task_*.md")):
        task_id = task_file.stem
        # Normalize: task_11_clawdhub → task_11_config_update (our internal ID)
        # We use PinchBench's canonical ID from frontmatter
        try:
            task = load_task(task_id)
            canonical_id = task["id"]
            tasks[canonical_id] = task
        except Exception as e:
            print(f"  Warning: failed to load {task_file.name}: {e}")
    return tasks


def print_task_summary(task: dict):
    """Print a human-readable summary of a task definition."""
    print(f"\n{'='*60}")
    print(f"  {task['id']} — {task['name']}")
    print(f"  Category: {task['category']} | Grading: {task['grading_type']}")
    print(f"{'='*60}")
    if task["expected_files"]:
        print(f"  Expected files: {task['expected_files']}")
    if task["expected_values"]:
        print(f"  Expected values: {task['expected_values'][:5]}")
    print(f"  Content preview: {task['raw_content'][:200]}...")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        task = load_task(sys.argv[1])
        print_task_summary(task)
    else:
        tasks = load_tasks()
        print(f"\nLoaded {len(tasks)} tasks:\n")
        for tid, task in tasks.items():
            files = task["expected_files"]
            print(f"  {tid:<35} {task['name']:<40} files={files}")
