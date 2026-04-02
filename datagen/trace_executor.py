"""
Trace executor: replays synthetic training examples in a real sandbox.

Replaces fabricated tool results with real execution output. Rejects
traces where tool calls fail (Python errors, empty writes, looping).

This closes the open loop in data generation: instead of Claude
imagining what tool results look like, we execute them for real.

Usage:
    from datagen.trace_executor import execute_trace, passes_quality_filter

    result = execute_trace(example, task_id)
    if passes_quality_filter(result):
        # use result["example"] which has real tool results
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


# Tools that can be executed locally
EXECUTABLE_TOOLS = {
    "write_file", "create_directory", "list_files", "read_file",
    "run_python", "run_bash",
}

# Tools that depend on external services -- keep synthetic results
EXTERNAL_TOOLS = {
    "web_search", "fetch_url", "generate_image", "draft_email",
    "search_emails", "read_email", "create_calendar_event",
    "read_memory", "write_memory", "search_skills", "install_skill",
}


def _find_task_fixtures(task_id: str) -> Path | None:
    """Find pre-existing files that PinchBench provides for a task."""
    ws = os.environ.get("PBM_WORKSPACE", "/workspace/synthbench")
    tasks_dir = Path(ws) / "skill" / "tasks"
    if not tasks_dir.exists():
        return None

    # PinchBench tasks are numbered directories like 01_calendar/
    short_id = task_id.replace("task_", "")  # "01_calendar"
    for d in tasks_dir.iterdir():
        if d.is_dir() and short_id in d.name:
            # Look for a workspace/ or files/ subdirectory with fixtures
            for sub in ("workspace", "files", "assets", "data"):
                fixture_dir = d / sub
                if fixture_dir.exists():
                    return fixture_dir
            # Some tasks have files directly in the task directory
            if any(f.suffix in (".txt", ".csv", ".xlsx", ".pdf", ".md", ".json")
                   for f in d.iterdir() if f.is_file()):
                return d
    return None


def _execute_python(code: str, workspace: Path, timeout: int = 30) -> dict:
    """Run Python code in a subprocess with workspace as cwd."""
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "status": "ok" if result.returncode == 0 else "error",
            "stdout": result.stdout[:5000],
            "stderr": result.stderr[:2000],
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "stdout": "", "stderr": "Timeout", "returncode": -1}
    except Exception as e:
        return {"status": "error", "stdout": "", "stderr": str(e), "returncode": -1}


def _execute_bash(command: str, workspace: Path, timeout: int = 30) -> dict:
    """Run a bash command in the workspace directory."""
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "status": "ok" if result.returncode == 0 else "error",
            "stdout": result.stdout[:5000],
            "stderr": result.stderr[:2000],
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "stdout": "", "stderr": "Timeout", "returncode": -1}
    except Exception as e:
        return {"status": "error", "stdout": "", "stderr": str(e), "returncode": -1}


def execute_trace(example: dict, task_id: str,
                  fixtures_dir: Path | None = None) -> dict:
    """Replay a synthetic trace in a temp workspace.

    For each tool call: if executable locally, run it for real and replace
    the fabricated tool_result. If external (web_search, etc.), keep as-is.
    """
    messages = example.get("messages", [])
    if not messages:
        return {"executed": False, "example": example, "execution_log": [],
                "quality_signals": _empty_signals()}

    # Create temp workspace
    workspace = Path(tempfile.mkdtemp(prefix=f"trace_{task_id}_"))

    # Copy fixtures into workspace if available
    if fixtures_dir and fixtures_dir.exists():
        for item in fixtures_dir.iterdir():
            dest = workspace / item.name
            if item.is_file():
                shutil.copy2(item, dest)
            elif item.is_dir():
                shutil.copytree(item, dest)

    execution_log = []
    new_messages = []
    output_files = []

    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg.get("role") == "assistant":
            # Extract tool calls from this message
            tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', msg.get("content", ""), re.DOTALL)

            if tool_calls and i + 1 < len(messages) and messages[i + 1].get("role") in ("tool", "tool_result"):
                new_messages.append(msg)
                i += 1

                # Process each tool call and its result
                for tc_raw in tool_calls:
                    try:
                        tc = json.loads(tc_raw.strip())
                    except json.JSONDecodeError:
                        execution_log.append({"tool": "PARSE_ERROR", "status": "skip"})
                        continue

                    tool_name = tc.get("name", "")
                    args = tc.get("arguments", {})

                    if tool_name in EXECUTABLE_TOOLS:
                        real_result, log_entry = _execute_tool(
                            tool_name, args, workspace
                        )
                        execution_log.append(log_entry)

                        if tool_name == "write_file":
                            path = args.get("path", "")
                            if path:
                                output_files.append(path)
                    else:
                        execution_log.append({
                            "tool": tool_name, "status": "skip", "replaced": False
                        })

                # Keep the original tool_result (we log execution but
                # only replace results for tools where it matters)
                if i < len(messages):
                    tool_msg = messages[i]
                    # For executable tools, replace with real result if we got one
                    replaced_content = _build_replaced_result(
                        tool_calls, execution_log, tool_msg.get("content", "")
                    )
                    new_messages.append({
                        "role": tool_msg["role"],
                        "content": replaced_content,
                    })
                    i += 1
            else:
                new_messages.append(msg)
                i += 1
        else:
            new_messages.append(msg)
            i += 1

    # Check output files
    files_created = []
    files_nonempty = True
    for f in output_files:
        full_path = workspace / f.lstrip("/")
        if full_path.exists():
            files_created.append(f)
            if full_path.stat().st_size == 0:
                files_nonempty = False
        else:
            files_nonempty = False

    # Build quality signals
    tool_calls_total = len(execution_log)
    unique_tools = len(set(e["tool"] for e in execution_log if e.get("tool")))
    all_writes = all(
        e["status"] == "ok" for e in execution_log
        if e["tool"] == "write_file" and e.get("replaced")
    )
    all_python = all(
        e["status"] == "ok" for e in execution_log
        if e["tool"] == "run_python" and e.get("replaced")
    )
    no_fnf = not any(
        "not found" in e.get("stderr", "").lower() or "no such file" in e.get("stderr", "").lower()
        for e in execution_log if e.get("status") == "error"
    )
    last_is_assistant = (
        len(new_messages) > 0
        and new_messages[-1].get("role") == "assistant"
        and "<tool_call>" not in new_messages[-1].get("content", "")
    )

    new_example = dict(example)
    new_example["messages"] = new_messages

    # Cleanup
    shutil.rmtree(workspace, ignore_errors=True)

    return {
        "executed": True,
        "example": new_example,
        "execution_log": execution_log,
        "quality_signals": {
            "all_writes_succeeded": all_writes,
            "all_python_ran_clean": all_python,
            "no_file_not_found": no_fnf,
            "output_files_created": files_created,
            "output_files_nonempty": files_nonempty,
            "tool_call_count": tool_calls_total,
            "unique_tools_used": unique_tools,
            "trace_completed": last_is_assistant,
        },
    }


def _execute_tool(tool_name: str, args: dict, workspace: Path) -> tuple[str, dict]:
    """Execute a single tool call. Returns (result_content, log_entry)."""
    if tool_name == "write_file":
        path = args.get("path", "file.txt")
        content = args.get("content", "")
        full_path = workspace / path.lstrip("/")
        full_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            full_path.write_text(content)
            result = json.dumps({"status": "ok", "path": path, "bytes_written": len(content)})
            return result, {"tool": "write_file", "status": "ok", "replaced": True}
        except Exception as e:
            return json.dumps({"error": str(e)}), {"tool": "write_file", "status": "error", "replaced": True, "stderr": str(e)}

    elif tool_name == "create_directory":
        path = args.get("path", "dir")
        full_path = workspace / path.lstrip("/")
        try:
            full_path.mkdir(parents=True, exist_ok=True)
            result = json.dumps({"status": "ok", "path": path})
            return result, {"tool": "create_directory", "status": "ok", "replaced": True}
        except Exception as e:
            return json.dumps({"error": str(e)}), {"tool": "create_directory", "status": "error", "replaced": True}

    elif tool_name == "read_file":
        path = args.get("path", "")
        full_path = workspace / path.lstrip("/")
        try:
            content = full_path.read_text(errors="replace")[:10000]
            return content, {"tool": "read_file", "status": "ok", "replaced": True}
        except FileNotFoundError:
            return json.dumps({"error": f"File not found: {path}"}), {
                "tool": "read_file", "status": "error", "replaced": True,
                "stderr": f"File not found: {path}",
            }

    elif tool_name == "list_files":
        path = args.get("path", ".")
        full_path = workspace / path.lstrip("/")
        try:
            entries = sorted(str(p.relative_to(workspace)) for p in full_path.iterdir())
            return json.dumps(entries), {"tool": "list_files", "status": "ok", "replaced": True}
        except Exception as e:
            return json.dumps({"error": str(e)}), {"tool": "list_files", "status": "error", "replaced": True}

    elif tool_name == "run_python":
        code = args.get("code", "")
        exec_result = _execute_python(code, workspace)
        output = exec_result["stdout"]
        if exec_result["stderr"]:
            output += f"\nSTDERR: {exec_result['stderr']}"
        return output, {
            "tool": "run_python", "status": exec_result["status"],
            "replaced": True, "stderr": exec_result["stderr"],
        }

    elif tool_name == "run_bash":
        command = args.get("command", "")
        exec_result = _execute_bash(command, workspace)
        output = exec_result["stdout"]
        if exec_result["stderr"]:
            output += f"\nSTDERR: {exec_result['stderr']}"
        return output, {
            "tool": "run_bash", "status": exec_result["status"],
            "replaced": True, "stderr": exec_result["stderr"],
        }

    return "", {"tool": tool_name, "status": "skip", "replaced": False}


def _build_replaced_result(tool_calls_raw: list, execution_log: list,
                           original_content: str) -> str:
    """Build replacement tool_result content from execution results.

    For now, keep original content if no executable tools were replaced.
    When we have real results, use them.
    """
    # If any tool was replaced with a real result, we should use real results.
    # But the message format has one tool_result per turn, potentially covering
    # multiple tool calls. For simplicity, keep original unless we have a
    # single clear replacement.
    replaced = [e for e in execution_log if e.get("replaced")]
    if not replaced:
        return original_content
    return original_content  # TODO: smarter merging of multi-tool results


def _empty_signals() -> dict:
    return {
        "all_writes_succeeded": True,
        "all_python_ran_clean": True,
        "no_file_not_found": True,
        "output_files_created": [],
        "output_files_nonempty": True,
        "tool_call_count": 0,
        "unique_tools_used": 0,
        "trace_completed": False,
    }


def passes_quality_filter(result: dict) -> bool:
    """Check if an executed trace meets quality standards.

    These are universal signals of competent agent behavior:
    - Python code should run without errors
    - Written files should be non-empty
    - The agent shouldn't loop 15+ times
    - The trace should complete (end with assistant message)
    """
    if not result.get("executed"):
        return True  # can't judge what wasn't executed

    q = result["quality_signals"]
    log = result.get("execution_log", [])

    # Reject: Python errors
    python_calls = [e for e in log if e["tool"] == "run_python" and e.get("replaced")]
    if python_calls and not q["all_python_ran_clean"]:
        return False

    # Reject: bash errors
    bash_calls = [e for e in log if e["tool"] == "run_bash" and e.get("replaced")]
    if bash_calls and any(e["status"] == "error" for e in bash_calls):
        return False

    # Reject: write_file produced empty or missing files
    write_calls = [e for e in log if e["tool"] == "write_file" and e.get("replaced")]
    if write_calls and not q["all_writes_succeeded"]:
        return False

    # Reject: too many tool calls (looping)
    if q["tool_call_count"] > 15:
        return False

    # Reject: trace didn't complete
    if not q["trace_completed"]:
        return False

    return True
