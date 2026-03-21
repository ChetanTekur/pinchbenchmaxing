# Claude Code Rules for Synthdata / PinchBench

## Give accurate pod instructions — verify before suggesting
Before giving the user any shell commands to run on the pod:
1. **Verify the command is correct** — check the actual script flags, paths, and arguments by reading the code. Do not guess.
2. **Commit and push first** — never give pod commands unless the relevant code changes are already pushed to GitHub.
3. **Don't redundantly pass defaults** — if a script already has a sensible default, don't add the flag.
4. **When unsure about external tools** — use WebFetch to check the actual source code rather than guessing CLI flags.

The user's time on the pod costs real money. Every wrong instruction wastes it.

## Always update BOTH scripts AND Dockerfile simultaneously
Any fix that touches a script (`scripts/`, `stages/`) must also be reflected in the Dockerfile if it affects installed packages, system deps, or PATH. Never fix one without checking the other. Same applies to setup_pod.sh.

## Fix root cause, not symptoms
When something is broken on the pod, fix the underlying scripts/Dockerfile so the next pod starts correctly — not just the immediate workaround.

## Project context
- Benchmark: PinchBench (pinchbench.com), 23 tasks, target 85%
- Model: Qwen3.5-9B fine-tuned via Unsloth LoRA → GGUF → Ollama
- Framework: OpenClaw (gateway on port 18789), installed via `npm install -g openclaw@latest` (requires Node 22)
- Workspace: `/workspace/synthbench` on RunPod network volume (CA-2)
- Config source of truth: `config.yaml` + `utils/config.py` — never hardcode paths in scripts
- GGUF location after convert: `{workspace}/models/{model_name}_gguf/{model_name}.{QUANT}.gguf`
- PinchBench skill repo: cloned to `/workspace/synthbench/skill/` from github.com/pinchbench/skill
- PinchBench requires ollama patch on `skill/scripts/lib_agent.py` (validate_openrouter_model must skip ollama/ models)
- Judge model: openrouter/anthropic/claude-opus-4.5 (PinchBench default, no flag needed)
- Always upload benchmark results to leaderboard (no --no-upload)
