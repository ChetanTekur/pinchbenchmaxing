# Claude Code Rules for Synthdata / PinchBench

## Always commit and push BEFORE giving pod instructions
Never give RunPod shell commands unless the relevant code changes are already committed and pushed to GitHub. The pod pulls from GitHub — instructions without a push are useless and waste money.

## Always update BOTH scripts AND Dockerfile simultaneously
Any fix that touches a script (`scripts/`, `stages/`) must also be reflected in the Dockerfile if it affects installed packages, system deps, or PATH. Never fix one without checking the other. Same applies to setup_pod.sh.

## Fix root cause, not symptoms
When something is broken on the pod, fix the underlying scripts/Dockerfile so the next pod starts correctly — not just the immediate workaround.

## Project context
- Benchmark: PinchBench (pinchbench.com), 23 tasks, target 80%
- Model: Qwen3.5-9B fine-tuned via Unsloth LoRA → GGUF → Ollama
- Framework: OpenClaw (gateway on port 18789), installed via `npm install -g openclaw@latest` (requires Node 22)
- Workspace: `/workspace/synthbench` on RunPod network volume (CA-2)
- Config source of truth: `config.yaml` + `utils/config.py` — never hardcode paths in scripts
- GGUF location after convert: `{workspace}/models/{model_name}_gguf/{model_name}.{QUANT}.gguf`
