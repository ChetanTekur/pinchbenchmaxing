#!/usr/bin/env bash
# Fix Qwen3 fine-tuned GGUF Modelfile with correct chat template and tool support
#
# WHY THIS SCRIPT EXISTS:
#   When you convert a fine-tuned model to GGUF and register it with ollama, the
#   default Modelfile template does NOT include the <tools>...</tools> block needed
#   for function/tool calling. Without this, OpenClaw's tool calls are silently
#   ignored and the model never invokes any tools — causing near-zero benchmark scores.
#
#   This script regenerates the Modelfile by copying the full chat template from the
#   base qwen3:8b model (which ollama ships with proper tool-call support), then
#   pointing FROM at the fine-tuned GGUF file. Re-creating the model with this fixed
#   Modelfile restores full tool-calling capability for the fine-tuned weights.

GGUF_PATH="/workspace/synthbench/qwen35-9b-clawd_gguf/qwen35-9b-clawd.Q4_K_M.gguf"
MODEL_NAME="qwen35-9b-gguf-claw"
MODELFILE="/tmp/Modelfile-clawd"

cat > "$MODELFILE" << 'EOF'
FROM /workspace/synthbench/qwen35-9b-clawd_gguf/qwen35-9b-clawd.Q4_K_M.gguf
TEMPLATE """
{{- $lastUserIdx := -1 -}}
{{- range $idx, $msg := .Messages -}}
{{- if eq $msg.Role "user" }}{{ $lastUserIdx = $idx }}{{ end -}}
{{- end }}
{{- if or .System .Tools }}<|im_start|>system
{{ if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end -}}
<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}
{{- if and $.IsThinkSet (eq $i $lastUserIdx) }}
   {{- if $.Think -}}
      {{- " "}}/think
   {{- else -}}
      {{- " "}}/no_think
   {{- end -}}
{{- end }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if (and $.IsThinkSet (and .Thinking (or $last (gt $i $lastUserIdx)))) -}}
<think>{{ .Thinking }}</think>
{{ end -}}
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ if and $.IsThinkSet (not $.Think) -}}
<think>

</think>

{{ end -}}
{{ end }}
{{- end }}"""
PARAMETER top_k 20
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1
PARAMETER stop <|im_start|>
PARAMETER stop <|im_end|>
PARAMETER temperature 0.6
SYSTEM """You are Clawd, an autonomous AI agent powered by OpenClaw. You help users accomplish real-world tasks by using tools. Be direct and competent — start with action, not explanation. Get things done."""
EOF

echo "Removing old model..."
ollama rm "$MODEL_NAME" 2>/dev/null

echo "Creating fixed model..."
ollama create "$MODEL_NAME" -f "$MODELFILE"

echo "Testing..."
ollama run "$MODEL_NAME" "Say hello and stop."
