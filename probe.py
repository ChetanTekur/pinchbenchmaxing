#!/usr/bin/env python3
"""
Probe a fine-tuned model directly from merged weights (no GGUF needed).

Usage:
  python probe.py                                      # interactive mode
  python probe.py --model /workspace/synthbench/qwen35-9b-clawd_merged
  python probe.py --prompt "Search the web for AI news"
  python probe.py --prompt "What is 2+2?" --max-tokens 128

The model is loaded in bfloat16 and runs on GPU automatically.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "/workspace/synthbench/qwen35-9b-clawd_merged"
SYSTEM_PROMPT = (
    "You are Clawd, an autonomous AI agent powered by OpenClaw. "
    "You help users accomplish real-world tasks by using tools. "
    "Be direct and competent — start with action, not explanation."
)


def load_model(model_path: str):
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Loaded. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'auto'}\n")
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.1,
        )
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def interactive(model, tokenizer, max_new_tokens: int):
    print("Interactive mode — type your prompt and press Enter. Ctrl+C or 'exit' to quit.\n")
    while True:
        try:
            prompt = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not prompt or prompt.lower() in ("exit", "quit"):
            break
        print("\nClawd:", generate(model, tokenizer, prompt, max_new_tokens), "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to merged model directory")
    parser.add_argument("--prompt", default=None, help="Single prompt (non-interactive)")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    if args.prompt:
        response = generate(model, tokenizer, args.prompt, args.max_tokens)
        print("Clawd:", response)
    else:
        interactive(model, tokenizer, args.max_tokens)


if __name__ == "__main__":
    main()
