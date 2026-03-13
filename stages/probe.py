#!/usr/bin/env python3
"""
Probe a fine-tuned model from merged weights (no GGUF needed).

Usage:
  python stages/probe.py                                    # interactive
  python stages/probe.py --prompt "Search for AI news"
  python stages/probe.py --config /path/to/config.yaml
  python stages/probe.py --merged /custom/path/to/merged
"""

import argparse
import torch
from pathlib import Path

from utils.config import load_config

SYSTEM_PROMPT = (
    "You are Clawd, an autonomous AI agent powered by OpenClaw. "
    "You help users accomplish real-world tasks by using tools. "
    "Be direct and competent — start with action, not explanation."
)


def load_model(model_path: Path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("Loaded.\n")
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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
    return tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default=None)
    parser.add_argument("--merged",     default=None, help="Override merged model path")
    parser.add_argument("--prompt",     default=None, help="Single prompt (non-interactive)")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    cfg         = load_config(args.config)
    merged_path = Path(args.merged) if args.merged else cfg.merged_dir

    if not merged_path.exists():
        raise FileNotFoundError(f"Merged model not found at {merged_path}")

    model, tokenizer = load_model(merged_path)

    if args.prompt:
        print("Clawd:", generate(model, tokenizer, args.prompt, args.max_tokens))
    else:
        print("Interactive mode — Ctrl+C or type 'exit' to quit.\n")
        while True:
            try:
                prompt = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not prompt or prompt.lower() in ("exit", "quit"):
                break
            print("\nClawd:", generate(model, tokenizer, prompt, args.max_tokens), "\n")


if __name__ == "__main__":
    main()
