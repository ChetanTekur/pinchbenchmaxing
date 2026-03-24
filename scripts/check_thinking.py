#!/usr/bin/env python3
"""Check if enable_thinking=False works with the Qwen3.5 tokenizer."""
from transformers import AutoTokenizer

print("Loading Qwen3.5-9B tokenizer...")
t = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")

msgs = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hi there"},
]

# Test 1: with enable_thinking=False
print("\nTest 1: enable_thinking=False")
try:
    out = t.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    print("  WORKS")
    print(f"  Output: {repr(out[:300])}")
    has_think = "<think>" in out
    print(f"  Contains <think>: {has_think}")
except TypeError as e:
    print(f"  TypeError: {e}")
    print("  ⚠ FALLING BACK — training data may include thinking tokens!")

# Test 2: without enable_thinking (default)
print("\nTest 2: default (no enable_thinking)")
try:
    out = t.apply_chat_template(msgs, tokenize=False)
    print("  WORKS")
    print(f"  Output: {repr(out[:300])}")
    has_think = "<think>" in out
    print(f"  Contains <think>: {has_think}")
except Exception as e:
    print(f"  Error: {e}")

# Test 3: with enable_thinking=True
print("\nTest 3: enable_thinking=True")
try:
    out = t.apply_chat_template(msgs, tokenize=False, enable_thinking=True)
    print("  WORKS")
    print(f"  Output: {repr(out[:300])}")
    has_think = "<think>" in out
    print(f"  Contains <think>: {has_think}")
except TypeError as e:
    print(f"  TypeError: {e}")
