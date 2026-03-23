#!/usr/bin/env python3
"""Quick test: is the batch API working?"""
import os
import time
import anthropic

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Test 1: Real-time API
print("Test 1: Real-time Messages API (claude-sonnet-4-6)...")
try:
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=50,
        messages=[{"role": "user", "content": "Say hello in one word."}],
    )
    print(f"  OK: {resp.content[0].text}")
except Exception as e:
    print(f"  FAILED: {e}")

# Test 2: Real-time with sonnet 4-5
print("\nTest 2: Real-time Messages API (claude-sonnet-4-5)...")
try:
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=50,
        messages=[{"role": "user", "content": "Say hello in one word."}],
    )
    print(f"  OK: {resp.content[0].text}")
except Exception as e:
    print(f"  FAILED: {e}")

# Test 3: Batch API
print("\nTest 3: Batch API (claude-sonnet-4-6)...")
batch = client.messages.batches.create(requests=[{
    "custom_id": "test_hello",
    "params": {
        "model": "claude-sonnet-4-6",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
    },
}])
print(f"  Batch: {batch.id}")

while True:
    status = client.messages.batches.retrieve(batch.id)
    print(f"  Status: {status.processing_status} (ok={status.request_counts.succeeded}, err={status.request_counts.errored})")
    if status.processing_status == "ended":
        break
    time.sleep(10)

for result in client.messages.batches.results(batch.id):
    if result.result.type == "succeeded":
        print(f"  OK: {result.result.message.content[0].text}")
    else:
        print(f"  FAILED: {result.result.error}")

print("\nDone.")
