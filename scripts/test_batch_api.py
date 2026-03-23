#!/usr/bin/env python3
"""Quick test: is the batch API working?"""
import os
import time
import anthropic

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

print("Submitting test batch...")
batch = client.messages.batches.create(requests=[{
    "custom_id": "test_hello",
    "params": {
        "model": "claude-sonnet-4-5",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
    },
}])
print(f"Batch: {batch.id}")

while True:
    status = client.messages.batches.retrieve(batch.id)
    print(f"  Status: {status.processing_status} (ok={status.request_counts.succeeded}, err={status.request_counts.errored})")
    if status.processing_status == "ended":
        break
    time.sleep(10)

for result in client.messages.batches.results(batch.id):
    if result.result.type == "succeeded":
        print(f"  Response: {result.result.message.content[0].text}")
    else:
        print(f"  Error: {result.result.error}")

print("API is working!" if status.request_counts.succeeded > 0 else "API has issues.")
