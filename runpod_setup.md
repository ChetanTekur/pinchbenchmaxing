# RunPod Deployment Guide

This document describes how to build, push, and run the synthdata Docker image on RunPod for fine-tuning Qwen3-8B and running PinchBench benchmarks.

---

## 1. Build and Push the Docker Image

### Prerequisites (local machine)
- Docker Desktop installed and running
- A Docker Hub account (or another registry)

### Build

```bash
cd /Users/chetantekur/Documents/Claude-dir/synthdata

docker build -t <your-dockerhub-username>/synthdata:latest .
```

Replace `<your-dockerhub-username>` with your actual Docker Hub username (e.g. `chetant/synthdata:latest`).

### Push to Docker Hub

```bash
docker login
docker push <your-dockerhub-username>/synthdata:latest
```

Tag with a version string when you want a stable snapshot:

```bash
docker tag <your-dockerhub-username>/synthdata:latest <your-dockerhub-username>/synthdata:v1
docker push <your-dockerhub-username>/synthdata:v1
```

---

## 2. Deploy on RunPod

### Step 1 — Create a Network Volume

1. RunPod console → **Storage** → **New Network Volume**
2. Region: **CA-2** (required — existing data lives here)
3. Size: **50 GB minimum** (100 GB recommended for checkpoints + GGUF files)
4. Name: `synthbench` (or any memorable name)

The volume will be mounted at `/workspace` inside the pod.

### Step 2 — Deploy a GPU Pod

1. RunPod console → **Deploy** → **GPU Pod**
2. GPU: **A100 80 GB** or **H100** (both work; H100 is faster for fine-tuning)
3. Template: **Custom** (paste your Docker image URL)
4. Container image: `<your-dockerhub-username>/synthdata:latest`
5. Container disk: **20 GB** (OS + pip cache; data lives on network volume)
6. Attach the network volume created above → mount path `/workspace`

### Step 3 — Set Environment Variables in RunPod UI

Set these in the **Environment Variables** section of the pod configuration:

| Variable | Value | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | `sk-ant-...` | Required for generate.py, llm_judge.py, topup.py |
| `OLLAMA_HOST` | `0.0.0.0:11434` | Already set in Dockerfile; override here if needed |
| `HF_TOKEN` | `hf_...` | Only needed if pulling gated HuggingFace models |

Do NOT hardcode `ANTHROPIC_API_KEY` in the Dockerfile or any committed file. RunPod env vars are injected securely at pod start.

The `startup.sh` script also reads the key from `/workspace/synthbench/anthropic_key` as a fallback — make sure that file exists on the network volume.

### Step 4 — Expose Ports (optional)

If you want to call the Ollama API from outside the pod:

| Port | Protocol | Purpose |
|---|---|---|
| `11434` | HTTP | Ollama REST API |
| `22` | TCP | SSH (RunPod provides this automatically) |

---

## 3. Volume Mount Summary

| Host path (network volume) | Container path | Purpose |
|---|---|---|
| `synthbench/` (volume root) | `/workspace/synthbench/` | Training data, GGUF models, API key file |
| `synthbench/data/` | `/workspace/synthbench/data/` | train.jsonl, val.jsonl |
| `synthbench/skill/` | `/workspace/synthbench/skill/` | PinchBench repo (cloned once) |
| `synthbench/models/` | `/workspace/synthbench/models/` | Fine-tuned adapters + GGUF exports |
| `synthbench/anthropic_key` | `/workspace/synthbench/anthropic_key` | Plain-text API key (startup.sh reads this) |

Everything under `/workspace` persists across pod restarts. The container disk (`/root`, `/tmp`, etc.) is ephemeral — all important outputs must be written to `/workspace`.

---

## 4. Startup Sequence (once pod is running)

`startup.sh` runs automatically as the container CMD. It performs:

1. Kill any stale Ollama or OpenClaw gateway processes from a previous session
2. Load `ANTHROPIC_API_KEY` from `/workspace/synthbench/anthropic_key` if not already set
3. Copy `openclaw.json` config to `/root/.openclaw/openclaw.json`
4. Start Ollama in the background (`ollama serve`)
5. Start the OpenClaw gateway in the background
6. Health-check: wait for Ollama to respond on port 11434
7. Exit 0 — `tail -f /dev/null` in CMD then keeps the container alive

### Manual verification after deploy

SSH into the pod (RunPod provides the SSH command in the console):

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Check OpenClaw gateway
curl http://localhost:3000/health   # adjust port if openclaw uses a different one

# Pull the base model (only needed once; persists on network volume via ollama's model dir)
ollama pull qwen3:8b

# Run PinchBench baseline
bash /root/benchmark_run.sh ollama/qwen3:8b
```

### If startup.sh failed

```bash
# Re-run manually
bash /root/startup.sh

# Or start services individually
ollama serve &
# then start openclaw gateway per its docs
```

---

## 5. Network Storage Requirements

| Requirement | Detail |
|---|---|
| Region | **CA-2** — existing train.jsonl / val.jsonl data is already here |
| Minimum size | 50 GB |
| Recommended size | 100 GB (leaves room for multiple GGUF exports and HF checkpoints) |
| Mount path | `/workspace` |
| Persistence | Data survives pod stop/restart/termination as long as the volume is not deleted |

**Important**: Always attach the CA-2 volume when creating new pods. If you create a pod in a different region, the network volume will not be available and you will lose access to the training data.

---

## 6. Rebuilding the Image After Code Changes

When you update scripts in the repo (e.g. `finetune.py`, `startup.sh`):

```bash
# On your local machine
cd /Users/chetantekur/Documents/Claude-dir/synthdata

docker build -t <your-dockerhub-username>/synthdata:latest .
docker push <your-dockerhub-username>/synthdata:latest
```

Then in RunPod:
- Stop the current pod
- Edit the pod → update the image tag (or force a repull)
- Restart

Alternatively, for quick one-off changes during a session, you can `scp` a file directly to `/root/` inside a running pod — but always commit the change and rebuild for reproducibility.

---

## 7. Cost Notes

- A100 80 GB on RunPod: ~$2.49/hr (spot) or ~$3.19/hr (on-demand) as of early 2026
- H100 80 GB: ~$3.89/hr (spot)
- Network volume storage: ~$0.07/GB/month
- Stop the pod (do not terminate) when not in use to avoid GPU charges while keeping the network volume
