# RunPod GPU image: CUDA 12.4 + PyTorch 2.6 + Python 3.12 + Ubuntu 22.04
FROM runpod/pytorch:2.6.0-py3.12-cuda12.4.1-devel-ubuntu22.04

# ── System packages ─────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        wget \
        git \
        vim \
        jq \
        build-essential \
        nodejs \
        npm \
    && rm -rf /var/lib/apt/lists/*

# ── Python packages ──────────────────────────────────────────────────────────
# Install core ML / data / utility packages first (stable PyPI)
RUN pip install --no-cache-dir \
        anthropic \
        trl \
        transformers \
        peft \
        datasets \
        accelerate \
        huggingface_hub \
        safetensors \
        tqdm \
        pandas \
        openpyxl \
        pdfplumber \
        PyPDF2

# ── uv (fast Python package manager) ─────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# ── Unsloth (CUDA 12.4 + Torch 2.6 build) ────────────────────────────────────
RUN pip install --no-cache-dir \
    "unsloth[cu124-torch260] @ git+https://github.com/unslothai/unsloth.git"

# ── Ollama ────────────────────────────────────────────────────────────────────
# Installed to /usr/local/bin — must be *started* at runtime (needs GPU)
RUN curl -fsSL https://ollama.com/install.sh | sh

# ── OpenClaw (npm global) ─────────────────────────────────────────────────────
RUN npm install -g openclaw

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /workspace/synthbench

# ── Copy repo scripts ─────────────────────────────────────────────────────────
# /workspace is a RunPod network volume — do NOT copy data files there at
# build time.  Only copy the scripts that live in the repo root.
COPY startup.sh              /root/startup.sh
COPY benchmark_run.sh        /root/benchmark_run.sh
COPY fix_modelfile.sh        /root/fix_modelfile.sh
COPY setup.sh                /root/setup.sh
COPY generate.py             /root/generate.py
COPY inspect_data.py         /root/inspect_data.py
COPY llm_judge.py            /root/llm_judge.py
COPY repair.py               /root/repair.py
COPY topup.py                /root/topup.py
COPY finetune.py             /root/finetune.py
COPY prepare_data.py         /root/prepare_data.py
COPY openclaw_template.json  /root/openclaw_template.json

RUN chmod +x /root/startup.sh /root/benchmark_run.sh /root/fix_modelfile.sh /root/setup.sh

# ── ENV ───────────────────────────────────────────────────────────────────────
# PATH already includes /root/.local/bin (uv) and standard npm global bin.
# ANTHROPIC_API_KEY must be injected at runtime via RunPod env vars — never
# hardcoded here.
ENV PATH="/root/.local/bin:/usr/local/bin:${PATH}" \
    OLLAMA_HOST="0.0.0.0:11434"

# ── Entrypoint ────────────────────────────────────────────────────────────────
# startup.sh handles: kill stale procs, load API key from /workspace, start
# ollama, configure OpenClaw, health-check.  tail -f keeps the container alive
# so RunPod keeps the pod running and you can SSH / exec in.
CMD ["/bin/bash", "-c", "bash /root/startup.sh && tail -f /dev/null"]
