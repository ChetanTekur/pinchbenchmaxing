# Official PyTorch image: CUDA 12.4 + PyTorch 2.6 + Python 3.12
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# ── System packages ─────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        wget \
        git \
        vim \
        jq \
        build-essential \
        ca-certificates \
        libssl-dev \
        libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Python packages ──────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
        pyyaml \
        anthropic \
        httpx \
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
        PyPDF2 \
        scikit-learn \
        jupyter \
        ipywidgets

# ── uv (fast Python package manager) ─────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# ── Unsloth (CUDA 12.4 + Torch 2.6 build) ────────────────────────────────────
RUN pip install --no-cache-dir \
    "unsloth[cu124-torch260] @ git+https://github.com/unslothai/unsloth.git"

# ── llama.cpp build dependencies (so convert.py doesn't apt-get at runtime) ──
# NOTE: install_llama_cpp_blocking() requires a GPU at import time (Unsloth checks
# for CUDA), so we can't pre-compile llama.cpp in CI. Instead we ensure the build
# deps are present so the first convert run only compiles, no apt-get needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
    && rm -rf /var/lib/apt/lists/*

# ── Ollama ────────────────────────────────────────────────────────────────────
RUN curl -fsSL https://ollama.com/install.sh | sh

# ── Node.js 22 + OpenClaw ─────────────────────────────────────────────────────
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*
RUN npm install -g openclaw@latest && openclaw --version

# ── Bootstrap scripts only (code is cloned by startup.sh at runtime) ─────────
# The full repo is cloned to /root/pbm on first start, then git pull on restarts.
# This keeps the image stable and lets you update code with just git push + pod restart.
COPY scripts/startup.sh  /root/scripts/startup.sh
COPY scripts/set_env.sh  /root/scripts/set_env.sh
RUN chmod +x /root/scripts/*.sh

# ── ENV ───────────────────────────────────────────────────────────────────────
ENV PATH="/root/.local/bin:/root/.openclaw/bin:/usr/local/bin:${PATH}" \
    OLLAMA_HOST="0.0.0.0:11434" \
    PBM_WORKSPACE="/workspace/synthbench" \
    PYTHONPATH="/root/pbm"

WORKDIR /root/pbm

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["/bin/bash", "-c", "bash /root/scripts/startup.sh && tail -f /dev/null"]
