#!/usr/bin/env python3
"""
Validate that the configured base model is usable for fine-tuning.

Checks:
  1. Model exists on HuggingFace
  2. It's a text generation model (not vision-only, diffusion, embedding, etc.)
  3. It's supported by Unsloth (architecture check)
  4. Tokenizer is available
  5. Estimated VRAM requirement

Usage:
  python -m stages.validate_model                    # check config.yaml model
  python -m stages.validate_model --model mistralai/Mistral-7B-v0.3
"""

import argparse
import sys

from utils.config import load_config


# Architectures supported by Unsloth (from their docs + source)
UNSLOTH_SUPPORTED = {
    "llama", "mistral", "gemma", "gemma2", "gemma3",
    "qwen2", "qwen2_moe", "qwen3", "qwen3_moe", "qwen2_vl", "qwen3_5",
    "phi", "phi3", "phi3_v", "phimoe",
    "cohere", "cohere2",
    "deepseek_v2", "deepseek_v3",
    "starcoder2",
    "olmo", "olmo2",
    "granite", "granitemoeshared",
}

# Model types that are NOT LLMs — reject these
NOT_LLM_TYPES = {
    "stable-diffusion", "stable_diffusion", "sdxl",
    "vae", "vit", "clip", "clap", "deit", "beit", "swin",
    "wav2vec2", "whisper", "hubert", "encodec",
    "bert", "roberta", "deberta", "electra", "albert",
    "t5", "bart", "pegasus", "mbart",  # encoder-decoder, not causal LM
    "sam", "mask2former", "detr", "yolos",
    "resnet", "convnext", "mobilenet",
}

# Tags that indicate it's not a text generation model
NOT_TEXT_GEN_TAGS = {
    "text-classification", "token-classification", "fill-mask",
    "image-classification", "object-detection", "image-segmentation",
    "automatic-speech-recognition", "text-to-image", "image-to-text",
    "audio-classification", "voice-activity-detection",
    "sentence-similarity", "feature-extraction",
}


def validate_model(model_id: str) -> dict:
    """
    Validate a HuggingFace model ID. Returns a dict with:
      ok: bool
      errors: list[str]    — fatal issues
      warnings: list[str]  — non-fatal concerns
      info: dict           — model metadata
    """
    errors   = []
    warnings = []
    info     = {}

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        return {"ok": False, "errors": ["huggingface_hub not installed"],
                "warnings": [], "info": {}}

    api = HfApi()

    # ── Check 1: Model exists on HuggingFace ──────────────────────────────
    try:
        model_info = api.model_info(model_id)
        info["model_id"] = model_id
        info["downloads"] = model_info.downloads
        info["likes"] = model_info.likes
        info["tags"] = list(model_info.tags or [])
        info["pipeline_tag"] = model_info.pipeline_tag
        info["library"] = model_info.library_name
    except Exception as e:
        errors.append(f"Model '{model_id}' not found on HuggingFace: {e}")
        return {"ok": False, "errors": errors, "warnings": warnings, "info": info}

    # ── Check 2: It's a text generation model ─────────────────────────────
    pipeline = model_info.pipeline_tag or ""
    if pipeline and pipeline not in ("text-generation", "conversational", "text2text-generation"):
        if pipeline in ("image-classification", "object-detection", "text-to-image",
                         "automatic-speech-recognition", "feature-extraction",
                         "image-segmentation", "audio-classification"):
            errors.append(f"This is a {pipeline} model, not a text generation LLM. "
                          f"Pipeline tag: {pipeline}")
        else:
            warnings.append(f"Unexpected pipeline tag: {pipeline}. "
                            f"Expected 'text-generation'.")

    # Check tags for non-LLM indicators
    tags = set(model_info.tags or [])
    bad_tags = tags & NOT_TEXT_GEN_TAGS
    if bad_tags:
        errors.append(f"Model has non-LLM tags: {bad_tags}. "
                      "This doesn't appear to be a text generation model.")

    # ── Check 3: Get model architecture from config.json ──────────────────
    try:
        import json
        config_path = hf_hub_download(model_id, "config.json")
        config = json.loads(open(config_path).read())
        model_type = config.get("model_type", "unknown")
        info["model_type"] = model_type
        info["hidden_size"] = config.get("hidden_size")
        info["num_hidden_layers"] = config.get("num_hidden_layers")
        info["vocab_size"] = config.get("vocab_size")

        # Calculate approximate parameter count
        hidden = config.get("hidden_size", 0)
        layers = config.get("num_hidden_layers", 0)
        vocab  = config.get("vocab_size", 0)
        if hidden and layers and vocab:
            # Rough estimate: 12 * L * H^2 + V * H (transformer formula)
            params_b = (12 * layers * hidden * hidden + vocab * hidden) / 1e9
            info["estimated_params_b"] = round(params_b, 1)

            # VRAM estimate: ~2 bytes per param for 16-bit, LoRA adds ~10%
            vram_gb = params_b * 2 * 1.1
            info["estimated_vram_gb"] = round(vram_gb, 1)

            if vram_gb > 80:
                warnings.append(f"Estimated {vram_gb:.0f} GB VRAM needed. "
                                "May not fit on a single GPU (A100 80GB max).")
            elif vram_gb > 24:
                warnings.append(f"Estimated {vram_gb:.0f} GB VRAM needed. "
                                "Requires A100 or similar (RTX 4090 = 24GB).")

        # Check if it's a known non-LLM architecture
        if model_type.lower() in NOT_LLM_TYPES:
            errors.append(f"Architecture '{model_type}' is not a causal language model. "
                          "Fine-tuning with Unsloth requires a decoder-only LLM.")

        # Check Unsloth support
        model_type_lower = model_type.lower().replace("-", "_")
        if model_type_lower in UNSLOTH_SUPPORTED:
            info["unsloth_supported"] = True
        else:
            # Check partial matches (e.g. "qwen3_5" matches "qwen3")
            matched = any(model_type_lower.startswith(s) for s in UNSLOTH_SUPPORTED)
            if matched:
                info["unsloth_supported"] = True
            else:
                warnings.append(
                    f"Architecture '{model_type}' is not in the known Unsloth supported list. "
                    f"Unsloth supports: {', '.join(sorted(UNSLOTH_SUPPORTED))}. "
                    "It may still work — Unsloth adds support for new models frequently. "
                    "Run `python -m stages.finetune --dry-run` to verify."
                )
                info["unsloth_supported"] = False

    except Exception as e:
        warnings.append(f"Could not download config.json: {e}")
        info["model_type"] = "unknown"

    # ── Check 4: Tokenizer available ──────────────────────────────────────
    try:
        hf_hub_download(model_id, "tokenizer_config.json")
        info["has_tokenizer"] = True
    except Exception:
        try:
            hf_hub_download(model_id, "tokenizer.json")
            info["has_tokenizer"] = True
        except Exception:
            errors.append("No tokenizer found (tokenizer_config.json or tokenizer.json). "
                          "Fine-tuning requires a tokenizer.")
            info["has_tokenizer"] = False

    # ── Check 5: Gated model ─────────────────────────────────────────────
    if model_info.gated:
        warnings.append("This is a gated model. You need to accept the license on "
                        f"huggingface.co/{model_id} and set HF_TOKEN before training.")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "info": info,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate base model for fine-tuning")
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model ID (default: from config.yaml)")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_id = args.model or cfg.base_model

    print(f"Validating: {model_id}\n")
    result = validate_model(model_id)

    # Print info
    info = result["info"]
    if info.get("model_type"):
        print(f"  Architecture:     {info['model_type']}")
    if info.get("estimated_params_b"):
        print(f"  Parameters:       ~{info['estimated_params_b']}B")
    if info.get("estimated_vram_gb"):
        print(f"  Est. VRAM (LoRA): ~{info['estimated_vram_gb']} GB")
    if info.get("downloads"):
        print(f"  Downloads:        {info['downloads']:,}")
    if "unsloth_supported" in info:
        status = "yes" if info["unsloth_supported"] else "unknown"
        print(f"  Unsloth support:  {status}")
    if info.get("has_tokenizer") is not None:
        print(f"  Tokenizer:        {'found' if info['has_tokenizer'] else 'MISSING'}")
    print()

    # Print warnings
    for w in result["warnings"]:
        print(f"  WARNING: {w}")

    # Print errors
    for e in result["errors"]:
        print(f"  ERROR: {e}")

    if result["ok"]:
        print(f"\n  OK — {model_id} is ready for fine-tuning")
    else:
        print(f"\n  FAILED — fix the errors above before training")
        sys.exit(1)


if __name__ == "__main__":
    main()
