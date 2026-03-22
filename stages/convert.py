#!/usr/bin/env python3
"""
Convert merged fine-tuned model to GGUF for Ollama.

The GGUF file is written to <workspace>/<name>_merged_gguf/<name>_merged.<QUANT>.gguf

Usage:
  python -m stages.convert
  python -m stages.convert --quant q8_0

After this runs, register with Ollama:
  bash scripts/register_model.sh
"""

import argparse
import shutil
from pathlib import Path

from utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--quant",  default=None, help="Override quantization (q4_k_m, q8_0, f16)")
    parser.add_argument("--merged", default=None, help="Override merged model path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    quant       = args.quant or cfg["convert"]["quantization"]
    merged_path = Path(args.merged) if args.merged else cfg.merged_dir
    gguf_dir    = cfg.gguf_dir
    # Build gguf_file using the effective quant (which may be overridden via --quant)
    gguf_file   = gguf_dir / f"{cfg.model_name}_merged.{quant.upper()}.gguf"

    if not merged_path.exists():
        raise FileNotFoundError(
            f"Merged model not found at {merged_path}\n"
            f"Run stages/finetune.py first."
        )

    gguf_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base model : {cfg.base_model}")
    print(f"Source     : {merged_path}")
    print(f"Output     : {gguf_file}")
    print(f"Quant      : {quant.upper()}")

    # Preflight: check disk space — conversion needs ~20GB temp on root
    import shutil
    root_free = shutil.disk_usage("/").free / (1024**3)
    print(f"Root disk  : {root_free:.1f} GB free")
    if root_free < 15:
        raise RuntimeError(
            f"Not enough disk space on root ({root_free:.1f} GB free, need ≥15 GB). "
            f"Free space: rm -rf ~/.cache/huggingface/hub or symlink to network volume."
        )
    print()

    from unsloth import FastLanguageModel

    print("Loading merged model ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        str(merged_path),
        max_seq_length=cfg["training"]["max_seq_len"],
        load_in_4bit=False,
        dtype=None,
    )

    # Unsloth always writes to {merged_path}_gguf/ regardless of what path we pass.
    # Pass merged_path (where config.json lives), then find the output in merged_gguf_dir.
    print(f"Converting to GGUF ({quant.upper()}) ...")
    model.save_pretrained_gguf(
        str(merged_path),
        tokenizer,
        quantization_method=quant,
    )

    # Unsloth saves to <merged_path>_gguf/, not gguf_dir
    unsloth_out = merged_path.parent / (merged_path.name + "_gguf")
    candidates = list(unsloth_out.glob(f"*.{quant.upper()}.gguf")) or list(unsloth_out.glob("*.gguf"))
    if not candidates:
        raise RuntimeError(f"No .gguf file found in {unsloth_out} after conversion.")

    src = candidates[0]
    gguf_dir.mkdir(parents=True, exist_ok=True)
    print(f"Moving {src.name} → {gguf_file}")
    shutil.move(str(src), str(gguf_file))

    print(f"\nDone! GGUF at: {gguf_file}")
    print(f"\nNext step: register with Ollama")
    print(f"  OLLAMA_MODEL=<name> bash scripts/register_model.sh")


if __name__ == "__main__":
    main()
