#!/usr/bin/env python3
"""
Convert merged fine-tuned model to GGUF (Q4_K_M) for Ollama.

Usage:
  python convert_to_gguf.py
  python convert_to_gguf.py --merged /workspace/synthbench/qwen35-9b-clawd_merged
  python convert_to_gguf.py --quant q8_0   # higher quality, bigger file

Output: /workspace/synthbench/qwen35-9b-clawd_gguf/qwen35-9b.Q4_K_M.gguf
After this runs, register with Ollama: bash /root/fix_modelfile.sh
"""

import argparse
import shutil
from pathlib import Path

DEFAULT_MERGED = "/workspace/synthbench/qwen35-9b-clawd_merged"
DEFAULT_OUT    = "/workspace/synthbench/qwen35-9b-clawd_gguf"
FINAL_NAME     = "qwen35-9b-clawd"   # fix_modelfile.sh expects qwen35-9b-clawd.Q4_K_M.gguf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", default=DEFAULT_MERGED)
    parser.add_argument("--out",    default=DEFAULT_OUT)
    parser.add_argument("--quant",  default="q4_k_m", help="Quantization method (q4_k_m, q8_0, f16)")
    args = parser.parse_args()

    merged_path = Path(args.merged)
    out_dir     = Path(args.out)

    if not merged_path.exists():
        raise FileNotFoundError(f"Merged model not found: {merged_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading merged model from {merged_path} ...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        str(merged_path),
        max_seq_length=4096,
        load_in_4bit=False,   # already merged 16-bit weights
        dtype=None,
    )

    print(f"Converting to GGUF ({args.quant.upper()}) → {out_dir} ...")
    model.save_pretrained_gguf(
        str(out_dir),
        tokenizer,
        quantization_method=args.quant,
    )

    # Unsloth names the file differently depending on version — find it and rename
    gguf_files = list(out_dir.glob("*.gguf"))
    if not gguf_files:
        raise RuntimeError(f"No .gguf file found in {out_dir} after conversion")

    expected_name = f"{FINAL_NAME}.{args.quant.upper()}.gguf"
    target = out_dir / expected_name

    if gguf_files[0].name != expected_name:
        print(f"Renaming {gguf_files[0].name} → {expected_name}")
        shutil.move(str(gguf_files[0]), str(target))

    print(f"\nDone! GGUF at: {target}")
    print(f"\nNext: bash /root/fix_modelfile.sh")


if __name__ == "__main__":
    main()
