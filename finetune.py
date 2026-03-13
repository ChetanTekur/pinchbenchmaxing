#!/usr/bin/env python3
"""
Fine-tune Qwen3-8B on PinchBench synthetic agent traces.

Uses Unsloth for fast LoRA training + TRL SFTTrainer.
Trains only on assistant turns (tool calls + responses).

Usage:
  python finetune.py              # full training run
  python finetune.py --dry-run    # load model + data, skip training (sanity check)

Requirements:
  pip install "unsloth[cu124-torch260] @ git+https://github.com/unslothai/unsloth.git"
  pip install trl transformers peft datasets accelerate

Model output: /workspace/synthbench/qwen3-8b-clawd/
"""

import os, json, argparse
from pathlib import Path
from datasets import Dataset

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen3-8B"
OUTPUT_DIR   = "/workspace/synthbench/qwen3-8b-clawd"
DATA_DIR     = Path("/workspace/synthbench/data")
TRAIN_FILE   = DATA_DIR / "train_sft.jsonl"
VAL_FILE     = DATA_DIR / "val_sft.jsonl"

MAX_SEQ_LEN  = 4096
LOAD_IN_4BIT = True

# LoRA
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training
EPOCHS            = 3
BATCH_SIZE        = 2       # per device
GRAD_ACCUM        = 4       # effective batch = 8
LEARNING_RATE     = 2e-4
WARMUP_RATIO      = 0.05
LR_SCHEDULER      = "cosine"
SAVE_STEPS        = 100
LOGGING_STEPS     = 10
MAX_GRAD_NORM     = 1.0
WEIGHT_DECAY      = 0.01


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def build_dataset(path: Path) -> Dataset:
    records = load_jsonl(path)
    return Dataset.from_list(records)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(dry_run: bool = False):
    # ── 1. Load model + tokenizer via Unsloth ────────────────────────────────
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainingArguments

    print(f"Loading {MODEL_NAME} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name    = MODEL_NAME,
        max_seq_length= MAX_SEQ_LEN,
        load_in_4bit  = LOAD_IN_4BIT,
        dtype         = None,   # auto-detect (bf16 on Ampere+)
    )

    # ── 2. Apply LoRA ─────────────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        target_modules = TARGET_MODULES,
        bias           = "none",
        use_gradient_checkpointing = "unsloth",
        random_state   = 42,
    )
    model.print_trainable_parameters()

    # ── 3. Load datasets ──────────────────────────────────────────────────────
    print(f"\nLoading datasets ...")
    train_dataset = build_dataset(TRAIN_FILE)
    val_dataset   = build_dataset(VAL_FILE)
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Val:   {len(val_dataset)} examples")

    if dry_run:
        print("\nDry run complete — model and data loaded successfully.")
        return

    # ── 4. Apply chat template ─────────────────────────────────────────────────
    # Format messages using Qwen3's chat template
    def format_example(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize        = False,
            add_generation_prompt = False,
        )
        return {"text": text}

    train_dataset = train_dataset.map(format_example, remove_columns=["messages"])
    val_dataset   = val_dataset.map(format_example,   remove_columns=["messages"])

    # ── 5. Training args ──────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = LEARNING_RATE,
        lr_scheduler_type           = LR_SCHEDULER,
        warmup_ratio                = WARMUP_RATIO,
        weight_decay                = WEIGHT_DECAY,
        max_grad_norm               = MAX_GRAD_NORM,
        logging_steps               = LOGGING_STEPS,
        save_steps                  = SAVE_STEPS,
        eval_strategy               = "steps",
        eval_steps                  = SAVE_STEPS,
        save_total_limit            = 3,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        bf16                        = True,
        max_seq_length              = MAX_SEQ_LEN,
        dataset_text_field          = "text",
        packing                     = False,
        report_to                   = "none",
    )

    # ── 6. Train only on assistant turns ─────────────────────────────────────
    # Unsloth's train_on_responses_only masks everything except assistant turns
    from unsloth.chat_templates import train_on_responses_only

    trainer = SFTTrainer(
        model         = model,
        tokenizer     = tokenizer,
        train_dataset = train_dataset,
        eval_dataset  = val_dataset,
        args          = training_args,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part    = "<|im_start|>assistant\n",
    )

    # ── 7. Train ──────────────────────────────────────────────────────────────
    print("\nStarting training ...")
    trainer.train()

    # ── 8. Save final model ───────────────────────────────────────────────────
    print(f"\nSaving model to {OUTPUT_DIR} ...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Also save merged 16-bit weights for easy inference/upload
    merged_dir = OUTPUT_DIR + "_merged"
    print(f"Saving merged 16-bit model to {merged_dir} ...")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    print("\nDone!")
    print(f"  LoRA adapter: {OUTPUT_DIR}")
    print(f"  Merged model: {merged_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Load model and data without training")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
