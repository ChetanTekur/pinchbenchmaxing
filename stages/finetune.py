#!/usr/bin/env python3
"""
Fine-tune a model on PinchBench synthetic agent traces.

Uses Unsloth for fast LoRA training + TRL SFTTrainer.
Trains only on assistant turns. Thinking mode disabled for agent tasks.

Usage:
  python stages/finetune.py
  python stages/finetune.py --dry-run
  python stages/finetune.py --config /path/to/config.yaml
"""

import argparse
import json
from pathlib import Path

from utils.config import load_config


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def auto_batch_size(config_batch: int) -> tuple[int, int]:
    """
    Calculate batch size based on ACTUAL free GPU VRAM after model loading.

    Called AFTER model is loaded so we measure real free VRAM, not a guess.
    Targets effective batch ~32 via grad_accum.
    ~2 GB per batch item for 9B 4-bit LoRA (activations + grad buffers).
    """
    try:
        import torch
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            free_vram = total_vram - reserved

            # ~2 GB per batch item for 4-bit LoRA (measured empirically)
            # Use 70% of free VRAM to leave headroom for optimizer states
            usable_vram = free_vram * 0.7
            max_batch = max(2, int(usable_vram / 2))
            # Cap at 16 — beyond this, offloading kicks in on most GPUs
            batch_size = min(max_batch, 16)
            # Adjust grad_accum to keep effective batch ~32
            effective_target = 32
            grad_accum = max(1, effective_target // batch_size)

            print(f"GPU        : {torch.cuda.get_device_name(0)}")
            print(f"VRAM       : {total_vram:.0f} GB total, {allocated:.1f} GB allocated, "
                  f"{free_vram:.0f} GB free")
            print(f"Batch size : {config_batch} (config) → {batch_size} (auto)")
            print(f"Grad accum : {grad_accum} (effective batch = {batch_size * grad_accum})")
            return batch_size, grad_accum
    except Exception:
        pass

    print(f"Batch size : {config_batch} (config, no GPU detected for auto-tuning)")
    return config_batch, 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Load model and data but skip training")
    args = parser.parse_args()

    cfg = load_config(args.config)
    t   = cfg["training"]

    print(f"Model      : {cfg.base_model}")
    print(f"Output     : {cfg.adapter_dir}")
    print(f"Train data : {cfg.train_sft_file}")
    print(f"Val data   : {cfg.val_sft_file}")
    print()

    for path in [cfg.train_sft_file, cfg.val_sft_file]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run stages/prepare.py first.")

    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    print(f"Loading {cfg.base_model} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name    =cfg.base_model,
        max_seq_length=t["max_seq_len"],
        load_in_4bit  =True,
        dtype         =None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r              =t["lora_r"],
        lora_alpha     =t["lora_alpha"],
        lora_dropout   =t["lora_dropout"],
        target_modules =["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias           ="none",
        use_gradient_checkpointing="unsloth",
        random_state   =42,
    )
    model.print_trainable_parameters()

    # Auto batch size AFTER model loaded — measures actual free VRAM
    batch_size, grad_accum = auto_batch_size(int(t["batch_size"]))

    train_dataset = Dataset.from_list(load_jsonl(cfg.train_sft_file))
    val_dataset   = Dataset.from_list(load_jsonl(cfg.val_sft_file))
    print(f"\nTrain: {len(train_dataset)} examples")
    print(f"Val:   {len(val_dataset)} examples")

    if args.dry_run:
        print("\nDry run complete.")
        return

    def format_example(example):
        try:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        return {"text": text}

    train_dataset = train_dataset.map(format_example, remove_columns=["messages"])
    val_dataset   = val_dataset.map(format_example,   remove_columns=["messages"])

    cfg.adapter_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir                  =str(cfg.adapter_dir),
        num_train_epochs            =int(t["epochs"]),
        per_device_train_batch_size =batch_size,
        per_device_eval_batch_size  =1,  # eval skips grad checkpointing → more VRAM
        gradient_accumulation_steps =grad_accum,
        learning_rate               =float(t["learning_rate"]),
        lr_scheduler_type           =str(t["lr_scheduler"]),
        warmup_ratio                =float(t["warmup_ratio"]),
        weight_decay                =float(t["weight_decay"]),
        max_grad_norm               =float(t["max_grad_norm"]),
        logging_steps               =10,
        logging_strategy            ="steps",
        save_steps                  =50,
        eval_strategy               ="no",     # eval OOMs on 24GB; train loss is sufficient
        save_total_limit            =3,
        disable_tqdm                =False,
        bf16                        =True,
        max_seq_length              =t["max_seq_len"],
        dataset_text_field          ="text",
        packing                     =False,
        report_to                   ="none",
    )

    from unsloth.chat_templates import train_on_responses_only

    # ── Verify chat template markers before training ────────────────────
    # If these don't match, train_on_responses_only silently becomes a
    # no-op and the model trains on full text (including user turns).
    test_msgs = [
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": "reply"},
    ]
    try:
        formatted = tokenizer.apply_chat_template(
            test_msgs, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        formatted = tokenizer.apply_chat_template(
            test_msgs, tokenize=False, add_generation_prompt=False,
        )

    instruction_marker = "<|im_start|>user\n"
    response_marker = "<|im_start|>assistant\n"

    if instruction_marker not in formatted:
        print(f"\n⚠ WARNING: instruction_part '{instruction_marker}' NOT FOUND in chat template!")
        print(f"  Actual template:\n{repr(formatted)}")
        print(f"  train_on_responses_only will be a NO-OP — model will train on ALL tokens!")
        raise RuntimeError(
            f"Chat template mismatch: '{instruction_marker}' not in tokenizer output. "
            f"Update instruction_part/response_part in finetune.py to match."
        )
    if response_marker not in formatted:
        print(f"\n⚠ WARNING: response_part '{response_marker}' NOT FOUND in chat template!")
        print(f"  Actual template:\n{repr(formatted)}")
        raise RuntimeError(
            f"Chat template mismatch: '{response_marker}' not in tokenizer output."
        )
    print(f"\n✓ Chat template markers verified:")
    print(f"  instruction: '{instruction_marker}' — found")
    print(f"  response:    '{response_marker}' — found")

    trainer = SFTTrainer(
        model         =model,
        tokenizer     =tokenizer,
        train_dataset =train_dataset,
        eval_dataset  =val_dataset,
        args          =training_args,
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_marker,
        response_part   =response_marker,
    )

    print("\nStarting training ...")
    trainer.train()

    print(f"\nSaving adapter to {cfg.adapter_dir} ...")
    model.save_pretrained(str(cfg.adapter_dir))
    tokenizer.save_pretrained(str(cfg.adapter_dir))

    print(f"Saving merged 16-bit model to {cfg.merged_dir} ...")
    cfg.merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(cfg.merged_dir), tokenizer, save_method="merged_16bit")

    print("\nDone!")
    print(f"  Adapter : {cfg.adapter_dir}")
    print(f"  Merged  : {cfg.merged_dir}")
    print(f"\nNext step: python stages/convert.py")


if __name__ == "__main__":
    main()
