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
        per_device_train_batch_size =int(t["batch_size"]),
        per_device_eval_batch_size  =int(t["batch_size"]),
        gradient_accumulation_steps =int(t["grad_accum"]),
        learning_rate               =float(t["learning_rate"]),
        lr_scheduler_type           =str(t["lr_scheduler"]),
        warmup_ratio                =float(t["warmup_ratio"]),
        weight_decay                =float(t["weight_decay"]),
        max_grad_norm               =float(t["max_grad_norm"]),
        logging_steps               =10,
        save_steps                  =100,
        eval_strategy               ="steps",
        eval_steps                  =100,
        save_total_limit            =3,
        load_best_model_at_end      =True,
        metric_for_best_model       ="eval_loss",
        bf16                        =True,
        max_seq_length              =t["max_seq_len"],
        dataset_text_field          ="text",
        packing                     =False,
        report_to                   ="none",
    )

    from unsloth.chat_templates import train_on_responses_only

    trainer = SFTTrainer(
        model         =model,
        tokenizer     =tokenizer,
        train_dataset =train_dataset,
        eval_dataset  =val_dataset,
        args          =training_args,
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part   ="<|im_start|>assistant\n",
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
