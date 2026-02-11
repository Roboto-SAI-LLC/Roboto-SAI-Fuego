"""
train_lora_fim.py — LoRA fine-tuning pilot for Qwen2.5-Coder-3B on FIM data.

Loads the base model from HuggingFace, applies LoRA adapters, and trains on
our locally-generated FIM examples. Produces a LoRA adapter that can be
merged and exported to GGUF for llama.cpp inference.

CPU Pilot Mode (--pilot):
    Runs 5 steps on 10 examples to verify the pipeline end-to-end.
    Takes ~10-20 minutes on CPU.

Full Training (GPU required):
    python train_lora_fim.py --epochs 3 --batch-size 4

Usage:
    python train_lora_fim.py --pilot                    # Quick CPU pilot
    python train_lora_fim.py --epochs 3 --batch-size 4  # Full GPU training
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# ── Config ──────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-Coder-3B"  # Base model (not instruct)
FIM_DATA = Path("apps/model-training/data/fim_pilot.jsonl")
OUTPUT_DIR = Path("apps/model-training/output/lora-fim-qwen3b")
MAX_SEQ_LENGTH = 2048
SEED = 929  # Roberto's sigil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning on FIM data")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Quick pilot run: 10 examples, 5 steps (CPU-safe).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=FIM_DATA,
        help="Path to FIM JSONL data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for LoRA adapter.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID,
        help="HuggingFace model ID.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=MAX_SEQ_LENGTH,
        help="Maximum sequence length.",
    )
    return parser.parse_args()


def load_fim_data(data_path: Path, max_examples: int = 0) -> Dataset:
    """Load FIM JSONL, extracting the 'text' field."""
    records = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records.append({"text": record["text"]})
            if max_examples and len(records) >= max_examples:
                break
    print(f"Loaded {len(records)} FIM examples from {data_path}")
    return Dataset.from_list(records)


def tokenize_dataset(
    dataset: Dataset,
    tokenizer,
    max_length: int,
) -> Dataset:
    """Tokenize the text field and truncate to max_length."""
    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        # For causal LM, labels = input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )


def main():
    args = parse_args()

    # Pilot mode overrides
    if args.pilot:
        print("=" * 60)
        print("  PILOT MODE: 10 examples, 5 steps (CPU-safe)")
        print("=" * 60)
        max_examples = 10
        max_steps = 5
    else:
        max_examples = 0  # all
        max_steps = 0     # use epochs instead

    # ── Check data ──────────────────────────────────────────────────────
    if not args.data.exists():
        print(f"Error: FIM data not found at {args.data}")
        print("Run extract_fim_pilot.py first.")
        sys.exit(1)

    # ── Device ──────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"
    use_bf16 = False
    if device == "cuda" and torch.cuda.is_bf16_supported():
        use_bf16 = True
        use_fp16 = False

    print(f"Device: {device}")
    print(f"FP16: {use_fp16} | BF16: {use_bf16}")

    # ── Load tokenizer ──────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Verify FIM tokens are in vocabulary
    fim_tokens = ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"]
    for token in fim_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id:
            print(f"WARNING: FIM token '{token}' not in vocabulary!")
        else:
            print(f"  FIM token '{token}' -> id {token_id}")

    # ── Load data ───────────────────────────────────────────────────────
    dataset = load_fim_data(args.data, max_examples=max_examples)
    dataset = tokenize_dataset(dataset, tokenizer, args.max_seq_length)

    avg_tokens = sum(len(ids) for ids in dataset["input_ids"]) / len(dataset)
    print(f"Avg tokens per example: {avg_tokens:.0f}")

    # ── Load model ──────────────────────────────────────────────────────
    print(f"Loading model: {args.model_id}")
    model_kwargs = {
        "trust_remote_code": True,
    }

    # Use bfloat16 on CPU to halve memory (3B × 2 bytes ≈ 6GB vs 12GB for float32)
    # For GPU, could add load_in_4bit=True with bitsandbytes
    if device == "cpu":
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["low_cpu_mem_usage"] = True
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    # ── Apply LoRA ──────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training args ───────────────────────────────────────────────────
    training_args_dict = {
        "output_dir": str(args.output),
        "per_device_train_batch_size": 1 if args.pilot else args.batch_size,
        "gradient_accumulation_steps": 1 if args.pilot else args.grad_accum,
        "learning_rate": args.lr,
        "warmup_steps": 2 if args.pilot else 10,
        "logging_steps": 1,
        "save_steps": 50 if not args.pilot else 999999,
        "save_total_limit": 2,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "optim": "adamw_torch",
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "seed": SEED,
        "report_to": "none",
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
    }

    if args.pilot or max_steps > 0:
        training_args_dict["max_steps"] = max_steps if max_steps else 5
    else:
        training_args_dict["num_train_epochs"] = args.epochs

    training_args = TrainingArguments(**training_args_dict)

    # ── Data collator ───────────────────────────────────────────────────
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
    )

    # ── Train ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("\nStarting training...")
    result = trainer.train()

    print(f"\nTraining complete!")
    print(f"  Loss: {result.training_loss:.4f}")
    print(f"  Steps: {result.global_step}")
    print(f"  Runtime: {result.metrics['train_runtime']:.1f}s")

    # ── Save adapter ────────────────────────────────────────────────────
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"\nLoRA adapter saved to {args.output}")

    # ── Save training metadata ──────────────────────────────────────────
    meta = {
        "base_model": args.model_id,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "data_source": str(args.data),
        "examples": len(dataset),
        "final_loss": result.training_loss,
        "total_steps": result.global_step,
        "runtime_seconds": result.metrics["train_runtime"],
        "device": device,
        "pilot": args.pilot,
        "seed": SEED,
    }
    meta_path = args.output / "training_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Training metadata saved to {meta_path}")

    if args.pilot:
        print("\n" + "=" * 60)
        print("  PILOT COMPLETE — Pipeline verified end-to-end!")
        print("  For real training, use GPU:")
        print("    python train_lora_fim.py --epochs 3 --batch-size 4")
        print("=" * 60)


if __name__ == "__main__":
    main()
