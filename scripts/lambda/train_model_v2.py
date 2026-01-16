#!/usr/bin/env python3
"""
AdTech LLM - Model Training Script V2 (Fixed)
Fine-tunes Llama 3.1 8B with improved hyperparameters to prevent catastrophic forgetting.

Key Fixes from V1:
1. Lower learning rate (5e-5 vs 2e-4) - preserves base model capabilities
2. Single epoch - prevents overfitting
3. Higher LoRA alpha ratio - better adaptation
4. Warmup ratio increased - smoother training start
5. Lower weight decay - less regularization on small data
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
from datasets import Dataset

# Training imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ============================================
# V2 CONFIGURATION - FIXED HYPERPARAMETERS
# ============================================
DEFAULT_CONFIG = {
    # Model
    "base_model": "meta-llama/Llama-3.1-8B-Instruct",
    "output_dir": "./models/adtech-llm-v2",

    # Training - CHANGED FROM V1
    "num_epochs": 1,           # V1 was 2-3, reduced to prevent overfitting
    "batch_size": 4,
    "gradient_accumulation_steps": 4,  # Effective batch = 16
    "learning_rate": 5e-5,     # V1 was 2e-4 (4x LOWER now)
    "warmup_ratio": 0.1,       # V1 was 0.03 (increased for stability)
    "weight_decay": 0.001,     # V1 was 0.01 (reduced)
    "max_seq_length": 2048,

    # LoRA - Slightly adjusted
    "lora_r": 16,
    "lora_alpha": 64,          # V1 was 32, now 4x rank (stronger adaptation)
    "lora_dropout": 0.1,       # V1 was 0.05 (increased for regularization)

    # Quantization
    "use_4bit": True,
    "use_flash_attention": True,
}


def check_gpu():
    """Check GPU availability and print info."""
    print("\n" + "=" * 50)
    print("GPU CHECK")
    print("=" * 50)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name}")
        print(f"  Memory: {gpu_memory:.1f} GB")
        print(f"  CUDA: {torch.version.cuda}")
        return True
    else:
        print("  ERROR: No GPU detected!")
        return False


def load_training_data(data_dir: Path):
    """Load prepared training data."""
    print("\n" + "=" * 50)
    print("LOADING DATA")
    print("=" * 50)

    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "validation.jsonl"

    if not train_path.exists():
        print(f"  ERROR: Training data not found at {train_path}")
        print("  Run download_data_v2.py first!")
        sys.exit(1)

    # Load JSONL files
    train_data = []
    with open(train_path, 'r') as f:
        for line in f:
            train_data.append(json.loads(line))

    val_data = []
    if val_path.exists():
        with open(val_path, 'r') as f:
            for line in f:
                val_data.append(json.loads(line))

    print(f"  Training examples: {len(train_data):,}")
    print(f"  Validation examples: {len(val_data):,}")

    # Show task distribution
    task_counts = {}
    for item in train_data:
        task = item.get("task_type", "unknown")
        task_counts[task] = task_counts.get(task, 0) + 1

    print("\n  Task Distribution:")
    for task, count in sorted(task_counts.items()):
        pct = count / len(train_data) * 100
        print(f"    {task}: {count:,} ({pct:.1f}%)")

    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data) if val_data else None

    return train_dataset, val_dataset


def format_instruction(example):
    """
    Format example into Llama 3.1 Instruct template.

    IMPORTANT: This must match inference format exactly!
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    # Build the prompt
    system_prompt = """You are AdTech LLM, an expert AI assistant specialized in advertising technology.

You help with:
- Creating comprehensive media plans across CTV, Linear TV, Digital, and Social channels
- Writing compelling ad creative (headlines, copy, CTAs)
- Predicting ad performance and CTR
- Detecting ad fraud and invalid traffic
- Analyzing content for brand safety
- Optimizing campaign budgets and bidding strategies

Always provide detailed, actionable responses. When given a budget, calculate specific allocations. When asked for creative, provide multiple options."""

    # Llama 3.1 Instruct format
    if input_text:
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction

    text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|}

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""

    return text


def setup_model_and_tokenizer(config: dict):
    """Load and configure the base model with quantization."""
    print("\n" + "=" * 50)
    print(f"LOADING MODEL: {config['base_model']}")
    print("=" * 50)

    # Quantization config for 4-bit training
    bnb_config = None
    if config["use_4bit"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("  Using 4-bit quantization (QLoRA)")

    # Load tokenizer
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"],
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    print("  Loading model weights...")
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }

    if config["use_flash_attention"]:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("  Using Flash Attention 2")
        except Exception:
            print("  Flash Attention not available, using default")

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        **model_kwargs,
    )

    # Prepare for k-bit training
    if config["use_4bit"]:
        model = prepare_model_for_kbit_training(model)

    print("  Model loaded successfully!")
    return model, tokenizer


def setup_lora(model, config: dict):
    """Configure and apply LoRA adapters."""
    print("\n" + "=" * 50)
    print("CONFIGURING LoRA")
    print("=" * 50)

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",      # MLP
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"  LoRA rank (r): {config['lora_r']}")
    print(f"  LoRA alpha: {config['lora_alpha']}")
    print(f"  LoRA dropout: {config['lora_dropout']}")
    print(f"  Alpha/Rank ratio: {config['lora_alpha']/config['lora_r']}x")
    print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model


def train(config: dict):
    """Main training function."""
    print("\n" + "=" * 60)
    print("  AdTech LLM V2 - Fine-Tuning Pipeline")
    print("  (Fixed hyperparameters to prevent catastrophic forgetting)")
    print("=" * 60)

    # Check GPU
    if not check_gpu():
        print("\nERROR: Training requires a GPU. Exiting.")
        sys.exit(1)

    # Load data
    data_dir = Path("./data/processed")
    train_dataset, val_dataset = load_training_data(data_dir)

    # Setup model
    model, tokenizer = setup_model_and_tokenizer(config)
    model = setup_lora(model, config)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config["output_dir"]) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments - V2 OPTIMIZED
    print("\n" + "=" * 50)
    print("TRAINING CONFIGURATION (V2 - Fixed)")
    print("=" * 50)

    training_args = TrainingArguments(
        output_dir=str(output_dir),

        # Core training params
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],

        # Learning rate - CRITICAL FIX
        learning_rate=config["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=config["warmup_ratio"],

        # Regularization
        weight_decay=config["weight_decay"],
        max_grad_norm=0.3,

        # Precision & optimization
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,

        # Logging & saving
        logging_steps=25,
        save_steps=200,
        save_total_limit=2,
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=200 if val_dataset else None,

        # Other
        group_by_length=True,
        report_to="none",
        seed=42,
    )

    effective_batch = config['batch_size'] * config['gradient_accumulation_steps']

    print(f"  Epochs: {config['num_epochs']} (V1 was 2-3)")
    print(f"  Learning Rate: {config['learning_rate']} (V1 was 2e-4, now 4x lower)")
    print(f"  Warmup Ratio: {config['warmup_ratio']} (V1 was 0.03)")
    print(f"  Weight Decay: {config['weight_decay']} (V1 was 0.01)")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Gradient Accumulation: {config['gradient_accumulation_steps']}")
    print(f"  Effective Batch Size: {effective_batch}")
    print(f"  Output: {output_dir}")

    # Estimate training time
    steps_per_epoch = len(train_dataset) // effective_batch
    total_steps = steps_per_epoch * config['num_epochs']
    estimated_time = total_steps * 2.5 / 60  # ~2.5 sec/step on A100
    print(f"\n  Estimated Steps: {total_steps:,}")
    print(f"  Estimated Time: {estimated_time:.1f} minutes ({estimated_time/60:.1f} hours)")

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        formatting_func=format_instruction,
        max_seq_length=config["max_seq_length"],
        packing=False,
    )

    # Start training
    print("\n" + "=" * 50)
    print("STARTING TRAINING")
    print("=" * 50)

    trainer.train()

    # Save final model
    print("\n" + "=" * 50)
    print("SAVING MODEL")
    print("=" * 50)

    final_model_path = output_dir / "final_model"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    # Save training config
    config_path = output_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Save V2 changes documentation
    changes_doc = """# AdTech LLM V2 Training Changes

## Hyperparameter Fixes (vs V1)

| Parameter | V1 Value | V2 Value | Reason |
|-----------|----------|----------|--------|
| Learning Rate | 2e-4 | 5e-5 | 4x lower to preserve base model knowledge |
| Epochs | 2-3 | 1 | Prevent overfitting on domain data |
| Warmup Ratio | 0.03 | 0.1 | Smoother training start |
| Weight Decay | 0.01 | 0.001 | Less aggressive regularization |
| LoRA Alpha | 32 | 64 | Stronger adaptation signal |
| LoRA Dropout | 0.05 | 0.1 | More regularization |

## Data Fixes

- Added general instruction data (10%) to prevent catastrophic forgetting
- Balanced task distribution (20-25% per category)
- Added real media planning examples
- Cleaned categorical label leakage from outputs

## Expected Improvements

1. Model retains general knowledge (what is programmatic advertising?)
2. Model generates real media plans, not CSV field names
3. Model respects user input (budgets, brands, timelines)
4. Creative output is contextually relevant
"""

    with open(output_dir / "V2_CHANGES.md", "w") as f:
        f.write(changes_doc)

    print(f"  Model saved: {final_model_path}")
    print(f"  Config saved: {config_path}")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n  Next: Test the model with:")
    print(f"  python scripts/lambda/test_model.py --model {final_model_path}")

    return str(final_model_path)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune Llama for AdTech (V2)")

    parser.add_argument("--base-model", default=DEFAULT_CONFIG["base_model"])
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["num_epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--lora-r", type=int, default=DEFAULT_CONFIG["lora_r"])
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_CONFIG["lora_alpha"])
    parser.add_argument("--no-4bit", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    config = DEFAULT_CONFIG.copy()
    config.update({
        "base_model": args.base_model,
        "output_dir": args.output_dir,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "use_4bit": not args.no_4bit,
    })

    train(config)


if __name__ == "__main__":
    main()
