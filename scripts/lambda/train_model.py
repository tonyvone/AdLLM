#!/usr/bin/env python3
"""
AdTech LLM - Model Training Script for Lambda Labs
Fine-tunes Llama 3.1 8B on adtech data using LoRA/PEFT.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
from datasets import Dataset, load_dataset

# Training imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Configuration
DEFAULT_CONFIG = {
    "base_model": "meta-llama/Llama-3.1-8B-Instruct",
    "output_dir": "./models/adtech-llm",
    "num_epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "max_seq_length": 2048,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "use_4bit": True,
    "use_flash_attention": True,
}


def check_gpu():
    """Check GPU availability and print info."""
    print("\n🔍 GPU Check:")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✓ GPU: {gpu_name}")
        print(f"  ✓ Memory: {gpu_memory:.1f} GB")
        print(f"  ✓ CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("  ✗ No GPU detected!")
        print("  This script requires a CUDA GPU for training.")
        return False


def load_training_data(data_dir: Path):
    """Load prepared training data."""
    print("\n📂 Loading training data...")

    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "validation.jsonl"

    if not train_path.exists():
        print(f"  ✗ Training data not found at {train_path}")
        print("  Run download_data.py first!")
        sys.exit(1)

    train_data = pd.read_json(train_path, lines=True)
    val_data = pd.read_json(val_path, lines=True) if val_path.exists() else None

    print(f"  ✓ Training examples: {len(train_data):,}")
    if val_data is not None:
        print(f"  ✓ Validation examples: {len(val_data):,}")

    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data) if val_data is not None else None

    return train_dataset, val_dataset


def format_instruction(example):
    """Format example into instruction template for Llama."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    # Llama 3.1 Instruct format
    if input_text:
        text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are AdTech LLM, an AI assistant specialized in advertising technology. You help with ad creative generation, CTR prediction, fraud detection, and contextual analysis.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}

{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""
    else:
        text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are AdTech LLM, an AI assistant specialized in advertising technology. You help with ad creative generation, CTR prediction, fraud detection, and contextual analysis.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""

    return text


def setup_model_and_tokenizer(config: dict):
    """Load and configure the base model with quantization."""
    print(f"\n🤖 Loading model: {config['base_model']}")

    # Quantization config for 4-bit training
    bnb_config = None
    if config["use_4bit"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("  ✓ Using 4-bit quantization")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"],
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("  ✓ Tokenizer loaded")

    # Load model
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }

    # Add flash attention if available and requested
    if config["use_flash_attention"]:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("  ✓ Using Flash Attention 2")
        except:
            print("  ⚠ Flash Attention not available, using default")

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        **model_kwargs,
    )
    print("  ✓ Model loaded")

    # Prepare for k-bit training
    if config["use_4bit"]:
        model = prepare_model_for_kbit_training(model)
        print("  ✓ Model prepared for 4-bit training")

    return model, tokenizer


def setup_lora(model, config: dict):
    """Configure and apply LoRA adapters."""
    print("\n🔧 Configuring LoRA...")

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ LoRA rank: {config['lora_r']}")
    print(f"  ✓ LoRA alpha: {config['lora_alpha']}")
    print(f"  ✓ Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model


def train(config: dict):
    """Main training function."""
    print("=" * 60)
    print("AdTech LLM - Fine-Tuning Pipeline")
    print("=" * 60)

    # Check GPU
    if not check_gpu():
        print("\n❌ Training requires a GPU. Exiting.")
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

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=100 if val_dataset else None,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        group_by_length=True,
        report_to="none",  # Set to "wandb" if using Weights & Biases
        seed=42,
    )

    print(f"\n📊 Training Configuration:")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Gradient accumulation: {config['gradient_accumulation_steps']}")
    print(f"  Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Output directory: {output_dir}")

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
    print("\n🚀 Starting training...")
    print("-" * 60)

    trainer.train()

    # Save final model
    print("\n💾 Saving model...")
    final_model_path = output_dir / "final_model"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    # Save training config
    config_path = output_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print("✓ Training complete!")
    print(f"  Model saved to: {final_model_path}")
    print(f"  Config saved to: {config_path}")
    print("=" * 60)

    return str(final_model_path)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune Llama for AdTech")

    parser.add_argument("--base-model", default=DEFAULT_CONFIG["base_model"],
                        help="Base model to fine-tune")
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG["output_dir"],
                        help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["num_epochs"],
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"],
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG["learning_rate"],
                        help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=DEFAULT_CONFIG["lora_r"],
                        help="LoRA rank")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization")

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
        "use_4bit": not args.no_4bit,
    })

    train(config)


if __name__ == "__main__":
    main()
