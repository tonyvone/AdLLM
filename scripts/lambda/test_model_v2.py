#!/usr/bin/env python3
"""
AdTech LLM V2 - Model Testing Script
Run this after training to verify the model works correctly.
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_model(model_path: str, base_model: str = "meta-llama/Llama-3.1-8B-Instruct"):
    """Load the fine-tuned model."""
    print(f"\nLoading model from: {model_path}")

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model
    print("  Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Load LoRA adapter
    print("  Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, model_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    print("  Model loaded successfully!")
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 1024, temperature: float = 0.7):
    """Generate a response from the model."""

    system_prompt = """You are AdTech LLM, an expert AI assistant specialized in advertising technology.

You help with:
- Creating comprehensive media plans across CTV, Linear TV, Digital, and Social channels
- Writing compelling ad creative (headlines, copy, CTAs)
- Predicting ad performance and CTR
- Detecting ad fraud and invalid traffic
- Analyzing content for brand safety
- Optimizing campaign budgets and bidding strategies

Always provide detailed, actionable responses. When given a budget, calculate specific allocations. When asked for creative, provide multiple options."""

    # Format as Llama 3.1 Instruct
    full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the assistant's response
    if "assistant" in response.lower():
        response = response.split("assistant")[-1].strip()

    return response


def run_tests(model, tokenizer):
    """Run diagnostic tests on the model."""

    tests = [
        {
            "name": "Test 1: Basic Math (Reasoning Check)",
            "prompt": "What is 2 + 2?",
            "expected_contains": ["4"],
        },
        {
            "name": "Test 2: General Knowledge (Catastrophic Forgetting Check)",
            "prompt": "What is programmatic advertising?",
            "expected_contains": ["automated", "digital", "buying", "real-time"],
            "should_not_contain": ["CTR prediction", "AdTech AI"],
        },
        {
            "name": "Test 3: Creative Generation",
            "prompt": "Write 3 headlines for a Pepsi summer campaign targeting Gen Z.",
            "expected_contains": ["Pepsi", "summer"],
        },
        {
            "name": "Test 4: Media Planning",
            "prompt": "Build a full media plan for Pepsi for Q2. $5M budget. Include CTV, Linear TV, and digital.",
            "expected_contains": ["$", "CTV", "budget", "channel"],
            "should_not_contain": ["$1000", "Tech_enthusiasts", "Referral"],
        },
        {
            "name": "Test 5: Fraud Detection",
            "prompt": "Analyze this traffic: IP from AWS data center, 500 clicks/minute, bot user agent.",
            "expected_contains": ["fraud", "bot", "block"],
        },
        {
            "name": "Test 6: Brand Safety",
            "prompt": "Is an article about new iPhone releases brand safe for advertising?",
            "expected_contains": ["safe", "technology"],
        },
    ]

    print("\n" + "=" * 60)
    print("RUNNING DIAGNOSTIC TESTS")
    print("=" * 60)

    results = []

    for test in tests:
        print(f"\n{'-' * 50}")
        print(f"{test['name']}")
        print(f"{'-' * 50}")
        print(f"Prompt: {test['prompt'][:100]}...")

        response = generate_response(model, tokenizer, test['prompt'])

        print(f"\nResponse:\n{response[:500]}...")

        # Check expected content
        passed = True
        response_lower = response.lower()

        for expected in test.get('expected_contains', []):
            if expected.lower() not in response_lower:
                print(f"\n  WARNING: Expected '{expected}' not found in response")
                passed = False

        for forbidden in test.get('should_not_contain', []):
            if forbidden.lower() in response_lower:
                print(f"\n  WARNING: Found forbidden term '{forbidden}' in response")
                passed = False

        status = "PASS" if passed else "NEEDS REVIEW"
        print(f"\n  Status: {status}")
        results.append({"test": test['name'], "passed": passed})

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed_count = sum(1 for r in results if r['passed'])
    total_count = len(results)

    for r in results:
        status = "PASS" if r['passed'] else "FAIL"
        print(f"  [{status}] {r['test']}")

    print(f"\n  Total: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n  Model is working correctly!")
    else:
        print("\n  Some tests need review. Check responses above.")

    return results


def interactive_mode(model, tokenizer):
    """Run interactive chat mode."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("Type 'quit' to exit")
    print("=" * 60)

    while True:
        try:
            prompt = input("\nYou: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            if not prompt:
                continue

            print("\nAdTech LLM: ", end="", flush=True)
            response = generate_response(model, tokenizer, prompt)
            print(response)

        except KeyboardInterrupt:
            break

    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description="Test AdTech LLM V2")
    parser.add_argument("--model", required=True, help="Path to fine-tuned model")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run interactive mode")
    parser.add_argument("--skip-tests", action="store_true", help="Skip diagnostic tests")

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.model, args.base_model)

    # Run tests
    if not args.skip_tests:
        run_tests(model, tokenizer)

    # Interactive mode
    if args.interactive:
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
