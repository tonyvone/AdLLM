#!/usr/bin/env python3
"""
AdTech LLM - Data Download Script for Lambda Labs
Downloads and prepares adtech datasets for fine-tuning.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np

# Configuration
DATA_DIR = Path("./data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Kaggle datasets to download
KAGGLE_DATASETS = [
    "avazu/avazu-ctr-prediction",  # Large CTR dataset
    "fayomi/advertising",  # Advertising metrics
    "lsjsj/advertising-campaign-dataset",  # Campaign data
]

def setup_directories():
    """Create necessary directories."""
    for dir_path in [RAW_DIR, PROCESSED_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directories: {DATA_DIR}")

def check_kaggle_credentials():
    """Check if Kaggle credentials are set up."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        # Try environment variable
        if os.environ.get("KAGGLE_API_TOKEN"):
            kaggle_json.parent.mkdir(exist_ok=True)
            with open(kaggle_json, "w") as f:
                json.dump({"key": os.environ["KAGGLE_API_TOKEN"]}, f)
            os.chmod(kaggle_json, 0o600)
            print("✓ Created kaggle.json from environment variable")
            return True
        print("✗ Kaggle credentials not found!")
        print("  Set KAGGLE_API_TOKEN environment variable or create ~/.kaggle/kaggle.json")
        return False
    print("✓ Kaggle credentials found")
    return True

def download_kaggle_datasets():
    """Download datasets from Kaggle."""
    print("\n📥 Downloading Kaggle datasets...")

    for dataset in KAGGLE_DATASETS:
        dataset_name = dataset.split("/")[-1]
        output_dir = RAW_DIR / "kaggle" / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Downloading: {dataset}")
        try:
            result = subprocess.run(
                ["kaggle", "datasets", "download", "-d", dataset, "-p", str(output_dir), "--unzip"],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0:
                print(f"  ✓ Downloaded: {dataset}")
            else:
                print(f"  ✗ Failed: {dataset}")
                print(f"    Error: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print(f"  ✗ Timeout downloading: {dataset}")
        except FileNotFoundError:
            print("  ✗ Kaggle CLI not found. Install with: pip install kaggle")
            break

def create_synthetic_adtech_data():
    """Create synthetic adtech data if downloads fail."""
    print("\n🔧 Creating synthetic adtech training data...")

    np.random.seed(42)
    n_samples = 100000

    # CTR prediction data
    ctr_data = {
        "impression_id": [f"imp_{i}" for i in range(n_samples)],
        "campaign_id": np.random.randint(1, 100, n_samples),
        "creative_id": np.random.randint(1, 500, n_samples),
        "advertiser_id": np.random.randint(1, 50, n_samples),
        "site_id": np.random.randint(1, 1000, n_samples),
        "device_type": np.random.choice(["mobile", "desktop", "tablet"], n_samples, p=[0.6, 0.3, 0.1]),
        "os": np.random.choice(["ios", "android", "windows", "macos", "linux"], n_samples, p=[0.3, 0.35, 0.25, 0.08, 0.02]),
        "browser": np.random.choice(["chrome", "safari", "firefox", "edge", "other"], n_samples, p=[0.5, 0.25, 0.1, 0.1, 0.05]),
        "country": np.random.choice(["US", "UK", "DE", "FR", "CA", "AU", "JP", "BR"], n_samples, p=[0.4, 0.15, 0.1, 0.08, 0.08, 0.07, 0.07, 0.05]),
        "hour": np.random.randint(0, 24, n_samples),
        "day_of_week": np.random.randint(0, 7, n_samples),
        "ad_position": np.random.choice(["above_fold", "below_fold", "sidebar"], n_samples, p=[0.4, 0.35, 0.25]),
        "ad_format": np.random.choice(["banner", "native", "video", "interstitial"], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        "bid_price": np.random.exponential(1.0, n_samples).round(4),
        "click": np.random.binomial(1, 0.025, n_samples),  # ~2.5% CTR
    }

    # Add conversion for clicked ads
    ctr_data["conversion"] = [
        np.random.binomial(1, 0.15) if click == 1 else 0
        for click in ctr_data["click"]
    ]

    df_ctr = pd.DataFrame(ctr_data)

    # Save CTR data
    ctr_path = PROCESSED_DIR / "ctr_training_data.parquet"
    df_ctr.to_parquet(ctr_path, index=False)
    print(f"  ✓ Created CTR data: {ctr_path} ({len(df_ctr):,} rows)")

    # Ad creative data for text generation
    products = [
        ("FitLife Pro", "Smart fitness tracker with heart rate monitoring and sleep analysis"),
        ("CloudSync", "Enterprise file synchronization and collaboration platform"),
        ("EcoClean", "Eco-friendly household cleaning products made from natural ingredients"),
        ("TechGear Plus", "Premium electronics and gadgets for the modern home"),
        ("StyleBox", "Personalized fashion subscription service curated by stylists"),
        ("FreshMeals", "Healthy meal delivery service with chef-prepared dishes"),
        ("LearnHub", "Online learning platform with expert-led courses"),
        ("TravelEase", "AI-powered travel planning and booking assistant"),
        ("PetCare Pro", "Premium pet food and supplies delivered to your door"),
        ("HomeSecure", "Smart home security system with 24/7 monitoring"),
    ]

    tones = ["professional", "casual", "urgent", "friendly", "luxurious"]
    audiences = ["young_professionals", "families", "tech_enthusiasts", "budget_conscious", "luxury_seekers"]

    ad_creative_data = []
    for _ in range(10000):
        product_name, product_desc = products[np.random.randint(0, len(products))]
        tone = np.random.choice(tones)
        audience = np.random.choice(audiences)

        # Generate instruction-output pairs
        instruction = f"Generate a compelling ad headline for {product_name}."
        input_text = f"Product: {product_name}\nDescription: {product_desc}\nTarget Audience: {audience}\nTone: {tone}"

        # Template-based outputs (in real training, these would be human-written examples)
        headlines = {
            "professional": f"Discover {product_name} - Professional Solutions for Modern Challenges",
            "casual": f"Hey! Check out {product_name} - You'll Love It!",
            "urgent": f"Don't Miss Out! {product_name} - Limited Time Offer",
            "friendly": f"Meet {product_name} - Your New Favorite Thing",
            "luxurious": f"Experience Excellence with {product_name} - Premium Quality",
        }
        output = headlines[tone]

        ad_creative_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output,
            "product_name": product_name,
            "tone": tone,
            "audience": audience,
        })

    df_creative = pd.DataFrame(ad_creative_data)
    creative_path = PROCESSED_DIR / "ad_creative_training_data.parquet"
    df_creative.to_parquet(creative_path, index=False)
    print(f"  ✓ Created ad creative data: {creative_path} ({len(df_creative):,} rows)")

    # CTR prediction instruction data
    ctr_instruction_data = []
    for _, row in df_ctr.sample(min(20000, len(df_ctr))).iterrows():
        instruction = "Predict whether this ad impression will result in a click based on the features."
        input_text = f"""Campaign ID: {row['campaign_id']}
Device: {row['device_type']}
OS: {row['os']}
Country: {row['country']}
Hour: {row['hour']}
Day: {row['day_of_week']}
Position: {row['ad_position']}
Format: {row['ad_format']}"""
        output = "Click: Yes" if row['click'] == 1 else "Click: No"

        ctr_instruction_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output,
        })

    df_ctr_inst = pd.DataFrame(ctr_instruction_data)
    ctr_inst_path = PROCESSED_DIR / "ctr_instruction_data.parquet"
    df_ctr_inst.to_parquet(ctr_inst_path, index=False)
    print(f"  ✓ Created CTR instruction data: {ctr_inst_path} ({len(df_ctr_inst):,} rows)")

    return True

def prepare_training_dataset():
    """Combine all data into a unified training format."""
    print("\n📦 Preparing unified training dataset...")

    all_data = []

    # Load ad creative data
    creative_path = PROCESSED_DIR / "ad_creative_training_data.parquet"
    if creative_path.exists():
        df = pd.read_parquet(creative_path)
        for _, row in df.iterrows():
            all_data.append({
                "instruction": row["instruction"],
                "input": row["input"],
                "output": row["output"],
                "task_type": "ad_generation",
            })

    # Load CTR instruction data
    ctr_path = PROCESSED_DIR / "ctr_instruction_data.parquet"
    if ctr_path.exists():
        df = pd.read_parquet(ctr_path)
        for _, row in df.iterrows():
            all_data.append({
                "instruction": row["instruction"],
                "input": row["input"],
                "output": row["output"],
                "task_type": "ctr_prediction",
            })

    # Add more task types
    # Fraud detection examples
    fraud_examples = [
        {
            "instruction": "Analyze this traffic for potential fraud indicators.",
            "input": "User Agent: Mozilla/5.0 (compatible; Googlebot/2.1)\nClick Rate: 500/minute\nIP Type: Datacenter",
            "output": "High fraud risk detected. Indicators: Bot user agent pattern, abnormally high click rate, datacenter IP. Recommendation: Block traffic.",
            "task_type": "fraud_detection",
        },
        {
            "instruction": "Analyze this traffic for potential fraud indicators.",
            "input": "User Agent: Mozilla/5.0 (iPhone; CPU iPhone OS 15_0)\nClick Rate: 2/hour\nIP Type: Residential",
            "output": "Low fraud risk. Traffic appears legitimate with normal user agent, reasonable click rate, and residential IP. Recommendation: Allow traffic.",
            "task_type": "fraud_detection",
        },
    ] * 1000  # Replicate for training volume
    all_data.extend(fraud_examples)

    # Contextual analysis examples
    context_examples = [
        {
            "instruction": "Analyze this webpage content for ad placement suitability.",
            "input": "Page Content: Technology news article about new smartphone releases and mobile app reviews.",
            "output": "Brand Safe: Yes\nTopics: Technology, Mobile, Consumer Electronics\nSuitable Ad Categories: Tech, Electronics, Mobile Apps, Gadgets\nSentiment: Neutral/Positive",
            "task_type": "contextual_analysis",
        },
        {
            "instruction": "Analyze this webpage content for ad placement suitability.",
            "input": "Page Content: Breaking news about violent incident in downtown area.",
            "output": "Brand Safe: No\nTopics: News, Violence, Crime\nSuitable Ad Categories: None recommended\nSentiment: Negative\nWarning: Content may not be suitable for most advertisers.",
            "task_type": "contextual_analysis",
        },
    ] * 500
    all_data.extend(context_examples)

    # Shuffle and save
    np.random.shuffle(all_data)
    df_final = pd.DataFrame(all_data)

    # Split into train/validation
    train_size = int(0.9 * len(df_final))
    df_train = df_final[:train_size]
    df_val = df_final[train_size:]

    train_path = PROCESSED_DIR / "train.parquet"
    val_path = PROCESSED_DIR / "validation.parquet"

    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)

    print(f"  ✓ Training set: {train_path} ({len(df_train):,} examples)")
    print(f"  ✓ Validation set: {val_path} ({len(df_val):,} examples)")

    # Also save as JSONL for easier loading
    train_jsonl = PROCESSED_DIR / "train.jsonl"
    val_jsonl = PROCESSED_DIR / "validation.jsonl"

    df_train.to_json(train_jsonl, orient="records", lines=True)
    df_val.to_json(val_jsonl, orient="records", lines=True)

    print(f"  ✓ JSONL files created")

    return len(df_train), len(df_val)

def main():
    print("=" * 50)
    print("AdTech LLM - Data Download & Preparation")
    print("=" * 50)

    setup_directories()

    # Try Kaggle first
    if check_kaggle_credentials():
        download_kaggle_datasets()

    # Create synthetic data (always, as backup/supplement)
    create_synthetic_adtech_data()

    # Prepare unified training dataset
    train_size, val_size = prepare_training_dataset()

    print("\n" + "=" * 50)
    print("✓ Data preparation complete!")
    print(f"  Training examples: {train_size:,}")
    print(f"  Validation examples: {val_size:,}")
    print("=" * 50)
    print("\nNext: Run python scripts/lambda/train_model.py")

if __name__ == "__main__":
    main()
