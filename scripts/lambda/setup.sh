#!/bin/bash
# Lambda Labs Setup Script for AdTech LLM Training
# Run this on a fresh Lambda Labs GPU instance

set -e

echo "=========================================="
echo "AdTech LLM - Lambda Labs Setup"
echo "=========================================="

# Update system
sudo apt-get update -qq

# Install required packages
sudo apt-get install -y -qq git wget unzip

# Create working directory
mkdir -p ~/adtech-llm
cd ~/adtech-llm

# Clone the repository (or copy files)
if [ ! -d "AdLLM" ]; then
    echo "Cloning AdTech LLM repository..."
    git clone https://github.com/tonyvone/AdLLM.git || {
        echo "Creating project structure manually..."
        mkdir -p AdLLM
    }
fi

cd AdLLM || mkdir -p AdLLM && cd AdLLM

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install training dependencies
pip install transformers>=4.36.0
pip install peft>=0.7.0
pip install datasets>=2.16.0
pip install accelerate>=0.25.0
pip install bitsandbytes>=0.41.0
pip install trl>=0.7.0
pip install scipy
pip install sentencepiece
pip install protobuf
pip install kaggle
pip install pandas numpy scikit-learn
pip install wandb  # Optional: for experiment tracking

# Set up Kaggle credentials
mkdir -p ~/.kaggle
echo "Setting up Kaggle credentials..."
# You'll need to provide your kaggle.json or set environment variables

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Set up Kaggle credentials: ~/.kaggle/kaggle.json"
echo "2. Run: source venv/bin/activate"
echo "3. Run: python scripts/lambda/download_data.py"
echo "4. Run: python scripts/lambda/train_model.py"
echo ""
