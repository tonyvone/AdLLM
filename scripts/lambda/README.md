# AdTech LLM - Lambda Labs Deployment Guide

This guide walks you through fine-tuning the AdTech LLM on Lambda Labs GPU cloud.

## Quick Start (5 minutes to training)

### 1. Create Lambda Labs Instance

1. Go to [Lambda Labs Cloud](https://cloud.lambdalabs.com/)
2. Create account and add payment method
3. Launch an instance:
   - **Recommended**: 1x A100 40GB ($1.29/hr) - sufficient for LoRA fine-tuning
   - **Alternative**: 1x A100 80GB ($1.79/hr) - for larger batch sizes
4. SSH into your instance:
   ```bash
   ssh ubuntu@<your-instance-ip>
   ```

### 2. Setup Environment

```bash
# Clone repository
git clone https://github.com/tonyvone/AdLLM.git
cd AdLLM

# Run setup script
chmod +x scripts/lambda/setup.sh
./scripts/lambda/setup.sh

# Activate virtual environment
source venv/bin/activate
```

### 3. Configure Kaggle (Optional)

If you have Kaggle credentials for additional datasets:

```bash
# Set your Kaggle API token
export KAGGLE_API_TOKEN="KGAT_your_token_here"

# Or create credentials file
mkdir -p ~/.kaggle
echo '{"key":"KGAT_your_token_here"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Download/Generate Training Data

```bash
python scripts/lambda/download_data.py
```

This will:
- Try to download real adtech datasets from Kaggle
- Generate synthetic training data as backup
- Create instruction-formatted data for fine-tuning

### 5. Start Training

```bash
python scripts/lambda/train_model.py
```

Training takes approximately **2-4 hours** on A100 40GB.

## Training Options

```bash
# Custom configuration
python scripts/lambda/train_model.py \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-4 \
    --lora-r 16

# Faster training (larger batch)
python scripts/lambda/train_model.py --batch-size 8 --epochs 2

# Higher quality (more epochs)
python scripts/lambda/train_model.py --epochs 5 --learning-rate 1e-4
```

## Cost Estimate

| GPU | Price/hr | Est. Training Time | Est. Total Cost |
|-----|----------|-------------------|-----------------|
| A100 40GB | $1.29 | 3-4 hours | $4-5 |
| A100 80GB | $1.79 | 2-3 hours | $4-6 |
| H100 80GB | $2.99 | 1-2 hours | $3-6 |

## After Training

### Download Your Model

```bash
# On Lambda instance - compress model
cd models/adtech-llm/run_*/final_model
tar -czvf ~/adtech-llm-model.tar.gz .

# On your local machine - download
scp ubuntu@<instance-ip>:~/adtech-llm-model.tar.gz .
```

### Use the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16,
)

# Load fine-tuned LoRA adapter
model = PeftModel.from_pretrained(base_model, "./adtech-llm-model")
tokenizer = AutoTokenizer.from_pretrained("./adtech-llm-model")

# Generate ad creative
prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are AdTech LLM, an AI assistant specialized in advertising technology.<|eot_id|><|start_header_id|>user<|end_header_id|>

Generate a compelling ad headline for CloudSync.

Product: CloudSync
Description: Enterprise file synchronization and collaboration platform
Target Audience: business_professionals
Tone: professional<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Troubleshooting

### Out of Memory
- Reduce `--batch-size` to 2
- Add `--gradient-accumulation-steps 8`

### Slow Training
- Ensure Flash Attention 2 is installed
- Check GPU utilization: `nvidia-smi -l 1`

### Model Access
If you get authentication errors for Llama:
1. Accept license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Login: `huggingface-cli login`

## Files

```
scripts/lambda/
├── setup.sh           # Environment setup script
├── download_data.py   # Data download and preparation
├── train_model.py     # Main training script
└── README.md          # This guide
```
