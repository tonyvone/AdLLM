# AdTech LLM V2 - Replit to Lambda Training Instructions

## Quick Start (Copy-Paste Commands)

### Step 1: SSH into Lambda from Replit
```bash
ssh ubuntu@YOUR_LAMBDA_IP
```

### Step 2: Clone/Update Repository
```bash
# If first time:
git clone https://github.com/tonyvone/AdLLM.git
cd AdLLM

# If already cloned, pull latest:
cd AdLLM
git pull origin claude/adtech-llm-model-gPlKP
```

### Step 3: Setup Environment (First Time Only)
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate peft trl bitsandbytes
pip install pandas numpy huggingface_hub
```

### Step 4: Activate Environment (Every Session)
```bash
cd AdLLM
source venv/bin/activate
```

### Step 5: Prepare Training Data (V2 - Fixed)
```bash
python scripts/lambda/download_data_v2.py
```

Expected output:
```
✓ Created 4,000 general instruction examples
✓ Created 8,000 media planning examples
✓ Created 8,000 creative generation examples
✓ Created 8,000 CTR prediction examples
✓ Created 8,000 fraud detection examples
✓ Created 8,000 contextual analysis examples

Dataset Distribution:
  general_instruction: 4,000 (9%)
  media_planning: 8,000 (18%)
  creative_generation: 8,000 (18%)
  ctr_prediction: 8,000 (18%)
  fraud_detection: 8,000 (18%)
  contextual_analysis: 8,000 (18%)

Total: ~44,000 examples
```

### Step 6: Start Training (V2 - Fixed Hyperparameters)
```bash
python scripts/lambda/train_model_v2.py
```

**V2 Fixes:**
- Learning rate: 5e-5 (was 2e-4) - 4x lower to preserve base knowledge
- Epochs: 1 (was 2-3) - prevents overfitting
- Balanced data distribution - no more CTR bias
- Added general instruction data - prevents catastrophic forgetting

**Expected Training Time:** ~2-3 hours on A100 40GB

### Step 7: Test the Model
```bash
python scripts/lambda/test_model_v2.py --model ./models/adtech-llm-v2/run_*/final_model
```

This runs 6 diagnostic tests:
1. Basic math (reasoning check)
2. General knowledge (catastrophic forgetting check)
3. Creative generation
4. Media planning
5. Fraud detection
6. Brand safety

### Step 8: Interactive Testing
```bash
python scripts/lambda/test_model_v2.py --model ./models/adtech-llm-v2/run_*/final_model --interactive --skip-tests
```

Try these prompts:
```
You: What is programmatic advertising?
You: Build a media plan for Nike with $2M budget for Q3
You: Write 3 headlines for a McDonald's breakfast campaign
You: Is this traffic fraudulent? 500 clicks/min from AWS datacenter
```

---

## Key Differences: V1 vs V2

| Issue in V1 | Fix in V2 |
|------------|-----------|
| "Programmatic advertising is an AdTech AI..." | Retains real definition |
| Output: "Type: Consideration, Channel: Email" | Outputs real media plans |
| Ignores user budget ($5M → $1000) | Respects and calculates from user budget |
| "Tech_enthusiasts" label leakage | Clean natural language outputs |
| Generic/irrelevant creative | Context-aware creative generation |

---

## Estimated Costs

| Phase | Time | Cost (A100 $1.29/hr) |
|-------|------|---------------------|
| Data Prep | 5 min | ~$0.10 |
| Training | 2-3 hrs | ~$3-4 |
| Testing | 15 min | ~$0.30 |
| **Total** | ~3 hrs | **~$4-5** |

---

## Troubleshooting

**Out of memory:**
```bash
# Reduce batch size
python scripts/lambda/train_model_v2.py --batch-size 2
```

**CUDA not available:**
```bash
# Check GPU
nvidia-smi

# Reinstall PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Model not generating good outputs:**
- Make sure you're using the V2 scripts (download_data_v2.py, train_model_v2.py)
- Run the test script to diagnose issues
- Check that training completed without errors

---

## After Training Succeeds

1. **Test thoroughly** with the test script
2. **Upload to HuggingFace** (optional):
```bash
huggingface-cli login
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='./models/adtech-llm-v2/run_*/final_model',
    repo_id='tonyvone/adtech-llm-v2',
    repo_type='model'
)
"
```

3. **Download adapter locally:**
```bash
# From your local machine
scp -r ubuntu@LAMBDA_IP:~/AdLLM/models/adtech-llm-v2/run_*/final_model ./adtech-llm-v2
```
