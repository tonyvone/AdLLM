# AdTech LLM

> Domain-Specific AI Model for Advertising Technology

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AdTech LLM is the world's first domain-specific large language model tailored exclusively for advertising technology. It leverages advanced NLP, ML, and agentic architectures to automate and optimize key adtech workflows including ad creative generation, audience targeting, real-time bidding optimization, fraud detection, and contextual analysis.

## Features

- **Ad Creative Generation**: AI-powered generation of ad headlines, descriptions, and CTAs with multi-variant A/B testing support
- **Audience Targeting**: Intelligent audience segmentation and CTR/conversion prediction
- **Real-Time Bidding (RTB)**: AI-driven bid optimization for programmatic advertising
- **Fraud Detection**: Multi-layered fraud analysis including bot detection, click fraud, and invalid traffic
- **Contextual Analysis**: Brand safety scoring, topic extraction, and content suitability analysis
- **Agentic Workflows**: Autonomous AI agents for end-to-end campaign optimization

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 15+
- Redis 7+
- CUDA-capable GPU (optional, for local model inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/adtech-llm/adtech-llm.git
cd adtech-llm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment configuration
cp .env.example .env
# Edit .env with your settings

# Start the API server
adtech-llm serve
```

### Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f api
```

## Architecture

```
AdTech LLM
├── src/
│   ├── agents/           # Agentic system components
│   │   ├── base.py       # Base agent class
│   │   ├── data_ingestion.py
│   │   ├── fine_tuning.py
│   │   └── orchestrator.py
│   ├── modules/          # Core functionality modules
│   │   ├── creative_generation.py
│   │   ├── audience_targeting.py
│   │   ├── fraud_detection.py
│   │   ├── contextual_analysis.py
│   │   ├── ctr_prediction.py
│   │   └── bid_optimization.py
│   ├── api/              # FastAPI REST API
│   │   ├── app.py
│   │   └── routes/
│   ├── data/             # Data models and schemas
│   │   ├── models.py     # SQLAlchemy ORM models
│   │   └── schemas.py    # Pydantic schemas
│   ├── config/           # Configuration management
│   └── utils/            # Utilities and helpers
├── tests/                # Test suite
├── data/                 # Data storage
└── docs/                 # Documentation
```

## API Endpoints

### Ad Creative Generation

```bash
# Generate ad creatives
POST /api/v1/ads/generate
{
  "product_name": "FitLife App",
  "product_description": "Smart fitness tracking app",
  "num_variants": 5,
  "tone": "motivational"
}
```

### Audience Targeting

```bash
# Get targeting recommendations
POST /api/v1/audiences/recommendations
{
  "product_category": "fitness",
  "campaign_objective": "conversions",
  "budget": 5000
}

# Predict CTR
POST /api/v1/audiences/predict-ctr
{
  "creative_features": {"format": "video"},
  "audience_features": {"device_type": "mobile"},
  "context_features": {"hour_of_day": 20}
}
```

### Fraud Detection

```bash
# Analyze for fraud
POST /api/v1/fraud/analyze
{
  "event_type": "click",
  "user_agent": "Mozilla/5.0...",
  "check_types": ["bot_detection", "click_fraud"]
}
```

### Bid Optimization

```bash
# Optimize bid
POST /api/v1/bids/optimize
{
  "campaign_id": "uuid",
  "ad_slot": {"size": {"width": 300, "height": 250}},
  "strategy": "target_cpa",
  "target_cpa": 10.0
}
```

### Contextual Analysis

```bash
# Analyze content
POST /api/v1/contextual/analyze
{
  "url": "https://example.com/article",
  "brand_safety_level": "standard"
}
```

## CLI Commands

```bash
# Start API server
adtech-llm serve --host 0.0.0.0 --port 8000

# Generate ad creatives
adtech-llm generate "My Product" --variants 5 --tone casual

# Discover and ingest datasets
adtech-llm ingest --source kaggle --query "adtech CTR"

# Train/fine-tune model
adtech-llm train ./data/processed/training_sft --epochs 3

# Run workflow
adtech-llm workflow training

# Show system info
adtech-llm info
```

## Configuration

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_MODEL_NAME` | Base LLM model | `meta-llama/Llama-3.1-8B-Instruct` |
| `LLM_QUANTIZATION` | Quantization mode | `4bit` |
| `DB_HOST` | PostgreSQL host | `localhost` |
| `REDIS_HOST` | Redis host | `localhost` |
| `SECURITY_SECRET_KEY` | JWT secret key | - |

See `.env.example` for full configuration options.

## Agentic System

AdTech LLM uses an agentic architecture with specialized agents:

- **Data Ingestion Agent**: Discovers, downloads, and preprocesses datasets from Kaggle, Hugging Face, and partner APIs
- **Fine-Tuning Agent**: Handles model training with LoRA/PEFT for efficient adaptation
- **Planner Agent**: Decomposes complex tasks into executable workflows
- **Orchestrator**: Coordinates multiple agents in graph-based workflows

```python
from src.agents.orchestrator import AgentOrchestrator

# Create and run a training workflow
orchestrator = AgentOrchestrator()
orchestrator.create_training_workflow()
state = await orchestrator.run(start_node="discover_data")
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_creative_generation.py
```

### Code Quality

```bash
# Format code
black src tests
isort src tests

# Lint
ruff check src tests

# Type check
mypy src
```

## Data Sources

AdTech LLM supports multiple data sources:

### Public Datasets
- **Kaggle**: Avazu CTR, Criteo Display Ads, Advertising datasets
- **Hugging Face**: AdImageNet, MS MARCO
- **Criteo AI Lab**: Terabyte Click Logs, Uplift Modeling

### Partner Integration
- Infolinks, PadSquad, The Trade Desk integrations
- Clean room data sharing support
- First-party data onboarding

## Performance Targets

| Metric | Target |
|--------|--------|
| API Latency | < 500ms |
| CTR Prediction Accuracy | > 85% AUC |
| Fraud Detection Accuracy | > 95% |
| Throughput | 10K+ requests/hour |
| Uptime | 99.9% |

## Roadmap

- **Q1 2026**: MVP with core modules
- **Q2 2026**: Full feature release with platform integrations
- **Q3 2026**: Advanced RLHF training and multi-modal support
- **Q4 2026**: Enterprise features and custom model training

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to the `develop` branch.

## Support

- Documentation: [docs.adtechllm.ai](https://docs.adtechllm.ai)
- Issues: [GitHub Issues](https://github.com/adtech-llm/adtech-llm/issues)
- Email: support@adtechllm.ai
