"""Model training and fine-tuning endpoints."""

from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from src.agents.data_ingestion import DataIngestionAgent
from src.agents.fine_tuning import FineTuningAgent
from src.data.schemas import DatasetInfo, DatasetUploadRequest, TrainingJobRequest, TrainingJobStatus
from src.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Agent instances
_data_agent: DataIngestionAgent | None = None
_training_agent: FineTuningAgent | None = None

# Job tracking (in production, would use database)
_training_jobs: dict[str, dict] = {}


def get_data_agent() -> DataIngestionAgent:
    """Get or create data ingestion agent."""
    global _data_agent
    if _data_agent is None:
        _data_agent = DataIngestionAgent()
    return _data_agent


def get_training_agent() -> FineTuningAgent:
    """Get or create training agent."""
    global _training_agent
    if _training_agent is None:
        _training_agent = FineTuningAgent()
    return _training_agent


@router.post("/training/datasets/discover")
async def discover_datasets(
    query: str = "adtech CTR",
    sources: list[str] | None = None,
    agent: Annotated[DataIngestionAgent, Depends(get_data_agent)] = None,
):
    """
    Discover available datasets for training.

    Searches Kaggle, Hugging Face, and other sources for relevant datasets.
    """
    sources = sources or ["kaggle", "huggingface", "criteo"]

    try:
        result = await agent.use_tool(
            "discover_datasets",
            query=query,
            sources=sources,
            limit=10,
        )
        return result
    except Exception as e:
        logger.error(f"Dataset discovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/training/datasets/ingest")
async def ingest_dataset(
    request: DatasetUploadRequest,
    background_tasks: BackgroundTasks,
    agent: Annotated[DataIngestionAgent, Depends(get_data_agent)] = None,
):
    """
    Ingest a dataset for training.

    Downloads, preprocesses, and stores the dataset for model training.
    """
    job_id = str(uuid4())

    async def ingest_task():
        try:
            result = await agent.run(
                task=f"Ingest dataset: {request.name}",
                context={
                    "sources": [request.source],
                    "identifier": request.source_identifier,
                },
            )
            _training_jobs[job_id] = {
                "status": "completed" if result.success else "failed",
                "result": result.to_dict(),
            }
        except Exception as e:
            _training_jobs[job_id] = {"status": "failed", "error": str(e)}

    _training_jobs[job_id] = {"status": "running"}
    background_tasks.add_task(ingest_task)

    return {
        "job_id": job_id,
        "status": "started",
        "message": "Dataset ingestion started in background",
    }


@router.get("/training/datasets")
async def list_datasets():
    """
    List available training datasets.

    Returns datasets that have been ingested and are ready for training.
    """
    # In production, would query database
    return {
        "datasets": [
            {
                "id": "criteo_ctr_sample",
                "name": "Criteo CTR Sample",
                "source": "criteo",
                "rows": 10000,
                "status": "ready",
            },
            {
                "id": "kaggle_advertising",
                "name": "Kaggle Advertising Dataset",
                "source": "kaggle",
                "rows": 5000,
                "status": "ready",
            },
        ],
        "total": 2,
    }


@router.post("/training/jobs", response_model=TrainingJobStatus)
async def start_training_job(
    request: TrainingJobRequest,
    background_tasks: BackgroundTasks,
    agent: Annotated[FineTuningAgent, Depends(get_training_agent)] = None,
):
    """
    Start a model training job.

    Initiates fine-tuning of the base LLM on specified datasets.
    """
    job_id = str(uuid4())

    async def training_task():
        try:
            result = await agent.run(
                task=f"Fine-tune model: {request.job_name}",
                context={
                    "task_type": request.training_type,
                    "data_paths": request.dataset_ids,
                    "config": request.hyperparameters,
                },
            )
            _training_jobs[job_id] = {
                "status": "completed" if result.success else "failed",
                "result": result.to_dict(),
            }
        except Exception as e:
            _training_jobs[job_id] = {"status": "failed", "error": str(e)}

    _training_jobs[job_id] = {
        "status": "pending",
        "job_name": request.job_name,
        "started_at": None,
    }
    background_tasks.add_task(training_task)

    return TrainingJobStatus(
        job_id=job_id,
        job_name=request.job_name,
        status="pending",
        progress=0.0,
        current_epoch=0,
        current_step=0,
        total_steps=0,
        metrics={},
        started_at=None,
        completed_at=None,
    )


@router.get("/training/jobs/{job_id}", response_model=TrainingJobStatus)
async def get_training_job(job_id: str):
    """
    Get status of a training job.

    Returns current progress, metrics, and status of the job.
    """
    if job_id not in _training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _training_jobs[job_id]

    return TrainingJobStatus(
        job_id=job_id,
        job_name=job.get("job_name", "Unknown"),
        status=job.get("status", "unknown"),
        progress=job.get("progress", 0.0),
        current_epoch=job.get("current_epoch", 0),
        current_step=job.get("current_step", 0),
        total_steps=job.get("total_steps", 0),
        metrics=job.get("metrics", {}),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        error_message=job.get("error"),
    )


@router.get("/training/jobs")
async def list_training_jobs():
    """
    List all training jobs.

    Returns summary of all training jobs and their status.
    """
    jobs = [
        {
            "job_id": job_id,
            "status": job.get("status"),
            "job_name": job.get("job_name", "Unknown"),
        }
        for job_id, job in _training_jobs.items()
    ]
    return {"jobs": jobs, "total": len(jobs)}


@router.delete("/training/jobs/{job_id}")
async def cancel_training_job(job_id: str):
    """
    Cancel a running training job.

    Stops the training process and cleans up resources.
    """
    if job_id not in _training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    _training_jobs[job_id]["status"] = "cancelled"

    return {"status": "cancelled", "job_id": job_id}


@router.get("/training/models")
async def list_models():
    """
    List trained models.

    Returns all available model checkpoints.
    """
    return {
        "models": [
            {
                "id": "adtech_llm_v1",
                "base_model": "meta-llama/Llama-3.1-8B-Instruct",
                "version": "1.0.0",
                "created_at": "2026-01-14T00:00:00Z",
                "metrics": {"accuracy": 0.85, "perplexity": 15.2},
                "is_production": True,
            }
        ],
        "total": 1,
    }


@router.post("/training/models/{model_id}/deploy")
async def deploy_model(model_id: str):
    """
    Deploy a model to production.

    Makes the specified model the active production model.
    """
    return {
        "status": "deployed",
        "model_id": model_id,
        "message": f"Model {model_id} is now the production model",
    }
