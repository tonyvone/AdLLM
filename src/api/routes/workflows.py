"""Workflow orchestration endpoints."""

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks

from src.agents.orchestrator import AgentOrchestrator
from src.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Workflow tracking
_workflows: dict[str, dict[str, Any]] = {}
_orchestrators: dict[str, AgentOrchestrator] = {}


@router.post("/workflows/start")
async def start_workflow(
    workflow_type: str,
    initial_context: dict[str, Any] | None = None,
    background_tasks: BackgroundTasks = None,
):
    """
    Start a new workflow.

    Initiates an orchestrated workflow based on the specified type.

    Available workflow types:
    - training: Full model training pipeline
    - campaign: Campaign optimization workflow
    - custom: Custom workflow with provided configuration
    """
    workflow_id = str(uuid4())

    orchestrator = AgentOrchestrator()

    if workflow_type == "training":
        orchestrator.create_training_workflow()
        start_node = "discover_data"
    elif workflow_type == "campaign":
        orchestrator.create_campaign_workflow()
        start_node = "analyze_goals"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown workflow type: {workflow_type}",
        )

    _orchestrators[workflow_id] = orchestrator
    _workflows[workflow_id] = {
        "type": workflow_type,
        "status": "pending",
    }

    async def run_workflow():
        try:
            state = await orchestrator.run(
                start_node=start_node,
                initial_context=initial_context or {},
            )
            _workflows[workflow_id] = {
                "type": workflow_type,
                "status": state.status.value,
                "result": state.to_dict(),
            }
        except Exception as e:
            _workflows[workflow_id] = {
                "type": workflow_type,
                "status": "failed",
                "error": str(e),
            }

    background_tasks.add_task(run_workflow)

    return {
        "workflow_id": workflow_id,
        "type": workflow_type,
        "status": "started",
        "message": "Workflow started in background",
    }


@router.get("/workflows/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """
    Get status of a workflow.

    Returns current state, progress, and outputs of the workflow.
    """
    if workflow_id not in _workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")

    workflow = _workflows[workflow_id]

    # Get live status from orchestrator if available
    if workflow_id in _orchestrators:
        orchestrator = _orchestrators[workflow_id]
        live_status = orchestrator.get_workflow_status()
        return {
            "workflow_id": workflow_id,
            **live_status,
        }

    return {
        "workflow_id": workflow_id,
        **workflow,
    }


@router.get("/workflows")
async def list_workflows():
    """
    List all workflows.

    Returns summary of all workflows and their status.
    """
    workflows = [
        {
            "workflow_id": wf_id,
            "type": wf.get("type"),
            "status": wf.get("status"),
        }
        for wf_id, wf in _workflows.items()
    ]
    return {"workflows": workflows, "total": len(workflows)}


@router.post("/workflows/{workflow_id}/pause")
async def pause_workflow(workflow_id: str):
    """
    Pause a running workflow.

    Suspends execution at the current step.
    """
    if workflow_id not in _orchestrators:
        raise HTTPException(status_code=404, detail="Workflow not found")

    orchestrator = _orchestrators[workflow_id]
    await orchestrator.pause()

    return {
        "workflow_id": workflow_id,
        "status": "paused",
    }


@router.post("/workflows/{workflow_id}/resume")
async def resume_workflow(
    workflow_id: str,
    background_tasks: BackgroundTasks,
):
    """
    Resume a paused workflow.

    Continues execution from the paused step.
    """
    if workflow_id not in _orchestrators:
        raise HTTPException(status_code=404, detail="Workflow not found")

    orchestrator = _orchestrators[workflow_id]

    async def resume_task():
        try:
            state = await orchestrator.resume()
            _workflows[workflow_id] = {
                "type": _workflows[workflow_id].get("type"),
                "status": state.status.value,
                "result": state.to_dict(),
            }
        except Exception as e:
            _workflows[workflow_id]["status"] = "failed"
            _workflows[workflow_id]["error"] = str(e)

    background_tasks.add_task(resume_task)

    return {
        "workflow_id": workflow_id,
        "status": "resuming",
    }


@router.delete("/workflows/{workflow_id}")
async def cancel_workflow(workflow_id: str):
    """
    Cancel a workflow.

    Stops execution and cleans up resources.
    """
    if workflow_id not in _workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")

    _workflows[workflow_id]["status"] = "cancelled"

    if workflow_id in _orchestrators:
        del _orchestrators[workflow_id]

    return {
        "workflow_id": workflow_id,
        "status": "cancelled",
    }


@router.get("/workflows/types")
async def list_workflow_types():
    """
    List available workflow types.

    Returns descriptions of predefined workflow templates.
    """
    return {
        "types": [
            {
                "id": "training",
                "name": "Model Training Pipeline",
                "description": "Complete pipeline for data ingestion, preprocessing, and model fine-tuning",
                "steps": [
                    "Discover datasets",
                    "Ingest and preprocess data",
                    "Configure training",
                    "Execute fine-tuning",
                    "Evaluate model",
                ],
            },
            {
                "id": "campaign",
                "name": "Campaign Optimization",
                "description": "End-to-end campaign setup and optimization workflow",
                "steps": [
                    "Analyze campaign goals",
                    "Generate ad creatives",
                    "Define target audiences",
                    "Optimize bidding strategy",
                    "Monitor performance",
                ],
            },
        ],
    }


@router.get("/workflows/{workflow_id}/visualization")
async def visualize_workflow(workflow_id: str):
    """
    Get workflow visualization.

    Returns a text representation of the workflow graph.
    """
    if workflow_id not in _orchestrators:
        raise HTTPException(status_code=404, detail="Workflow not found")

    orchestrator = _orchestrators[workflow_id]
    visualization = orchestrator.visualize()

    return {
        "workflow_id": workflow_id,
        "visualization": visualization,
    }
