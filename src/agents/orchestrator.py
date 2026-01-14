"""
Agent Orchestrator for AdTech LLM.

Coordinates multiple AI agents to accomplish complex tasks
using LangGraph-style state management and workflow orchestration.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, TypeVar
from uuid import uuid4

from src.agents.base import AgentResult, AgentState, AgentStatus, BaseAgent
from src.agents.data_ingestion import DataIngestionAgent
from src.agents.fine_tuning import FineTuningAgent
from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class WorkflowStatus(str, Enum):
    """Workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkflowState:
    """
    State shared across all agents in a workflow.

    Attributes:
        workflow_id: Unique workflow identifier
        status: Current workflow status
        context: Shared context data
        agent_states: States of individual agents
        outputs: Collected outputs
        errors: Collected errors
    """

    workflow_id: str = field(default_factory=lambda: str(uuid4()))
    status: WorkflowStatus = WorkflowStatus.PENDING
    context: dict[str, Any] = field(default_factory=dict)
    agent_states: dict[str, AgentState] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    current_node: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "context": self.context,
            "agent_states": {
                k: v.to_dict() for k, v in self.agent_states.items()
            },
            "outputs": self.outputs,
            "errors": self.errors,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "current_node": self.current_node,
        }


@dataclass
class WorkflowNode:
    """
    A node in the workflow graph.

    Attributes:
        name: Node identifier
        agent: Agent to execute at this node
        condition: Optional condition for execution
        next_nodes: Possible next nodes
        on_success: Callback on successful execution
        on_failure: Callback on failed execution
    """

    name: str
    agent: BaseAgent | None = None
    task: str = ""
    condition: Callable[[WorkflowState], bool] | None = None
    next_nodes: list[str] = field(default_factory=list)
    on_success: Callable[[WorkflowState, AgentResult], None] | None = None
    on_failure: Callable[[WorkflowState, Exception], None] | None = None
    parallel: bool = False


class AgentOrchestrator:
    """
    Orchestrates multiple agents in complex workflows.

    Features:
    - Graph-based workflow execution
    - Conditional branching
    - Parallel execution
    - State management
    - Error recovery
    - Human-in-the-loop support
    """

    def __init__(self):
        """Initialize the orchestrator."""
        self.logger = get_logger("AgentOrchestrator")
        self.nodes: dict[str, WorkflowNode] = {}
        self.state: WorkflowState = WorkflowState()
        self._agents: dict[str, BaseAgent] = {}

        # Register built-in agents
        self._register_default_agents()

    def _register_default_agents(self) -> None:
        """Register default system agents."""
        self._agents["data_ingestion"] = DataIngestionAgent()
        self._agents["fine_tuning"] = FineTuningAgent()

    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """Register an agent for use in workflows."""
        self._agents[name] = agent
        self.logger.info(f"Registered agent: {name}")

    def get_agent(self, name: str) -> BaseAgent | None:
        """Get a registered agent by name."""
        return self._agents.get(name)

    def add_node(
        self,
        name: str,
        agent_name: str | None = None,
        task: str = "",
        condition: Callable[[WorkflowState], bool] | None = None,
        next_nodes: list[str] | None = None,
    ) -> "AgentOrchestrator":
        """
        Add a node to the workflow.

        Args:
            name: Node name
            agent_name: Name of agent to use
            task: Task description for the agent
            condition: Condition for executing this node
            next_nodes: List of possible next nodes

        Returns:
            Self for chaining
        """
        agent = self._agents.get(agent_name) if agent_name else None
        node = WorkflowNode(
            name=name,
            agent=agent,
            task=task,
            condition=condition,
            next_nodes=next_nodes or [],
        )
        self.nodes[name] = node
        return self

    def add_edge(self, from_node: str, to_node: str) -> "AgentOrchestrator":
        """Add an edge between nodes."""
        if from_node in self.nodes:
            if to_node not in self.nodes[from_node].next_nodes:
                self.nodes[from_node].next_nodes.append(to_node)
        return self

    def add_conditional_edges(
        self,
        from_node: str,
        condition_map: dict[str, Callable[[WorkflowState], bool]],
    ) -> "AgentOrchestrator":
        """
        Add conditional edges from a node.

        Args:
            from_node: Source node
            condition_map: Map of target node to condition function
        """
        if from_node in self.nodes:
            for target, condition in condition_map.items():
                if target not in self.nodes:
                    continue
                self.nodes[from_node].next_nodes.append(target)
                # Store condition in target node
                self.nodes[target].condition = condition
        return self

    async def run(
        self,
        start_node: str,
        initial_context: dict[str, Any] | None = None,
    ) -> WorkflowState:
        """
        Execute the workflow starting from a given node.

        Args:
            start_node: Node to start from
            initial_context: Initial context data

        Returns:
            Final workflow state
        """
        self.state = WorkflowState(context=initial_context or {})
        self.state.status = WorkflowStatus.RUNNING
        self.state.started_at = datetime.utcnow()

        self.logger.info(
            "Starting workflow",
            workflow_id=self.state.workflow_id,
            start_node=start_node,
        )

        try:
            await self._execute_node(start_node)

            # Mark completed if no errors
            if not self.state.errors:
                self.state.status = WorkflowStatus.COMPLETED
            else:
                self.state.status = WorkflowStatus.FAILED

        except Exception as e:
            self.state.status = WorkflowStatus.FAILED
            self.state.errors.append(f"Workflow failed: {str(e)}")
            self.logger.error(f"Workflow failed: {e}", exc_info=True)

        finally:
            self.state.completed_at = datetime.utcnow()

        self.logger.info(
            "Workflow completed",
            workflow_id=self.state.workflow_id,
            status=self.state.status.value,
            duration=(
                self.state.completed_at - self.state.started_at
            ).total_seconds() if self.state.completed_at and self.state.started_at else 0,
        )

        return self.state

    async def _execute_node(self, node_name: str) -> None:
        """Execute a single node in the workflow."""
        if node_name not in self.nodes:
            self.logger.warning(f"Node not found: {node_name}")
            return

        node = self.nodes[node_name]
        self.state.current_node = node_name

        # Check condition
        if node.condition and not node.condition(self.state):
            self.logger.debug(f"Skipping node (condition not met): {node_name}")
            return

        self.logger.info(f"Executing node: {node_name}")

        # Record in history
        self.state.history.append({
            "node": node_name,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "started",
        })

        # Execute agent if present
        if node.agent:
            try:
                result = await node.agent.run(
                    task=node.task,
                    context=self.state.context,
                )

                # Store result
                self.state.outputs[node_name] = result.data
                self.state.agent_states[node_name] = result.state

                # Update context with results
                if result.data:
                    if isinstance(result.data, dict):
                        self.state.context.update(result.data)

                # Callback
                if result.success and node.on_success:
                    node.on_success(self.state, result)
                elif not result.success and node.on_failure:
                    node.on_failure(self.state, Exception(str(result.errors)))

                if result.errors:
                    self.state.errors.extend(result.errors)

            except Exception as e:
                self.logger.error(f"Node execution failed: {node_name}", exc_info=True)
                self.state.errors.append(f"Node {node_name} failed: {str(e)}")
                if node.on_failure:
                    node.on_failure(self.state, e)
                return

        # Record completion
        self.state.history.append({
            "node": node_name,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "completed",
        })

        # Execute next nodes
        if node.parallel and len(node.next_nodes) > 1:
            # Execute next nodes in parallel
            await self._execute_parallel(node.next_nodes)
        else:
            # Execute sequentially
            for next_node in node.next_nodes:
                await self._execute_node(next_node)

    async def _execute_parallel(self, node_names: list[str]) -> None:
        """Execute multiple nodes in parallel."""
        tasks = [self._execute_node(name) for name in node_names]
        await asyncio.gather(*tasks, return_exceptions=True)

    def create_training_workflow(self) -> "AgentOrchestrator":
        """
        Create a standard model training workflow.

        Workflow:
        1. Data Discovery
        2. Data Ingestion
        3. Data Preprocessing
        4. Model Training
        5. Model Evaluation
        """
        self.add_node(
            "discover_data",
            agent_name="data_ingestion",
            task="Discover and catalog available adtech datasets",
            next_nodes=["ingest_data"],
        )

        self.add_node(
            "ingest_data",
            agent_name="data_ingestion",
            task="Download and ingest discovered datasets",
            next_nodes=["train_model"],
        )

        self.add_node(
            "train_model",
            agent_name="fine_tuning",
            task="Fine-tune model on ingested data",
            next_nodes=["evaluate_model"],
        )

        self.add_node(
            "evaluate_model",
            agent_name="fine_tuning",
            task="Evaluate trained model performance",
            next_nodes=[],
        )

        return self

    def create_campaign_workflow(self) -> "AgentOrchestrator":
        """
        Create a campaign optimization workflow.

        Workflow:
        1. Analyze Campaign Goals
        2. Generate Creatives
        3. Define Audiences
        4. Optimize Bids
        5. Monitor Performance
        """
        # This would use additional agents when implemented
        self.add_node(
            "analyze_goals",
            task="Analyze campaign objectives and constraints",
            next_nodes=["generate_creatives", "define_audiences"],
        )

        self.add_node(
            "generate_creatives",
            task="Generate ad creative variants",
            next_nodes=["optimize_bids"],
        )

        self.add_node(
            "define_audiences",
            task="Define and segment target audiences",
            next_nodes=["optimize_bids"],
        )

        self.add_node(
            "optimize_bids",
            task="Optimize bidding strategy",
            next_nodes=["monitor_performance"],
        )

        self.add_node(
            "monitor_performance",
            task="Monitor and report on campaign performance",
            next_nodes=[],
        )

        return self

    def get_workflow_status(self) -> dict[str, Any]:
        """Get current workflow status."""
        return self.state.to_dict()

    async def pause(self) -> None:
        """Pause the current workflow."""
        self.state.status = WorkflowStatus.PAUSED
        self.logger.info(f"Workflow paused: {self.state.workflow_id}")

    async def resume(self) -> WorkflowState:
        """Resume a paused workflow."""
        if self.state.status != WorkflowStatus.PAUSED:
            raise ValueError("Workflow is not paused")

        self.state.status = WorkflowStatus.RUNNING
        self.logger.info(f"Workflow resumed: {self.state.workflow_id}")

        # Resume from current node
        if self.state.current_node:
            await self._execute_node(self.state.current_node)

        return self.state

    def visualize(self) -> str:
        """Generate a text visualization of the workflow."""
        lines = ["Workflow Graph:", "=" * 40]

        for name, node in self.nodes.items():
            agent_str = f" [{node.agent.name}]" if node.agent else ""
            lines.append(f"\n({name}){agent_str}")
            if node.task:
                lines.append(f"  Task: {node.task[:50]}...")
            if node.next_nodes:
                for next_node in node.next_nodes:
                    arrow = "==>" if node.parallel else "-->"
                    lines.append(f"  {arrow} {next_node}")

        return "\n".join(lines)


class PlannerAgent(BaseAgent):
    """
    Agent for decomposing high-level tasks into executable plans.

    Uses LLM to break down complex objectives into agent tasks.
    """

    def __init__(self):
        super().__init__(
            name="PlannerAgent",
            description="Decomposes PRD and tasks into executable agent workflows",
        )

    async def plan(self, task: str, context: dict[str, Any]) -> list[dict[str, Any]]:
        """Create a plan for accomplishing a task."""
        # Analyze task to determine required steps
        task_lower = task.lower()

        steps = []

        if "train" in task_lower or "fine-tune" in task_lower:
            steps.extend([
                {"name": "discover_datasets", "agent": "data_ingestion", "critical": True},
                {"name": "ingest_data", "agent": "data_ingestion", "critical": True},
                {"name": "prepare_training", "agent": "fine_tuning", "critical": True},
                {"name": "execute_training", "agent": "fine_tuning", "critical": True},
                {"name": "evaluate_model", "agent": "fine_tuning", "critical": False},
            ])

        elif "generate" in task_lower and "creative" in task_lower:
            steps.extend([
                {"name": "analyze_requirements", "critical": True},
                {"name": "generate_variants", "critical": True},
                {"name": "score_variants", "critical": False},
            ])

        elif "campaign" in task_lower:
            steps.extend([
                {"name": "analyze_goals", "critical": True},
                {"name": "define_audience", "critical": True},
                {"name": "generate_creatives", "critical": True},
                {"name": "set_bidding_strategy", "critical": True},
            ])

        else:
            # Generic task decomposition
            steps.append({"name": "execute_task", "description": task, "critical": True})

        return steps

    async def execute_step(
        self,
        step: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a planning step."""
        return {"planned": True, "step": step["name"]}
