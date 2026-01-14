"""
Base agent class for the AdTech LLM agentic system.

Provides common functionality for all specialized agents including
state management, logging, error handling, and tool execution.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, TypeVar
from uuid import uuid4

from src.config.settings import settings
from src.utils.logging import get_logger

T = TypeVar("T")


class AgentStatus(str, Enum):
    """Agent execution status."""

    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"  # Waiting for human approval or external input
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentState:
    """
    Represents the current state of an agent's execution.

    Attributes:
        agent_id: Unique identifier for the agent instance
        status: Current execution status
        current_task: Description of current task
        progress: Progress percentage (0-100)
        messages: List of status messages
        artifacts: Output artifacts produced
        errors: List of errors encountered
        metadata: Additional state metadata
    """

    agent_id: str = field(default_factory=lambda: str(uuid4()))
    status: AgentStatus = AgentStatus.IDLE
    current_task: str = ""
    progress: float = 0.0
    messages: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    iteration: int = 0
    max_iterations: int = field(default_factory=lambda: settings.agents.max_iterations)

    def add_message(self, message: str) -> None:
        """Add a status message."""
        timestamp = datetime.utcnow().isoformat()
        self.messages.append(f"[{timestamp}] {message}")

    def add_error(self, error: str) -> None:
        """Add an error message."""
        timestamp = datetime.utcnow().isoformat()
        self.errors.append(f"[{timestamp}] {error}")

    def set_artifact(self, key: str, value: Any) -> None:
        """Store an artifact."""
        self.artifacts[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "current_task": self.current_task,
            "progress": self.progress,
            "messages": self.messages,
            "artifacts": self.artifacts,
            "errors": self.errors,
            "metadata": self.metadata,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "iteration": self.iteration,
        }


@dataclass
class AgentResult:
    """
    Result of an agent's execution.

    Attributes:
        success: Whether execution was successful
        data: Result data
        errors: List of errors if any
        metrics: Execution metrics
        state: Final agent state
    """

    success: bool
    data: Any = None
    errors: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    state: AgentState | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "errors": self.errors,
            "metrics": self.metrics,
            "state": self.state.to_dict() if self.state else None,
        }


class AgentTool:
    """
    Wrapper for agent tools/actions.

    Provides a standardized interface for tools that agents can use.
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        parameters: dict[str, Any] | None = None,
    ):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters or {}

    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with given parameters."""
        import asyncio
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(**kwargs)
        return self.func(**kwargs)


class BaseAgent(ABC):
    """
    Abstract base class for all AdTech LLM agents.

    Provides common functionality including:
    - State management
    - Logging
    - Tool registration and execution
    - Error handling
    - Human-in-the-loop support
    """

    def __init__(
        self,
        name: str,
        description: str,
        tools: list[AgentTool] | None = None,
    ):
        self.name = name
        self.description = description
        self.tools: dict[str, AgentTool] = {}
        self.state = AgentState()
        self.logger = get_logger(f"agent.{name}")

        # Register provided tools
        if tools:
            for tool in tools:
                self.register_tool(tool)

    def register_tool(self, tool: AgentTool) -> None:
        """Register a tool for this agent."""
        self.tools[tool.name] = tool
        self.logger.debug(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> AgentTool | None:
        """Get a registered tool by name."""
        return self.tools.get(name)

    async def use_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """
        Execute a registered tool.

        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool parameters

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool is not found
        """
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        self.logger.info(f"Executing tool: {tool_name}", parameters=kwargs)
        try:
            result = await tool.execute(**kwargs)
            self.logger.info(f"Tool completed: {tool_name}")
            return result
        except Exception as e:
            self.logger.error(f"Tool failed: {tool_name}", error=str(e))
            raise

    def update_progress(self, progress: float, message: str | None = None) -> None:
        """Update agent progress."""
        self.state.progress = min(100.0, max(0.0, progress))
        if message:
            self.state.add_message(message)
            self.logger.info(message, progress=progress)

    def set_task(self, task: str) -> None:
        """Set the current task description."""
        self.state.current_task = task
        self.logger.info(f"Starting task: {task}")

    async def request_human_approval(self, request: str, context: dict[str, Any]) -> bool:
        """
        Request human approval for an action.

        If human-in-the-loop is disabled, automatically approves.

        Args:
            request: Description of what needs approval
            context: Additional context for the decision

        Returns:
            True if approved, False otherwise
        """
        if not settings.agents.enable_human_in_loop:
            self.logger.debug(f"Auto-approving (HITL disabled): {request}")
            return True

        self.state.status = AgentStatus.WAITING
        self.state.add_message(f"Awaiting approval: {request}")
        self.logger.warning(f"Human approval required: {request}", context=context)

        # In a real implementation, this would wait for human input
        # For now, we auto-approve after logging
        return True

    @abstractmethod
    async def plan(self, task: str, context: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Create an execution plan for a task.

        Args:
            task: Task description
            context: Additional context

        Returns:
            List of planned steps
        """
        pass

    @abstractmethod
    async def execute_step(
        self,
        step: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute a single step in the plan.

        Args:
            step: Step to execute
            context: Execution context

        Returns:
            Step result
        """
        pass

    async def run(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Run the agent on a task.

        Args:
            task: Task description
            context: Additional context

        Returns:
            Agent execution result
        """
        context = context or {}
        self.state = AgentState()
        self.state.status = AgentStatus.RUNNING
        self.state.started_at = datetime.utcnow()
        self.set_task(task)

        self.logger.info(f"Agent starting: {self.name}", task=task)

        try:
            # Create execution plan
            self.update_progress(5, "Creating execution plan...")
            plan = await self.plan(task, context)
            self.state.set_artifact("plan", plan)

            # Execute each step
            total_steps = len(plan)
            results = []

            for i, step in enumerate(plan):
                self.state.iteration += 1

                # Check iteration limit
                if self.state.iteration > self.state.max_iterations:
                    self.state.add_error("Maximum iterations exceeded")
                    break

                step_name = step.get("name", f"Step {i + 1}")
                self.set_task(step_name)

                progress = 10 + (80 * (i / total_steps))
                self.update_progress(progress, f"Executing: {step_name}")

                try:
                    step_result = await self.execute_step(step, context)
                    results.append({"step": step_name, "success": True, "result": step_result})
                    context.update(step_result)  # Pass results to next step
                except Exception as e:
                    error_msg = f"Step failed: {step_name} - {str(e)}"
                    self.state.add_error(error_msg)
                    results.append({"step": step_name, "success": False, "error": str(e)})
                    self.logger.error(error_msg, exc_info=True)

                    # Decide whether to continue or abort
                    if step.get("critical", False):
                        break

            # Complete
            self.state.status = AgentStatus.COMPLETED
            self.state.completed_at = datetime.utcnow()
            self.update_progress(100, "Task completed")

            success = all(r.get("success", False) for r in results)
            return AgentResult(
                success=success,
                data={"results": results, "artifacts": self.state.artifacts},
                errors=self.state.errors,
                metrics={
                    "total_steps": total_steps,
                    "completed_steps": len(results),
                    "iterations": self.state.iteration,
                    "duration_seconds": (
                        self.state.completed_at - self.state.started_at
                    ).total_seconds(),
                },
                state=self.state,
            )

        except Exception as e:
            self.state.status = AgentStatus.FAILED
            self.state.completed_at = datetime.utcnow()
            error_msg = f"Agent execution failed: {str(e)}"
            self.state.add_error(error_msg)
            self.logger.error(error_msg, exc_info=True)

            return AgentResult(
                success=False,
                errors=self.state.errors,
                state=self.state,
            )

    def get_state(self) -> AgentState:
        """Get current agent state."""
        return self.state

    def get_tools_schema(self) -> list[dict[str, Any]]:
        """Get schema for all registered tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self.tools.values()
        ]
