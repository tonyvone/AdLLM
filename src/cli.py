"""
Command-line interface for AdTech LLM.

Provides commands for running the API server, training models,
and managing workflows.
"""

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from src.config.settings import settings
from src.utils.logging import setup_logging

app = typer.Typer(
    name="adtech-llm",
    help="AdTech LLM - Domain-Specific AI for Advertising Technology",
)
console = Console()


@app.command()
def serve(
    host: str = typer.Option(settings.api_host, help="API host"),
    port: int = typer.Option(settings.api_port, help="API port"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
    workers: int = typer.Option(1, help="Number of workers"),
):
    """Start the AdTech LLM API server."""
    import uvicorn

    setup_logging()
    console.print(f"[green]Starting AdTech LLM API on {host}:{port}[/green]")

    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
    )


@app.command()
def train(
    dataset: str = typer.Argument(..., help="Dataset path or ID"),
    base_model: str = typer.Option(
        settings.llm.model_name,
        help="Base model to fine-tune",
    ),
    output_dir: Optional[str] = typer.Option(None, help="Output directory"),
    epochs: int = typer.Option(3, help="Number of training epochs"),
    batch_size: int = typer.Option(4, help="Training batch size"),
    learning_rate: float = typer.Option(2e-4, help="Learning rate"),
):
    """Fine-tune the model on a dataset."""
    from src.agents.fine_tuning import FineTuningAgent

    setup_logging()
    console.print("[blue]Starting model fine-tuning...[/blue]")

    async def run_training():
        agent = FineTuningAgent()
        result = await agent.run(
            task=f"Fine-tune on {dataset}",
            context={
                "data_paths": [dataset],
                "task_type": "sft",
                "config": {
                    "base_model": base_model,
                    "output_dir": output_dir,
                    "num_epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                },
            },
        )

        if result.success:
            console.print("[green]Training completed successfully![/green]")
            console.print(f"Output: {result.data}")
        else:
            console.print(f"[red]Training failed: {result.errors}[/red]")

    asyncio.run(run_training())


@app.command()
def ingest(
    source: str = typer.Option("kaggle", help="Data source"),
    query: str = typer.Option("adtech", help="Search query"),
    limit: int = typer.Option(5, help="Maximum datasets to download"),
):
    """Discover and ingest datasets."""
    from src.agents.data_ingestion import DataIngestionAgent

    setup_logging()
    console.print(f"[blue]Discovering datasets from {source}...[/blue]")

    async def run_ingestion():
        agent = DataIngestionAgent()
        result = await agent.run(
            task=f"Ingest {query} data from {source}",
            context={
                "sources": [source],
                "query": query,
                "limit": limit,
            },
        )

        if result.success:
            console.print("[green]Ingestion completed![/green]")
            if result.data:
                console.print(f"Results: {result.data}")
        else:
            console.print(f"[red]Ingestion failed: {result.errors}[/red]")

    asyncio.run(run_ingestion())


@app.command()
def generate(
    product: str = typer.Argument(..., help="Product name"),
    description: str = typer.Option("", help="Product description"),
    variants: int = typer.Option(5, help="Number of variants"),
    tone: str = typer.Option("professional", help="Ad tone"),
):
    """Generate ad creatives for a product."""
    from src.modules.creative_generation import AdCreativeGenerator
    from src.data.schemas import AdCreativeRequest, AdToneEnum

    setup_logging()
    console.print(f"[blue]Generating ad creatives for {product}...[/blue]")

    async def run_generation():
        generator = AdCreativeGenerator()

        try:
            tone_enum = AdToneEnum(tone)
        except ValueError:
            tone_enum = AdToneEnum.PROFESSIONAL

        request = AdCreativeRequest(
            product_name=product,
            product_description=description or f"Description of {product}",
            num_variants=variants,
            tone=tone_enum,
        )

        response = await generator.generate(request)

        table = Table(title=f"Ad Creatives for {product}")
        table.add_column("ID", style="cyan")
        table.add_column("Headline", style="green")
        table.add_column("Description")
        table.add_column("CTA", style="yellow")
        table.add_column("Score", style="magenta")

        for variant in response.variants:
            table.add_row(
                variant.variant_id[:8],
                variant.headline[:50] + "..." if len(variant.headline) > 50 else variant.headline,
                variant.description[:60] + "..." if len(variant.description) > 60 else variant.description,
                variant.cta_text or "-",
                f"{variant.confidence_score:.2f}",
            )

        console.print(table)

    asyncio.run(run_generation())


@app.command()
def workflow(
    workflow_type: str = typer.Argument(..., help="Workflow type: training, campaign"),
):
    """Run a workflow."""
    from src.agents.orchestrator import AgentOrchestrator

    setup_logging()
    console.print(f"[blue]Starting {workflow_type} workflow...[/blue]")

    async def run_workflow():
        orchestrator = AgentOrchestrator()

        if workflow_type == "training":
            orchestrator.create_training_workflow()
            start_node = "discover_data"
        elif workflow_type == "campaign":
            orchestrator.create_campaign_workflow()
            start_node = "analyze_goals"
        else:
            console.print(f"[red]Unknown workflow type: {workflow_type}[/red]")
            return

        console.print(orchestrator.visualize())

        state = await orchestrator.run(start_node=start_node)

        if state.status.value == "completed":
            console.print("[green]Workflow completed successfully![/green]")
        else:
            console.print(f"[red]Workflow failed: {state.errors}[/red]")

    asyncio.run(run_workflow())


@app.command()
def info():
    """Show system information."""
    table = Table(title="AdTech LLM System Information")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("App Name", settings.app_name)
    table.add_row("Version", settings.app_version)
    table.add_row("Environment", settings.environment)
    table.add_row("API Host", f"{settings.api_host}:{settings.api_port}")
    table.add_row("LLM Model", settings.llm.model_name)
    table.add_row("LLM Provider", settings.llm.provider)
    table.add_row("Database Host", settings.database.host)
    table.add_row("Redis Host", settings.redis.host)
    table.add_row("Vector DB", settings.vector_db.provider)

    console.print(table)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
