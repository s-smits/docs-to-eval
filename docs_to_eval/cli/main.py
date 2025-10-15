"""
Command line interface for docs-to-eval system using Typer and Rich
"""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Optional, Annotated

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from docs_to_eval.utils.config import EvaluationType
from ..core.classification import EvaluationTypeClassifier
from ..core.pipeline import PipelineFactory
from ..core.exceptions import DocsToEvalError
from ..utils.config import EvaluationConfig, ConfigManager, create_default_config
from ..utils.logging import setup_logging

app = typer.Typer(
    name="docs-to-eval",
    help="🤖 Automated LLM Evaluation System - Generate domain-specific benchmarks from documentation",
    add_completion=True,
    rich_markup_mode="rich",
    no_args_is_help=True
)

console = Console()


@app.command(name="evaluate", help="📊 Run evaluation on a corpus and generate benchmarks")
def evaluate(
    corpus: Annotated[str, typer.Argument(help="Path to corpus file or directory")],
    eval_type: Annotated[Optional[EvaluationType], typer.Option("--type", "-t", help="[bold]Evaluation type[/bold] (auto-detected if not specified)")] = None,
    num_questions: Annotated[int, typer.Option("--questions", "-q", min=1, max=1000, help="Number of questions to generate")] = 20,
    use_agentic: Annotated[bool, typer.Option("--agentic/--no-agentic", help="Use advanced agentic generation")] = True,
    temperature: Annotated[float, typer.Option("--temperature", min=0.0, max=2.0, help="LLM temperature")] = 0.7,
    output_dir: Annotated[str, typer.Option("--output", "-o", help="Output directory")] = "output",
    config_file: Annotated[Optional[str], typer.Option("--config", "-c", help="Configuration file path")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False
):
    """Run evaluation on a corpus"""
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load or create configuration
    config_manager = ConfigManager(config_file)
    if eval_type:
        config_manager.config.eval_type = eval_type
    config_manager.config.generation.num_questions = num_questions
    config_manager.config.generation.use_agentic = use_agentic
    config_manager.config.llm.temperature = temperature
    config_manager.config.system.output_dir = output_dir
    
    config = config_manager.config
    
    # Load corpus
    console.print(f"[blue]Loading corpus from: {corpus}[/blue]")
    corpus_text = load_corpus(corpus)
    
    if not corpus_text:
        console.print("[red]Error: Could not load corpus text[/red]")
        raise typer.Exit(1)
    
    # Show corpus stats
    show_corpus_stats(corpus_text)
    
    # Run evaluation
    run_id = str(uuid.uuid4())[:8]
    
    console.print(f"\n[green]Starting evaluation (Run ID: {run_id})[/green]")
    
    try:
        asyncio.run(run_evaluation_async(corpus_text, config, run_id, output_path))
        console.print("\n[green]✅ Evaluation completed successfully![/green]")
        console.print(f"[blue]Results saved to: {output_path}/[/blue]")
        
    except Exception as e:
        console.print(f"\n[red]❌ Evaluation failed: {str(e)}[/red]")
        raise typer.Exit(1)


async def run_evaluation_async(corpus_text: str, config: EvaluationConfig, run_id: str, output_path: Path):
    """Run evaluation asynchronously using the unified Pipeline"""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        # Create pipeline with configuration
        pipeline = PipelineFactory.create_pipeline(config)
        pipeline.run_id = run_id  # Override run_id to match CLI
        
        # Single progress task for the entire pipeline
        task = progress.add_task("Running evaluation pipeline...", total=4)
        
        # Add progress callback for pipeline phases
        
        def update_progress_callback(phase_name: str):
            phase_mapping = {
                "classification": (1, "Analyzing corpus..."),
                "generation": (2, "Generating questions..."),
                "evaluation": (3, "Evaluating with LLM..."),
                "verification": (4, "Verifying responses...")
            }
            if phase_name in phase_mapping:
                completed, description = phase_mapping[phase_name]
                progress.update(task, completed=completed, description=description)
        
        # Monkey patch the evaluation context to provide progress updates
        # This is a temporary solution until we implement proper progress callbacks in Pipeline
        import docs_to_eval.utils.logging as logging_module
        original_context_manager = logging_module.evaluation_context
        
        class ProgressAwareContext:
            def __init__(self, run_id):
                self.original_context = original_context_manager(run_id)
                self.logger = None
                
            def __enter__(self):
                self.logger = self.original_context.__enter__()
                # Override start_phase to trigger progress updates
                self.logger.original_start_phase = self.logger.start_phase
                def start_phase_with_progress(phase):
                    result = self.logger.original_start_phase(phase)
                    update_progress_callback(phase)
                    return result
                self.logger.start_phase = start_phase_with_progress
                return self.logger
                
            def __exit__(self, *args):
                return self.original_context.__exit__(*args)
        
        # Temporarily replace the context manager
        logging_module.evaluation_context = ProgressAwareContext
        
        try:
            # Run the unified pipeline
            results = await pipeline.run_async(corpus_text, output_path)
            
            # Show results summary
            show_results_summary(results)
            
        finally:
            # Restore original context manager
            logging_module.evaluation_context = original_context_manager


def load_corpus(corpus_path: str) -> str:
    """Load corpus from file or directory"""
    path = Path(corpus_path)
    
    if not path.exists():
        console.print(f"[red]Error: Path does not exist: {corpus_path}[/red]")
        return ""
    
    if path.is_file():
        try:
            return path.read_text(encoding='utf-8')
        except Exception as e:
            console.print(f"[red]Error reading file: {e}[/red]")
            return ""
    
    elif path.is_dir():
        # Load all text files from directory
        texts = []
        for file_path in path.rglob("*.txt"):
            try:
                content = file_path.read_text(encoding='utf-8')
                texts.append(f"=== {file_path.name} ===\n{content}")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not read {file_path}: {e}[/yellow]")
        
        return "\n\n".join(texts)
    
    return ""


def show_corpus_stats(corpus_text: str):
    """Display corpus statistics"""
    stats = {
        "Characters": f"{len(corpus_text):,}",
        "Words": f"{len(corpus_text.split()):,}",
        "Lines": f"{len(corpus_text.splitlines()):,}",
        "Estimated reading time": f"{len(corpus_text.split()) // 200} minutes"
    }
    
    table = Table(title="Corpus Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    for metric, value in stats.items():
        table.add_row(metric, value)
    
    console.print(table)


def show_classification_results(classification):
    """Display classification results"""
    panel_content = f"""
[cyan]Primary Type:[/cyan] {classification.primary_type}
[cyan]Secondary Types:[/cyan] {', '.join(classification.secondary_types)}
[cyan]Confidence:[/cyan] {classification.confidence:.2f}

[cyan]Analysis:[/cyan]
{classification.analysis}

[cyan]Reasoning:[/cyan]
{classification.reasoning}
    """.strip()
    
    console.print(Panel(panel_content, title="📋 Classification Results", border_style="blue"))


def show_results_summary(results: dict):
    """Display evaluation results summary"""
    metrics = results.get("aggregate_metrics", {})
    
    # Create results table
    table = Table(title="📊 Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Mean Score", f"{metrics.get('mean_score', 0.0):.3f}")
    table.add_row("Min Score", f"{metrics.get('min_score', 0.0):.3f}")
    table.add_row("Max Score", f"{metrics.get('max_score', 0.0):.3f}")
    table.add_row("Questions Evaluated", str(metrics.get('num_samples', 0)))
    
    console.print(table)
    
    # Performance level
    mean_score = metrics.get('mean_score', 0.0)
    if mean_score >= 0.8:
        level = "[green]Excellent[/green]"
    elif mean_score >= 0.6:
        level = "[yellow]Good[/yellow]"
    elif mean_score >= 0.4:
        level = "[orange3]Fair[/orange3]"
    else:
        level = "[red]Poor[/red]"
    
    console.print(f"\n[cyan]Performance Level:[/cyan] {level}")


@app.command(name="classify", help="🔍 Classify corpus evaluation type without running full evaluation")
def classify(
    corpus: Annotated[str, typer.Argument(help="Path to corpus file or directory")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed classification reasoning")] = False
):
    """Classify corpus evaluation type"""
    
    corpus_text = load_corpus(corpus)
    if not corpus_text:
        raise typer.Exit(1)
    
    console.print("[blue]Classifying corpus...[/blue]")
    
    classifier = EvaluationTypeClassifier()
    classification = classifier.classify_with_examples(corpus_text, num_examples=3)
    
    show_corpus_stats(corpus_text)
    show_classification_results(classification)
    
    if verbose:
        # Show sample questions
        sample_questions = classification.get('sample_questions', [])
        if sample_questions:
            console.print("\n[cyan]Sample Questions:[/cyan]")
            for i, example in enumerate(sample_questions, 1):
                question = example.get('question', 'N/A')
                answer = example.get('answer', 'N/A')
                console.print(f"  {i}. {question}")
                console.print(f"     Answer: {answer}")


@app.command(name="config", help="⚙️  Manage configuration files")
def config(
    create: Annotated[bool, typer.Option("--create", help="Create a new default config file")] = False,
    file: Annotated[str, typer.Option("--file", "-f", help="Config file path")] = "evaluation_config.yaml",
    show: Annotated[bool, typer.Option("--show", help="Display current configuration")] = False
):
    """Manage configuration"""
    
    if create:
        config = create_default_config()
        config_path = Path(file)
        
        # Save config
        from ..utils.config import save_config
        save_config(config, config_path)
        
        console.print(f"[green]✅ Default configuration created: {file}[/green]")
    
    elif show:
        if Path(file).exists():
            from ..utils.config import load_config
            config = load_config(file)
            console.print("[cyan]Current Configuration:[/cyan]")
            console.print(json.dumps(config.dict(), indent=2, default=str))
        else:
            console.print(f"[yellow]Configuration file not found: {file}[/yellow]")
            console.print("Use --create to create a default configuration")
    
    else:
        console.print("Use --create to create config or --show to display current config")


@app.command(name="server", help="🚀 Start the FastAPI web server")
def server(
    host: Annotated[str, typer.Option("--host", help="Host address to bind to")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", "-p", min=1024, max=65535, help="Port number")] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="Enable hot-reload for development")] = False,
    log_level: Annotated[str, typer.Option("--log-level", help="Logging level")] = "info"
):
    """Start the FastAPI server"""
    
    try:
        import uvicorn
        
        console.print(f"[green]🚀 Starting docs-to-eval server on {host}:{port}[/green]")
        console.print(f"[blue]📚 API docs available at: http://{host}:{port}/docs[/blue]")
        
        uvicorn.run(
            "docs_to_eval.ui_api.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level
        )
        
    except ImportError:
        console.print("[red]❌ FastAPI server dependencies not installed[/red]")
        console.print("[blue]Install with: pip install 'docs-to-eval[all]'[/blue]")
        raise typer.Exit(1)


@app.command(name="version", help="ℹ️  Show version and system information")
def version():
    """Display version and system information"""
    
    try:
        from .. import __version__
        version_str = __version__
    except ImportError:
        version_str = "1.0.0"
    
    console.print(f"[green]docs-to-eval version {version_str}[/green]")
    
    # Show system info
    import sys
    import platform
    
    table = Table(title="System Information")
    table.add_column("Component", style="cyan")
    table.add_column("Version", style="magenta")
    
    table.add_row("Python", sys.version.split()[0])
    table.add_row("Platform", platform.platform())
    table.add_row("Architecture", platform.architecture()[0])
    
    console.print(table)


@app.command(name="interactive", help="🎯 Start interactive guided session")
def interactive():
    """Launch the interactive session for step-by-step evaluation setup"""
    try:
        from .interactive import InteractiveSession
        
        console.print("[bold blue]Starting Interactive Mode...[/bold blue]\n")
        session = InteractiveSession()
        session.run()
        
    except ImportError as e:
        console.print(f"[red]✗ Error: Interactive mode dependencies not available: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interactive session cancelled by user[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"[red]✗ Interactive session error: {e}[/red]")
        if console.is_terminal:
            import traceback
            console.print("[dim]" + traceback.format_exc() + "[/dim]")
        raise typer.Exit(1)


def main_cli():
    """Main CLI entry point with global error handling"""
    try:
        app()
    except DocsToEvalError as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e.message}")
        if e.details:
            console.print("[dim]Details:[/dim]")
            for key, value in e.details.items():
                console.print(f"  [dim]{key}:[/dim] {value}")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Operation cancelled by user[/yellow]")
        raise typer.Exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        console.print(f"\n[bold red]✗ Unexpected error:[/bold red] {e}")
        if console.is_terminal:
            import traceback
            console.print("[dim]" + traceback.format_exc() + "[/dim]")
        raise typer.Exit(1)


if __name__ == "__main__":
    main_cli()