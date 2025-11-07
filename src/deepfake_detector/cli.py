"""
Command-line interface for DeepFake Detector.

Modern CLI using Typer with rich formatting and intuitive commands.
"""

from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing_extensions import Annotated

from deepfake_detector.__about__ import __version__
from deepfake_detector.core.config import Settings, load_config
from deepfake_detector.core.logger import setup_logger

app = typer.Typer(
    name="deepfake-detector",
    help="🎭 Advanced Deep Tree Network for DeepFake Detection",
    add_completion=True,
    rich_markup_mode="rich",
)
console = Console()


def version_callback(value: bool):
    """Display version information."""
    if value:
        console.print(
            Panel(
                f"[bold cyan]DeepFake Detector[/bold cyan]\n"
                f"[yellow]Version:[/yellow] {__version__}\n"
                f"[yellow]Author:[/yellow] Umit Kacar",
                title="🎭 DeepFake Detector",
                border_style="cyan",
            )
        )
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-v",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit",
        ),
    ] = None,
):
    """
    🎭 DeepFake Detector - Advanced Deep Tree Network for DeepFake Detection.

    A state-of-the-art deep learning solution for detecting deepfake videos
    and images using Deep Tree Networks (DTN) with Tree Routing Units (TRU).
    """
    pass


@app.command()
def train(
    data_dir: Annotated[
        List[Path],
        typer.Option(
            "--data-dir",
            "-d",
            help="Training data directories (can specify multiple)",
        ),
    ],
    val_dir: Annotated[
        Optional[List[Path]],
        typer.Option(
            "--val-dir",
            help="Validation data directories (optional)",
        ),
    ] = None,
    config_file: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="Configuration file (YAML/JSON)",
        ),
    ] = None,
    log_dir: Annotated[
        Path,
        typer.Option(
            "--log-dir",
            "-l",
            help="Directory for logs and model checkpoints",
        ),
    ] = Path("./logs/dtn"),
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            min=1,
            help="Batch size for training",
        ),
    ] = 20,
    learning_rate: Annotated[
        float,
        typer.Option(
            "--learning-rate",
            "--lr",
            min=0.0,
            help="Initial learning rate",
        ),
    ] = 0.0001,
    epochs: Annotated[
        int,
        typer.Option(
            "--epochs",
            "-e",
            min=1,
            help="Number of epochs to train",
        ),
    ] = 70,
    steps_per_epoch: Annotated[
        int,
        typer.Option(
            "--steps-per-epoch",
            min=1,
            help="Steps per epoch",
        ),
    ] = 2000,
    gpu: Annotated[
        bool,
        typer.Option(
            "--gpu/--no-gpu",
            help="Enable GPU acceleration",
        ),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-V",
            help="Enable verbose output",
        ),
    ] = False,
):
    """
    🎓 Train the DeepFake detection model.

    Train a Deep Tree Network on your dataset with customizable parameters.
    Supports multi-directory training and validation data.

    Example:
        $ deepfake-detector train -d ./data/fake -d ./data/real --epochs 100
    """
    try:
        # Load configuration
        if config_file:
            settings = load_config(config_file)
        else:
            settings = Settings()

        # Override with CLI arguments
        if data_dir:
            settings.data_dir = data_dir
        if val_dir:
            settings.data_dir_val = val_dir
        settings.log_dir = log_dir
        settings.training.batch_size = batch_size
        settings.training.learning_rate = learning_rate
        settings.training.max_epochs = epochs
        settings.training.steps_per_epoch = steps_per_epoch
        settings.gpu_usage = 1 if gpu else 0
        settings.verbose = verbose

        # Setup logger
        setup_logger(
            log_level="DEBUG" if verbose else "INFO",
            log_file=log_dir / "training.log",
            verbose=verbose,
        )

        # Display configuration
        console.print("\n")
        settings.display()
        console.print("\n")

        # Import and run training (lazy import to speed up CLI)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(description="Initializing training...", total=None)

            from deepfake_detector.training.trainer import Trainer

            trainer = Trainer(settings)
            trainer.train()

        console.print(
            Panel(
                "[bold green]✓ Training completed successfully![/bold green]",
                border_style="green",
            )
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def test(
    data_dir: Annotated[
        List[Path],
        typer.Option(
            "--data-dir",
            "-d",
            help="Test data directories",
        ),
    ],
    model_path: Annotated[
        Path,
        typer.Option(
            "--model",
            "-m",
            exists=True,
            help="Path to trained model checkpoint",
        ),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output file for results (CSV/JSON)",
        ),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            min=1,
        ),
    ] = 20,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-V"),
    ] = False,
):
    """
    🧪 Test the model on a dataset.

    Evaluate model performance on test data and generate metrics.

    Example:
        $ deepfake-detector test -d ./data/test -m ./logs/model.ckpt -o results.csv
    """
    try:
        console.print(
            Panel(
                f"[cyan]Testing model:[/cyan] {model_path}\n"
                f"[cyan]Data directories:[/cyan] {', '.join(str(d) for d in data_dir)}",
                title="🧪 Model Testing",
                border_style="cyan",
            )
        )

        # Setup logger
        setup_logger(log_level="DEBUG" if verbose else "INFO", verbose=verbose)

        # Import and run testing
        from deepfake_detector.testing.tester import Tester

        tester = Tester(model_path=model_path)
        results = tester.test(data_dir, batch_size=batch_size)

        # Save results if output specified
        if output:
            results.to_csv(output)
            console.print(f"[green]✓ Results saved to {output}[/green]")

        console.print(
            Panel(
                f"[bold green]✓ Testing completed![/bold green]\n"
                f"[cyan]Accuracy:[/cyan] {results.get('accuracy', 'N/A'):.2%}",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def predict(
    input_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            help="Input image or video file",
        ),
    ],
    model_path: Annotated[
        Path,
        typer.Option(
            "--model",
            "-m",
            exists=True,
            help="Path to trained model checkpoint",
        ),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output file for visualization",
        ),
    ] = None,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            "-t",
            min=0.0,
            max=1.0,
            help="Detection threshold",
        ),
    ] = 0.5,
    visualize: Annotated[
        bool,
        typer.Option(
            "--visualize/--no-visualize",
            help="Generate visualization",
        ),
    ] = True,
):
    """
    🔍 Predict if an image/video is a deepfake.

    Run inference on a single file and get prediction results.

    Example:
        $ deepfake-detector predict image.jpg -m ./model.ckpt --visualize
    """
    try:
        console.print(
            Panel(
                f"[cyan]Analyzing:[/cyan] {input_path}\n"
                f"[cyan]Model:[/cyan] {model_path}",
                title="🔍 DeepFake Detection",
                border_style="cyan",
            )
        )

        # Import and run prediction
        from deepfake_detector.inference.predictor import Predictor

        predictor = Predictor(model_path=model_path)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(description="Running inference...", total=None)
            result = predictor.predict(input_path, threshold=threshold)

        # Display results
        is_fake = result["is_fake"]
        confidence = result["confidence"]

        result_text = (
            f"[bold red]FAKE[/bold red]" if is_fake else "[bold green]REAL[/bold green]"
        )
        console.print(
            Panel(
                f"[bold]Result:[/bold] {result_text}\n"
                f"[cyan]Confidence:[/cyan] {confidence:.2%}\n"
                f"[cyan]Score:[/cyan] {result['score']:.4f}",
                title="🎯 Detection Result",
                border_style="green" if not is_fake else "red",
            )
        )

        # Save visualization
        if visualize and output:
            predictor.visualize(input_path, result, output)
            console.print(f"[green]✓ Visualization saved to {output}[/green]")

    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {str(e)}")
        console.print_exception()
        raise typer.Exit(1)


@app.command()
def config(
    show: Annotated[
        bool,
        typer.Option(
            "--show",
            "-s",
            help="Show current configuration",
        ),
    ] = False,
    generate: Annotated[
        Optional[Path],
        typer.Option(
            "--generate",
            "-g",
            help="Generate configuration file template",
        ),
    ] = None,
):
    """
    ⚙️  Manage configuration.

    View or generate configuration files for the detector.

    Example:
        $ deepfake-detector config --show
        $ deepfake-detector config --generate config.yaml
    """
    if show:
        settings = Settings()
        settings.display()

    if generate:
        import yaml

        settings = Settings()
        config_dict = settings.model_dump(mode="python")

        with open(generate, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        console.print(f"[green]✓ Configuration template saved to {generate}[/green]")


if __name__ == "__main__":
    app()
