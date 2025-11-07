"""
Configuration management for DeepFake Detector.

Modern configuration using Pydantic with environment variable support.
"""

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TRUParameters(BaseModel):
    """Tree Routing Unit parameters."""

    alpha: float = Field(default=1e-3, description="Alpha parameter for routing loss")
    beta: float = Field(default=1e-2, description="Beta parameter for trace loss")
    mu_update_rate: float = Field(default=1e-3, description="Learning rate for mu updates")


class TrainingConfig(BaseModel):
    """Training configuration."""

    batch_size: int = Field(default=20, gt=0, description="Batch size for training")
    learning_rate: float = Field(default=0.0001, gt=0, description="Initial learning rate")
    max_epochs: int = Field(default=70, gt=0, description="Maximum number of epochs")
    steps_per_epoch: int = Field(default=2000, gt=0, description="Steps per epoch")
    steps_per_epoch_val: Optional[int] = Field(
        default=500, gt=0, description="Validation steps per epoch"
    )


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    image_size: int = Field(default=256, description="Input image size")
    map_size: int = Field(default=64, description="Depth map size")
    filters: int = Field(default=32, description="Base number of filters")
    tru_parameters: TRUParameters = Field(default_factory=TRUParameters)


class Settings(BaseSettings):
    """
    Main application settings with environment variable support.

    Environment variables can be prefixed with DFD_ (DeepFake Detector).
    Example: DFD_LOG_LEVEL=DEBUG
    """

    model_config = SettingsConfigDict(
        env_prefix="DFD_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Paths
    log_dir: Path = Field(default=Path("./logs/dtn"), description="Directory for logs and models")
    data_dir: Optional[List[Path]] = Field(default=None, description="Training data directories")
    data_dir_val: Optional[List[Path]] = Field(
        default=None, description="Validation data directories"
    )

    # Mode
    mode: str = Field(default="training", description="Operation mode: training or testing")

    # GPU Configuration
    gpu_usage: int = Field(default=1, ge=0, le=1, description="Enable GPU usage")
    gpu_memory_growth: bool = Field(
        default=True, description="Enable GPU memory growth"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    verbose: bool = Field(default=False, description="Enable verbose output")

    # Model, Training configs
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @field_validator("log_dir", mode="before")
    @classmethod
    def create_log_dir(cls, v: Path) -> Path:
        """Ensure log directory exists."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @field_validator("data_dir", "data_dir_val", mode="before")
    @classmethod
    def validate_data_dirs(cls, v):
        """Validate data directories."""
        if v is None:
            return v
        if isinstance(v, str):
            return [Path(v)]
        if isinstance(v, list):
            return [Path(p) if isinstance(p, str) else p for p in v]
        return v

    def display(self) -> None:
        """Display configuration in a formatted way."""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="🎯 DeepFake Detector Configuration", show_header=True)
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        # Add rows
        table.add_row("Mode", self.mode)
        table.add_row("Log Directory", str(self.log_dir))
        table.add_row("GPU Usage", "Enabled" if self.gpu_usage else "Disabled")
        table.add_row("Log Level", self.log_level)
        table.add_row("", "")
        table.add_row("[bold]Model Configuration[/bold]", "")
        table.add_row("Image Size", str(self.model.image_size))
        table.add_row("Map Size", str(self.model.map_size))
        table.add_row("Base Filters", str(self.model.filters))
        table.add_row("", "")
        table.add_row("[bold]Training Configuration[/bold]", "")
        table.add_row("Batch Size", str(self.training.batch_size))
        table.add_row("Learning Rate", str(self.training.learning_rate))
        table.add_row("Max Epochs", str(self.training.max_epochs))
        table.add_row("Steps per Epoch", str(self.training.steps_per_epoch))

        console.print(table)


def load_config(config_file: Optional[Path] = None) -> Settings:
    """
    Load configuration from file or environment.

    Args:
        config_file: Optional path to configuration file (YAML/JSON)

    Returns:
        Settings object with loaded configuration
    """
    if config_file and config_file.exists():
        import yaml

        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)
        return Settings(**config_data)

    return Settings()
