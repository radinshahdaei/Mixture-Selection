"""Configuration loading from YAML."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class GMMConfig:
    """Parameters for the GMM component generation."""

    layout: str = "ring"  # "ring" or "grid"
    sigma: float = 0.2
    random_seed: int = 42

    # Ring layout
    n_components: int = 16
    radius: float = 1.0

    # Grid layout
    grid_rows: int = 6
    grid_cols: int = 6
    grid_spacing: float = 0.4


@dataclass
class SamplingConfig:
    """Parameters for sample generation."""

    n_samples: int = 5000
    output_dir: str = "data/samples"


@dataclass
class VisualizationConfig:
    """Parameters for visualization."""

    style: str = "light"
    dpi: int = 150
    figsize: tuple = (10, 10)
    figure_dir: str = "figures"
    total_samples: int = 8000
    scatter_alpha: float = 0.35
    scatter_point_size: float = 1.2


@dataclass
class Config:
    """Top-level configuration aggregating all sub-configs."""

    gmm: GMMConfig = field(default_factory=GMMConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        gmm_cfg = GMMConfig(**raw.get("gmm", {}))
        samp_cfg = SamplingConfig(**raw.get("sampling", {}))
        viz_cfg = VisualizationConfig(**raw.get("visualization", {}))

        return cls(gmm=gmm_cfg, sampling=samp_cfg, visualization=viz_cfg)


def load_config(path: Optional[str | Path] = None) -> Config:
    """Load config from path, or default to 'config.yaml' in the project root."""
    if path is None:
        # Resolve relative to this file's location: src/config.py → project root
        project_root = Path(__file__).resolve().parent.parent
        path = project_root / "config.yaml"
    return Config.from_yaml(path)
