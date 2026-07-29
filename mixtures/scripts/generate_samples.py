#!/usr/bin/env python3
"""Generate samples from all 140 GMM candidate mixtures and save as .npz files.

Usage:
    python scripts/generate_samples.py                  # uses default config.yaml
    python scripts/generate_samples.py --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.grid_mixtures import GridMixtureFactory
from src.mixtures import RingMixtureFactory
from src.random_mixtures import RandomMixtureFactory
from src.sampling import SampleManager


def main():
    parser = argparse.ArgumentParser(
        description="Generate samples from all 140 GMM candidate mixtures."
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML config file (default: config.yaml in project root).",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    layout = config.gmm.layout
    print(f"Layout: {layout}")
    print(f"Loaded config: σ={config.gmm.sigma}, seed={config.gmm.random_seed}")
    print(f"Samples per mixture: {config.sampling.n_samples}")

    # Route samples to layout-specific subdirectory
    base_dir = Path(config.sampling.output_dir)
    output_dir = base_dir / layout
    print(f"Output directory: {output_dir}")

    # Create all mixtures (ring: 140, grid: 265, random_3d: 140)
    print("\nCreating mixture candidates...")
    if layout == "grid":
        factory = GridMixtureFactory(config.gmm)
    elif layout == "random_3d":
        factory = RandomMixtureFactory(config.gmm)
    else:
        factory = RingMixtureFactory(config.gmm)
    mixtures = factory.create_all()

    n_type1 = sum(1 for m in mixtures.values() if m.mixture_type == 1)
    n_type2 = sum(1 for m in mixtures.values() if m.mixture_type == 2)
    n_type3 = sum(1 for m in mixtures.values() if m.mixture_type == 3)
    print(f"  Type 1: {n_type1}")
    print(f"  Type 2: {n_type2}")
    print(f"  Type 3: {n_type3}")
    print(f"  Total: {len(mixtures)}")

    # Generate and save samples
    print("\nGenerating samples...")
    manager = SampleManager(config)
    manifest = manager.generate_all(mixtures, output_dir=output_dir)

    print(f"\nDone! {len(manifest)} .npz files saved to {output_dir}/")
    print(f"Manifest: {output_dir}/manifest.json")


if __name__ == "__main__":
    main()
