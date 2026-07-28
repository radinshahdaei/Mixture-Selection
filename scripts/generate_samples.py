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
from src.mixtures import MixtureFactory
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
    print(f"Loaded config: n={config.gmm.n_components}, "
          f"R={config.gmm.radius}, σ={config.gmm.sigma}, "
          f"seed={config.gmm.random_seed}")
    print(f"Samples per mixture: {config.sampling.n_samples}")
    print(f"Output directory: {config.sampling.output_dir}")

    # Create all 140 mixtures
    print("\nCreating mixture candidates...")
    factory = MixtureFactory(config.gmm)
    mixtures = factory.create_all()

    n_type1 = sum(1 for m in mixtures.values() if m.mixture_type == 1)
    n_type2 = sum(1 for m in mixtures.values() if m.mixture_type == 2)
    n_type3 = sum(1 for m in mixtures.values() if m.mixture_type == 3)
    print(f"  Type 1 (single): {n_type1}")
    print(f"  Type 2 (pairs):  {n_type2}")
    print(f"  Type 3 (quartets): {n_type3}")
    print(f"  Total: {len(mixtures)}")

    # Generate and save samples
    print("\nGenerating samples...")
    manager = SampleManager(config)
    manifest = manager.generate_all(mixtures)

    print(f"\nDone! {len(manifest)} .npz files saved to {config.sampling.output_dir}/")
    print(f"Manifest: {config.sampling.output_dir}/manifest.json")


if __name__ == "__main__":
    main()
