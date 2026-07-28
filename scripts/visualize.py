#!/usr/bin/env python3
"""Visualize weighted mixture selections from a 140-dim simplex weight vector.

Usage:
    # Demo: generate figures for several interesting weight vectors
    python scripts/visualize.py --demo

    # Visualize a specific weight vector from a .npy file
    python scripts/visualize.py --weights path/to/weights.npy

    # Visualize a specific weight vector from a .npz file
    python scripts/visualize.py --weights path/to/weights.npz

    # Visualize the 4-sparse optimum (all 4 Type-3 quartets equally weighted)
    python scripts/visualize.py --preset four-sparse

    # Visualize the RKE optimum (all 16 Type-1 equally weighted)
    python scripts/visualize.py --preset rke-optimum

    # Show only specific mixture types
    python scripts/visualize.py --preset type1-only
    python scripts/visualize.py --preset type2-only
    python scripts/visualize.py --preset type3-only

    # Custom config
    python scripts/visualize.py --config path/to/config.yaml --demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.sampling import SampleManager
from src.visualize import (
    build_mixture_index,
    demo_weight_vectors,
    plot_mixture_selection,
    plot_reference,
)


# ======================================================================
# Preset weight vectors
# ======================================================================

def _build_presets(manifest: dict[str, Path]) -> dict[str, np.ndarray]:
    """Build preset weight vectors for common cases."""
    idx_map = build_mixture_index(manifest)
    label_to_idx = {v: k for k, v in idx_map.items()}
    n = len(manifest)

    presets: dict[str, np.ndarray] = {}

    # --- 4-sparse optimum: all 4 Type-3 equally weighted ---
    w = np.zeros(n)
    for qi in ["type3_0_1_2_3", "type3_4_5_6_7",
               "type3_8_9_10_11", "type3_12_13_14_15"]:
        w[label_to_idx[qi]] = 0.25
    presets["four-sparse"] = w

    # --- RKE optimum: all 16 Type-1 equally weighted ---
    w = np.zeros(n)
    for k in range(16):
        w[label_to_idx[f"type1_{k}"]] = 1.0 / 16
    presets["rke-optimum"] = w

    # --- All Type-1 equally weighted ---
    w = np.zeros(n)
    for k in range(16):
        w[label_to_idx[f"type1_{k}"]] = 1.0 / 16
    presets["type1-only"] = w

    # --- All Type-2 equally weighted ---
    w = np.zeros(n)
    type2_labels = [lbl for lbl in manifest if lbl.startswith("type2_")]
    for lbl in type2_labels:
        w[label_to_idx[lbl]] = 1.0 / len(type2_labels)
    presets["type2-only"] = w

    # --- All Type-3 equally weighted ---
    w = np.zeros(n)
    type3_labels = [lbl for lbl in manifest if lbl.startswith("type3_")]
    for lbl in type3_labels:
        w[label_to_idx[lbl]] = 1.0 / len(type3_labels)
    presets["type3-only"] = w

    # --- Mixed: one representative from each type ---
    w = np.zeros(n)
    w[label_to_idx["type1_0"]] = 0.2
    w[label_to_idx["type2_4_12"]] = 0.3
    w[label_to_idx["type3_8_9_10_11"]] = 0.5
    presets["mixed"] = w

    # --- Single Type-1 ---
    w = np.zeros(n)
    w[label_to_idx["type1_0"]] = 1.0
    presets["single-type1"] = w

    # --- Single Type-2 ---
    w = np.zeros(n)
    w[label_to_idx["type2_0_8"]] = 1.0
    presets["single-type2"] = w

    # --- Single Type-3 ---
    w = np.zeros(n)
    w[label_to_idx["type3_0_1_2_3"]] = 1.0
    presets["single-type3"] = w

    return presets


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize weighted GMM mixture selections."
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to YAML config file (default: config.yaml in project root).",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Generate demo figures for several interesting weight vectors.",
    )
    parser.add_argument(
        "--weights", "-w", type=str, default=None,
        help="Path to a .npy or .npz file containing a 140-dim weight vector.",
    )
    parser.add_argument(
        "--preset", "-p", type=str, default=None,
        choices=[
            "four-sparse", "rke-optimum",
            "type1-only", "type2-only", "type3-only",
            "mixed", "single-type1", "single-type2", "single-type3",
        ],
        help="Use a built-in preset weight vector.",
    )
    parser.add_argument(
        "--reference", action="store_true",
        help="Generate reference plot: all 16 base Gaussians with labels.",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output path for the figure (default: auto-generated).",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    viz_cfg = config.visualization

    # Load manifest
    samples_dir = Path(config.sampling.output_dir)
    try:
        manifest = SampleManager.load_manifest(samples_dir)
    except FileNotFoundError:
        print(f"ERROR: No manifest found at {samples_dir / 'manifest.json'}")
        print("Run 'python scripts/generate_samples.py' first.")
        sys.exit(1)

    print(f"Loaded {len(manifest)} mixtures from manifest.")

    figures_dir = Path(viz_cfg.figure_dir)

    # -- Reference plot ------------------------------------------------------
    if args.reference:
        print("  Plotting: Reference (all 16 base Gaussians)")
        ref_path = args.output or (figures_dir / "reference_16_gaussians.png")
        plot_reference(
            manifest=manifest,
            total_samples=viz_cfg.total_samples,
            figsize=viz_cfg.figsize,
            alpha=viz_cfg.scatter_alpha,
            point_size=viz_cfg.scatter_point_size,
            save_path=ref_path,
            dpi=viz_cfg.dpi,
        )
        print(f"  Saved to {ref_path}")

    # -- Determine which weight vectors to plot ----------------------------
    to_plot: dict[str, np.ndarray] = {}

    if args.weights:
        # Load from file
        w_path = Path(args.weights)
        if w_path.suffix == ".npz":
            data = np.load(w_path)
            # Try common key names
            for key in ["weights", "w", "arr_0"]:
                if key in data:
                    weights = data[key]
                    break
            else:
                weights = list(data.values())[0]
        else:
            weights = np.load(w_path)
        to_plot["Custom weights"] = weights

    if args.preset:
        presets = _build_presets(manifest)
        to_plot[args.preset] = presets[args.preset]

    if args.demo or (not args.weights and not args.preset and not args.reference):
        presets = _build_presets(manifest)
        demos = demo_weight_vectors(manifest)
        # Use the demo weight vectors
        to_plot.update(demos)

    if not to_plot:
        print("No weight vectors specified. Use --demo, --preset, or --weights.")
        sys.exit(1)

    # -- Plot each weight vector -------------------------------------------
    for name, weights in to_plot.items():
        n_active = int(np.sum(np.asarray(weights) > 0))
        print(f"  Plotting: {name} ({n_active} active mixtures)")

        if args.output and len(to_plot) == 1:
            save_path = Path(args.output)
        else:
            safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            save_path = figures_dir / f"{safe_name}.png"

        plot_mixture_selection(
            weights=weights,
            manifest=manifest,
            total_samples=viz_cfg.total_samples,
            figsize=viz_cfg.figsize,
            alpha=viz_cfg.scatter_alpha,
            point_size=viz_cfg.scatter_point_size,
            title=name,
            save_path=save_path,
            dpi=viz_cfg.dpi,
        )

    print(f"\nDone! Figures saved to {figures_dir.resolve()}/")


if __name__ == "__main__":
    main()
