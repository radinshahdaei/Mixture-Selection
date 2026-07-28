"""Visualization of weighted mixture selections from GMM candidates.

Given a 140-dimensional simplex weight vector (over the 140 candidate mixtures),
plot samples from each active mixture in a distinct color on a white background.

All plots are built **from samples only** (loaded from .npz files).
"""

from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from numpy.typing import NDArray

from .sampling import SampleManager


# ======================================================================
# Color palette generation
# ======================================================================

def _generate_colors(n: int) -> list[str]:
    """Generate *n* vibrant, perceptually distinct colors in HSV space."""
    colors = []
    for i in range(n):
        hue = i / n
        # High saturation + high value = vivid, punchy colors
        sat = 0.90 + 0.10 * ((i % 3) / 3)
        val = 0.88 + 0.12 * ((i % 2) / 2)
        rgb = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append("#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255),
        ))
    return colors


# Precomputed 140-color palette (one per candidate mixture)
CANDIDATE_COLORS = _generate_colors(140)

# Style-consistent colors
COMPONENT_MEAN_COLOR = "#333333"
REFERENCE_CIRCLE_COLOR = "#cccccc"
GRID_COLOR = "#e8e8e8"


# ======================================================================
# Mixture index mapping (simplex dimension → mixture label)
# ======================================================================

def build_mixture_index(manifest: dict[str, Path]) -> dict[int, str]:
    """Build a mapping from simplex dimension (0..N-1) to mixture label.

    Ordering: Type 1, then Type 2, then Type 3. Within each type, labels
    are sorted by their integer components. Works for both ring labels
    (e.g. ``type1_0``, ``type2_0_1``) and grid labels
    (e.g. ``type1_0_0``, ``type2_0_1_2_3``, ``type3_block_0``).
    """
    def _sort_key(label: str):
        parts = label.split("_")
        type_str = parts[0]  # "type1", "type2", "type3"
        type_num = int(type_str[-1])
        if type_num == 3 and len(parts) > 1 and parts[1] == "block":
            # type3_block_N
            return (type_num, int(parts[2]))
        else:
            return (type_num, tuple(int(p) for p in parts[1:]))

    type1 = sorted(
        [k for k in manifest if k.startswith("type1_")],
        key=_sort_key,
    )
    type2 = sorted(
        [k for k in manifest if k.startswith("type2_")],
        key=_sort_key,
    )
    type3 = sorted(
        [k for k in manifest if k.startswith("type3_")],
        key=_sort_key,
    )
    ordered = type1 + type2 + type3
    return {i: label for i, label in enumerate(ordered)}


def get_label_from_index(index: int, manifest: dict[str, Path]) -> str:
    """Get the mixture label for a given simplex dimension."""
    mapping = build_mixture_index(manifest)
    return mapping[index]


# ======================================================================
# Main visualization
# ======================================================================

def plot_mixture_selection(
    weights: Union[NDArray[np.floating], dict[str, float], Sequence[float]],
    manifest: dict[str, Path],
    total_samples: int = 8000,
    figsize: tuple[float, float] = (10, 10),
    alpha: float = 0.35,
    point_size: float = 1.2,
    show_means: bool = False,
    show_circle: bool = True,
    layout: str = "ring",
    title: Optional[str] = None,
    legend: bool = True,
    legend_max_entries: int = 20,
    rng_seed: int = 42,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Plot a weighted mixture-of-mixtures from a 140-dim simplex weight vector.

    Each candidate mixture with non-zero weight contributes samples in its own
    colour, proportional to its weight. This is the core visualization for
    mixture-selection experiments.

    Parameters
    ----------
    weights : array-like or dict
        If a 140-dim array/list: ``weights[i]`` is the weight for the i-th
        candidate (ordered: Type 1 g0..g15, Type 2 all pairs, Type 3 all quartets).
        If a dict: keys are mixture labels (e.g. ``"type1_0"``) and values are weights.
        Weights are normalised automatically.
    manifest : dict[str, Path]
        Mapping from mixture labels to .npz file paths.
    total_samples : int
        Total number of points to plot (subsampled from .npz files).
    figsize : tuple
        Figure size in inches.
    alpha : float
        Transparency of scatter points.
    point_size : float
        Size of scatter markers.
    show_means : bool
        Whether to mark component means with X markers.
    show_circle : bool
        Whether to draw the reference circle (ring layout only).
    layout : str
        ``"ring"`` or ``"grid"`` — controls reference geometry decorations.
    title : str, optional
        Plot title (auto-generated if None).
    legend : bool
        Whether to show a legend.
    legend_max_entries : int
        Maximum legend entries (collapse to summary beyond this).
    rng_seed : int
        Seed for subsampling.
    ax : Axes, optional
        Matplotlib Axes to draw on (creates a new figure if None).
    save_path : Path, optional
        If provided, save the figure to this path.
    dpi : int
        Resolution for saved figure.

    Returns
    -------
    fig : matplotlib Figure
    """
    rng = np.random.default_rng(rng_seed)

    # -- Resolve weights ----------------------------------------------------
    if isinstance(weights, dict):
        weight_dict = {k: float(v) for k, v in weights.items() if float(v) > 0}
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if len(w) != len(manifest):
            raise ValueError(
                f"Weight vector has {len(w)} entries, but manifest has "
                f"{len(manifest)} mixtures. Expected 140."
            )
        # Build ordered index
        idx_map = build_mixture_index(manifest)
        weight_dict = {}
        for i, wi in enumerate(w):
            if wi > 0:
                weight_dict[idx_map[i]] = float(wi)

    if not weight_dict:
        raise ValueError("No positive weights in the weight vector.")

    # Normalize
    total_w = sum(weight_dict.values())
    weight_dict = {k: v / total_w for k, v in weight_dict.items()}

    # -- Sample from each active mixture ------------------------------------
    active_labels = sorted(weight_dict.keys())
    n_active = len(active_labels)

    # Assign colors — always use vibrant HSV-generated palette
    colors = _generate_colors(n_active)

    # Determine per-mixture sample counts
    sample_counts = {}
    remaining = total_samples
    for i, label in enumerate(active_labels[:-1]):
        n = max(1, int(total_samples * weight_dict[label]))
        sample_counts[label] = n
        remaining -= n
    sample_counts[active_labels[-1]] = max(1, remaining)

    # -- Create figure ------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.set_facecolor("white")

    # Determine bounds from data
    first_data = np.load(list(manifest.values())[0], allow_pickle=True)
    sigma_val = float(first_data.get("sigma", 0.2))
    if layout == "ring" and "radius" in first_data:
        half_extent = float(first_data["radius"])
    else:
        # Compute half-extent from stored means (works for both ring and grid)
        means = first_data["means"]
        half_extent = float(np.max(np.abs(means))) if len(means) > 0 else 1.0
    margin = half_extent + 4 * sigma_val

    legend_handles = []

    for i, label in enumerate(active_labels):
        npz_path = manifest[label]
        data = np.load(npz_path, allow_pickle=True)
        all_samples = data["samples"]
        n_avail = len(all_samples)
        n_needed = sample_counts[label]

        # Subsample
        if n_needed >= n_avail:
            idx = np.arange(n_avail)
        else:
            idx = rng.choice(n_avail, size=n_needed, replace=False)
        samples = all_samples[idx]

        color = colors[i % len(colors)]

        ax.scatter(
            samples[:, 0], samples[:, 1],
            s=point_size, alpha=alpha, c=color,
            rasterized=True, zorder=2,
        )

        # Legend entry: show mixture type + indices + weight
        if legend and (n_active <= legend_max_entries):
            w = weight_dict[label]
            short = _short_label(label)
            legend_handles.append(
                Patch(color=color, alpha=0.7,
                      label=f"{short}  (w={w:.3f})")
            )

    # -- Overlay decorations ------------------------------------------------
    if show_means:
        # Collect all unique (component_index, mean) pairs from active mixtures
        seen_indices: set[int] = set()
        comp_info: list[tuple[int, tuple[float, float]]] = []
        for label in active_labels:
            data = np.load(manifest[label], allow_pickle=True)
            indices = data["component_indices"]
            means_arr = data["means"]
            for i in range(len(indices)):
                ci = int(indices[i])
                if ci not in seen_indices:
                    seen_indices.add(ci)
                    mx, my = float(means_arr[i, 0]), float(means_arr[i, 1])
                    comp_info.append((ci, (mx, my)))

        for k, (mx, my) in sorted(comp_info, key=lambda x: x[0]):
            ax.plot(mx, my, "X", color=COMPONENT_MEAN_COLOR,
                    markersize=10, markeredgewidth=1.5, zorder=10)
            ax.annotate(
                str(k), (mx, my),
                textcoords="offset points", xytext=(6, 6),
                fontsize=8, fontweight="bold", color=COMPONENT_MEAN_COLOR,
                zorder=11,
            )

    if show_circle and layout == "ring":
        circle = plt.Circle(
            (0, 0), half_extent, fill=False,
            edgecolor=REFERENCE_CIRCLE_COLOR,
            linewidth=0.8, linestyle="--", alpha=0.6, zorder=0,
        )
        ax.add_patch(circle)

    # -- Labels & title -----------------------------------------------------
    if title is None:
        if n_active == 1:
            title = f"Mixture: {_short_label(active_labels[0])}"
        else:
            title = (
                f"Weighted Mixture Selection — "
                f"{n_active} active / {len(manifest)} candidates"
            )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("x₁", fontsize=11)
    ax.set_ylabel("x₂", fontsize=11)
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_aspect("equal")

    # -- Legend -------------------------------------------------------------
    if legend and legend_handles:
        if n_active <= 10:
            ax.legend(
                handles=legend_handles, loc="upper left",
                framealpha=0.85, fontsize=8,
                bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
            )
        else:
            ax.legend(
                handles=legend_handles, loc="upper left",
                framealpha=0.85, fontsize=7, ncol=2,
                bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
            )

    # -- Grid & spines ------------------------------------------------------
    ax.grid(True, alpha=0.25, color=GRID_COLOR, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
        spine.set_linewidth(0.8)

    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, facecolor="white", bbox_inches="tight")
        plt.close(fig)

    return fig


# ======================================================================
# Reference plot — all 16 base Gaussians with labels
# ======================================================================

def plot_reference(
    manifest: dict[str, Path],
    figsize: tuple[float, float] = (12, 12),
    point_size: float = 1.0,
    alpha: float = 0.40,
    total_samples: int = 16000,
    rng_seed: int = 42,
    layout: str = "ring",
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Reference plot: all 16 base Gaussians, each in a distinct color, with X
    markers so we can identify which component is which (0..15).

    This is a special case of ``plot_mixture_selection`` with equal weights
    over all 16 Type-1 mixtures and ``show_means=True``.
    """
    # Build equal-weight vector over all 16 Type-1 mixtures
    type1_labels = sorted(
        [k for k in manifest if k.startswith("type1_")],
        key=lambda x: int(x.split("_")[1]),
    )
    weight_dict = {label: 1.0 / 16 for label in type1_labels}

    return plot_mixture_selection(
        weights=weight_dict,
        manifest=manifest,
        total_samples=total_samples,
        figsize=figsize,
        alpha=alpha,
        point_size=point_size,
        show_means=True,
        show_circle=True,
        layout=layout,
        title="Reference — All Base Gaussians" if layout == "grid" else "Reference — All 16 Base Gaussians on the Ring",
        legend=True,
        rng_seed=rng_seed,
        save_path=save_path,
        dpi=dpi,
    )


# ======================================================================
# Convenience: predefined interesting weight vectors
# ======================================================================

def demo_weight_vectors(manifest: dict[str, Path]) -> dict[str, np.ndarray]:
    """Return a dictionary of interesting weight vectors for demonstration.

    Returns
    -------
    dict[str, ndarray]
        Keys are human-readable descriptions, values are 140-dim weight vectors.
    """
    idx_map = build_mixture_index(manifest)
    label_to_idx = {v: k for k, v in idx_map.items()}
    n_total = len(manifest)

    demos: dict[str, np.ndarray] = {}

    # 1. Single Type-1 (g0)
    w = np.zeros(n_total)
    w[label_to_idx["type1_0"]] = 1.0
    demos["Single Type-1 (g0)"] = w

    # 2. Single Type-2 (g0 + g8, opposite sides)
    w = np.zeros(n_total)
    w[label_to_idx["type2_0_8"]] = 1.0
    demos["Single Type-2 (g0, g8 opposite)"] = w

    # 3. Single Type-3 (quartet 0-3)
    w = np.zeros(n_total)
    w[label_to_idx["type3_0_1_2_3"]] = 1.0
    demos["Single Type-3 (quartet 0-3)"] = w

    # 4. All 4 Type-3 equally weighted (the 4-sparse optimum)
    w = np.zeros(n_total)
    for qi in ["type3_0_1_2_3", "type3_4_5_6_7",
               "type3_8_9_10_11", "type3_12_13_14_15"]:
        w[label_to_idx[qi]] = 0.25
    demos["All 4 Type-3 (4-sparse optimum)"] = w

    # 5. All 16 Type-1 equally weighted (the RKE optimum)
    w = np.zeros(n_total)
    for k in range(16):
        w[label_to_idx[f"type1_{k}"]] = 1.0 / 16
    demos["All 16 Type-1 (RKE optimum)"] = w

    # 6. Mixed: one of each type
    w = np.zeros(n_total)
    w[label_to_idx["type1_0"]] = 0.2
    w[label_to_idx["type2_4_12"]] = 0.3
    w[label_to_idx["type3_8_9_10_11"]] = 0.5
    demos["Mixed: Type-1 + Type-2 + Type-3"] = w

    # 7. Sparse random: 5 random mixtures
    w = np.zeros(n_total)
    rng = np.random.default_rng(42)
    chosen = rng.choice(n_total, size=5, replace=False)
    rand_weights = rng.dirichlet(np.ones(5))
    for i, wi in zip(chosen, rand_weights):
        w[i] = wi
    demos["Sparse random (5 mixtures)"] = w

    return demos


# ======================================================================
# Grid demo weight vectors
# ======================================================================

def grid_demo_weight_vectors(manifest: dict[str, Path]) -> dict[str, np.ndarray]:
    """Return interesting weight vectors for the grid layout.

    Returns
    -------
    dict[str, ndarray]
        Keys are human-readable descriptions, values are 265-dim weight vectors.
    """
    idx_map = build_mixture_index(manifest)
    label_to_idx = {v: k for k, v in idx_map.items()}
    n_total = len(manifest)

    demos: dict[str, np.ndarray] = {}

    # 1. Single Type-1 (corner gaussian)
    w = np.zeros(n_total)
    w[label_to_idx["type1_0_0"]] = 1.0
    demos["Single Type-1 (top-left corner)"] = w

    # 2. Single Type-2 (2x2 block at top-left)
    w = np.zeros(n_total)
    w[label_to_idx["type2_0_1_0_1"]] = 1.0
    demos["Single Type-2 (2x2 block top-left)"] = w

    # 3. Single Type-3 (first 3x3 block)
    w = np.zeros(n_total)
    w[label_to_idx["type3_block_0"]] = 1.0
    demos["Single Type-3 (3x3 block 0)"] = w

    # 4. All 4 Type-3 equally weighted (the 4-sparse optimum)
    w = np.zeros(n_total)
    for bi in range(4):
        w[label_to_idx[f"type3_block_{bi}"]] = 0.25
    demos["All 4 Type-3 (4-sparse optimum)"] = w

    # 5. All 36 Type-1 equally weighted (the RKE optimum)
    w = np.zeros(n_total)
    type1_labels = sorted(
        [k for k in manifest if k.startswith("type1_")],
        key=lambda x: (int(x.split("_")[1]), int(x.split("_")[2])),
    )
    for lbl in type1_labels:
        w[label_to_idx[lbl]] = 1.0 / len(type1_labels)
    demos["All 36 Type-1 (RKE optimum)"] = w

    # 6. Mixed: one of each type
    w = np.zeros(n_total)
    w[label_to_idx["type1_3_3"]] = 0.2
    w[label_to_idx["type2_2_4_2_4"]] = 0.3
    w[label_to_idx["type3_block_2"]] = 0.5
    demos["Mixed: Type-1 + Type-2 + Type-3"] = w

    # 7. Sparse random: 5 random mixtures
    w = np.zeros(n_total)
    rng = np.random.default_rng(42)
    chosen = rng.choice(n_total, size=5, replace=False)
    rand_weights = rng.dirichlet(np.ones(5))
    for i, wi in zip(chosen, rand_weights):
        w[i] = wi
    demos["Sparse random (5 mixtures)"] = w

    return demos


# ======================================================================
# Helpers
# ======================================================================

def _short_label(label: str) -> str:
    """Convert a mixture label to a compact display form."""
    parts = label.split("_")
    mtype = parts[0]  # type1, type2, type3
    indices = parts[1:]
    idx_str = ",".join(indices)
    return f"{mtype}[{idx_str}]"


