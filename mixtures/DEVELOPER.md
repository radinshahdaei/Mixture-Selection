# Mixture Selection — Developer Guide

## Overview

Research project for **sparse mixture selection** on Gaussian mixture models.
Three layouts are supported: a ring of 16 Gaussians, a 6×6 grid of 36 Gaussians,
and a random 3D layout with full PSD covariances. Each layout defines 140 or 265
candidate mixtures of three types. The goal is to study how optimization
algorithms prune or select among these candidates under sparsity constraints.

### Layouts

#### Ring Layout (`gmm.layout: "ring"`)

- **N = 16** isotropic 2D Gaussians with shared covariance σ²I.
- Component *k* has mean at angle θₖ = 2πk/16 on a circle of radius R:
  μₖ = R · (cos θₖ, sin θₖ).
- Defaults: R = 1.0, σ = 0.08.

| Type | Count | Description |
|------|-------|-------------|
| Type 1 | 16 | Single base Gaussian (one per component) |
| Type 2 | 120 | All two-component equally-weighted pairs C(16,2) |
| Type 3 | 4 | Four-component equally-weighted disjoint quartets |
| **Total** | **140** | |

#### Grid Layout (`gmm.layout: "grid"`)

- **N = 36** isotropic 2D Gaussians on a 6×6 lattice, centered at origin.
- Component at row *r*, col *c* has mean at:
  μ = (c·spacing − x_offset, r·spacing − y_offset) where offsets center the grid.
- Defaults: 6×6, spacing = 0.4, σ = 0.08.

| Type | Count | Description |
|------|-------|-------------|
| Type 1 | 36 | Single base Gaussian (one per grid point) |
| Type 2 | 225 | Four-component blocks: pick 2 rows × 2 cols = C(6,2)² = 225 |
| Type 3 | 4 | Nine-component disjoint 3×3 blocks partitioning the 6×6 grid |
| **Total** | **265** | |

#### Random 3D Layout (`gmm.layout: "random_3d"`)

- **N = 20** 3D Gaussians with **full random PSD covariances** (not isotropic).
- Means drawn uniformly from [-scale, scale]³. Default scale = 2.0.
- Covariances generated via Wishart-like construction: A @ Aᵀ where
  A ~ N(0, cov_scale, (3,3)). Default cov_scale = 1.0.
- Unlike ring/grid, this layout produces **only Type 1** (single Gaussians).
  No Type-2 pairs or Type-3 quartets — just N standalone Gaussian candidates.
- **No visualization support** — the visualization code is 2D-only.

| Type | Count | Description |
|------|-------|-------------|
| Type 1 | 20 | Single base Gaussian (one per random component) |
| **Total** | **20** | |

All mixtures are equally weighted internally.

### Theoretical Properties

- The **RKE optimum** is the uniform mixture over all base Gaussians (16 for
  ring, 20 for random_3d, 36 for grid). It admits many equivalent decompositions:
  Type-1 singles, Type-2 combos, or any combination producing uniform mass.
- Under a **four-sparse constraint**, the only valid representation uses the
  four Type-3 mixtures in any layout.

---

## Setup

### Prerequisites

- Conda (Miniconda or Anaconda)
- Python ≥ 3.10

### Installation

```bash
# Clone and enter the repo
git clone <repo-url> && cd Mixture-Selection

# Create conda environment
conda create -n mixture-selection python=3.11 -y
conda activate mixture-selection

# Install dependencies
pip install -r requirements.txt
```

---

## Configuration

Each layout has a dedicated config file for convenience:

| File | Layout |
|------|--------|
| `config.yaml` | Ring (default) |
| `config_ring.yaml` | Ring |
| `config_grid.yaml` | Grid |
| `config_random_3d.yaml` | Random 3D |

All share the same schema — only `gmm.layout` and layout-specific parameters differ.

```yaml
gmm:
  layout: "ring"            # "ring" | "grid" | "random_3d"

  sigma: 0.08               # Std dev (covariance = sigma²·I; unused for random_3d)
  random_seed: 42

  # --- Ring layout ---
  n_components: 16          # Number of base Gaussians on the circle
  radius: 1.0               # Circle radius for component means

  # --- Grid layout ---
  grid_rows: 6              # Rows in the 2D lattice
  grid_cols: 6              # Columns in the 2D lattice
  grid_spacing: 0.4         # Spacing between adjacent grid points

  # --- Random 3D layout ---
  random_3d_scale: 2.0      # Means uniform in [-scale, scale]^3
  random_3d_cov_scale: 1.0  # Std of entries in A where cov = A @ A^T

sampling:
  n_samples: 5000           # Samples per mixture
  output_dir: "data/samples"  # Layout subdirectory appended automatically

visualization:
  style: "light"            # "light" | "dark"
  dpi: 150
  figsize: [10, 10]
  figure_dir: "figures"
  total_samples: 8000       # Points rendered in weighted-mixture view
  scatter_alpha: 0.35
  scatter_point_size: 1.2
```

---

## Architecture

### Module Map

```
config.yaml  ──►  src/config.py  ──►  Config dataclass
                       │
                       ▼
             src/mixtures.py        ──►  RingMixtureFactory
             src/grid_mixtures.py   ──►  GridMixtureFactory
             src/random_mixtures.py ──►  RandomMixtureFactory
                       │              │
                       │    BaseGaussian × 16 (ring/random) or 36 (grid)
                       │    Mixture × 140 (ring/random) or 265 (grid)
                       │
                       ▼
             src/sampling.py ──►  SampleManager
                       │
                       │    .generate(mixture) → (5000, D) ndarray
                       │    .save(samples, mixture, path) → .npz
                       │    .generate_all() → manifest.json
                       │
                       ▼
    data/samples/{ring,grid,random_3d}/type*/  .npz files + manifest.json
                       │
                       ▼
             src/visualize.py ──►  plot_mixture_selection()  [2D only]
                                   plot_reference()
                                   demo_weight_vectors()
                                   grid_demo_weight_vectors()
                       │
                       ▼
              figures/{ring,grid}/   .png output
```

### Key Classes

**`BaseGaussian`** (`gmm.py`) — Single multivariate Gaussian.
- `mean` (D,), `sigma` (float | None), `cov` (D×D, from sigma²·I or provided), `index` (int)
- Supports both isotropic (scalar sigma) and full-covariance construction.
- `ndim` property — auto-detected from `mean` length.
- `sample(n, rng)` → (n, D) — delegates to `multivariate_normal`, dimension-agnostic.

**`Mixture`** (`gmm.py`) — Weighted combination of `BaseGaussian`s.
- `components: list[tuple[float, BaseGaussian]]`
- `weights`, `component_indices`, `means`, `sigma` (None for full-cov components), `ndim`
- `sample(n, rng)` — choose component by weight, then draw from Gaussian
- `metadata()` — stores `sigma` for isotropic, `covariances` for full-cov

**`RingMixtureFactory`** (`mixtures.py`) — Builds all 140 ring candidates.
- `create_type1()` → 16 single-component mixtures
- `create_type2()` → 120 pair mixtures (all C(16,2) combinations)
- `create_type3()` → 4 quartet mixtures (consecutive disjoint groups of 4)
- `create_all()` → dict[label → Mixture]
- `create_uniform_superposition()` → all 16 bases equally weighted

**`GridMixtureFactory`** (`grid_mixtures.py`) — Builds all 265 grid candidates.
- `create_type1()` → 36 single-component mixtures (one per grid cell)
- `create_type2()` → 225 four-component mixtures (choose 2 rows + 2 columns,
  the 4 intersection points, all C(6,2)² combinations)
- `create_type3()` → 4 nine-component mixtures (disjoint 3×3 blocks
  partitioning the grid)
- `create_all()` → dict[label → Mixture]
- `create_uniform_superposition()` → all 36 bases equally weighted

**`RandomMixtureFactory`** (`random_mixtures.py`) — Builds N standalone 3D Gaussians.
- `_create_base_gaussians()` → N Gaussians with random 3D means (uniform in
  [-scale, scale]³) and random PSD covariances (A @ Aᵀ with A ~ N(0, cov_scale)).
- `create_all()` → N single-Gaussian candidates
- `create_uniform_superposition()` → all N bases equally weighted

**`SampleManager`** (`sampling.py`) — Sample generation and .npz I/O.
- `generate(mixture, n)` → ndarray
- `save(samples, mixture, path)` — self-describing .npz, layout-aware metadata
- `load(path)` → dict of arrays
- `generate_all(mixtures)` → manifest dict + manifest.json (in layout subdirectory)
- `load_manifest(dir)` → label → path mapping

**Visualization** (`visualize.py`) — 2D mixture-of-mixtures plotting.
- `plot_mixture_selection(weights, manifest, layout, ...)` — main function
- `plot_reference(manifest, layout, ...)` — all base Gaussians with X markers
- `build_mixture_index(manifest)` → simplex dimension → label
- `demo_weight_vectors(manifest)` → ring-specific predefined weight vectors
- `grid_demo_weight_vectors(manifest)` → grid-specific predefined weight vectors

### .npz File Format

Each file is self-describing:

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `samples` | float32 | (5000, D) | i.i.d. draws (D=2 for ring/grid, D=3 for random_3d) |
| `mixture_type` | int32 | scalar | 1, 2, or 3 |
| `component_indices` | int32 | (K,) | Base indices used |
| `weights` | float64 | (K,) | Mixture weights |
| `means` | float64 | (K, D) | Component means |
| `sigma` | float64 | scalar | Shared sigma (ring/grid only) |
| `covariances` | float64 | (K, D, D) | Full cov matrices (random_3d only) |
| `label` | str | scalar | e.g. "type2_3_7" or "type2_0_1_2_3" |
| `layout` | str | scalar | "ring", "grid", or "random_3d" |
| `n_components_base` | int32 | scalar | N (16, 20, or 36) |
| `radius` | float64 | scalar | Circle radius (ring only) |
| `scale` | float64 | scalar | Hypercube half-extent (random_3d only) |

### Simplex Index Ordering

The weight vector maps to mixtures in the order: Type 1, then Type 2, then
Type 3. Within each type, labels are sorted by their integer components.

**Ring (140-dim):**
1. **Indices 0..15**: Type 1 — g₀, g₁, ..., g₁₅
2. **Indices 16..135**: Type 2 — all C(16,2) pairs in lexicographic order
3. **Indices 136..139**: Type 3 — quartets (0,1,2,3), (4,5,6,7), (8,9,10,11), (12,13,14,15)

**Grid (265-dim):**
1. **Indices 0..35**: Type 1 — g₀,₀, g₀,₁, ..., g₅,₅ (row-major)
2. **Indices 36..260**: Type 2 — all 225 2×2 blocks in lexicographic (row pair, col pair) order
3. **Indices 261..264**: Type 3 — blocks 0, 1, 2, 3 (row-major over the 2×2 block grid)

**Random 3D (16-dim):** Same as ring — 16 Type-1 labels, no Type-2 or Type-3.

Use `build_mixture_index(manifest)` to get the exact mapping for any layout.

### Label Convention

| Layout | Type 1 | Type 2 | Type 3 |
|--------|--------|--------|--------|
| Ring | `type1_0` | `type2_3_7` | `type3_0_1_2_3` |
| Grid | `type1_2_3` | `type2_0_1_2_3` | `type3_block_0` |
| Random 3D | `type1_0` | — | — |

---

## Usage

### Generate Samples

```bash
# Each layout has a dedicated config file
python scripts/generate_samples.py --config config_ring.yaml       # 140 candidates
python scripts/generate_samples.py --config config_grid.yaml       # 265 candidates
python scripts/generate_samples.py --config config_random_3d.yaml  # 20 candidates (3D singles)
```

Samples are saved under `data/samples/{layout}/type{1,2,3}/` with a
`manifest.json` in the layout subdirectory.

### Validate Random 3D Data

```bash
python scripts/validate_random_3d.py
```

Runs 26 checks across 7 categories: manifest integrity, file existence,
sample shapes, means bounds, covariance validity (symmetry, PSD, non-isotropic),
reproducibility (same seed → same data), and mixture structure.

### Visualize

```bash
# Ring figures (2D only)
python scripts/visualize.py --config config_ring.yaml --demo --reference

# Grid figures (2D only)
python scripts/visualize.py --config config_grid.yaml --demo --reference

# Note: random_3d has no visualization (the script refuses with an error)
```

Built-in presets: `four-sparse`, `rke-optimum`, `type1-only`, `type2-only`,
`type3-only`, `mixed`, `single-type1`, `single-type2`, `single-type3`.

### Programmatic API

```python
import numpy as np
from pathlib import Path
from src.config import load_config
from src.sampling import SampleManager
from src.visualize import plot_mixture_selection, plot_reference, build_mixture_index

config = load_config("config_ring.yaml")
layout = config.gmm.layout
samples_dir = Path(config.sampling.output_dir) / layout
manifest = SampleManager.load_manifest(samples_dir)

# Reference plot
plot_reference(manifest, layout=layout, save_path=Path("reference.png"))

# 4-sparse optimum
idx_map = build_mixture_index(manifest)
label_to_idx = {v: k for k, v in idx_map.items()}

n = len(manifest)
w = np.zeros(n)
if layout == "ring" or layout == "random_3d":
    for qi in ["type3_0_1_2_3", "type3_4_5_6_7",
               "type3_8_9_10_11", "type3_12_13_14_15"]:
        w[label_to_idx[qi]] = 0.25
else:  # grid
    for bi in range(4):
        w[label_to_idx[f"type3_block_{bi}"]] = 0.25

plot_mixture_selection(
    weights=w,
    manifest=manifest,
    layout=layout,
    title="4-Sparse Optimum",
    save_path=Path("four_sparse.png"),
)
```

---

## Visualization Design

- **White background** — clean, publication-ready.
- **Vibrant HSV-generated colors** — each active mixture gets a distinct color
  with saturation ≥ 0.9 for maximum distinguishability.
- **Component mean X-markers** — turned off by default; only shown on the
  reference plot (`--reference` or `plot_reference()`).
- **Reference circle** — dashed gray circle at the component mean radius (ring
  layout only; not drawn for grid).
- **All plots built from samples** — `.npz` files are loaded; no analytical PDF
  is ever evaluated. This mirrors real experimental conditions.
- **Legend** — shows mixture label and weight for each active mixture. Placed
  outside the plot area for up to 10 mixtures, two-column for more.
- **2D only** — the random_3d layout has no visualization support.
  `scripts/visualize.py` exits with an error if `layout == "random_3d"`.

---

## Commit Conventions

- Commits are co-authored with Claude — every commit includes:
  `Co-Authored-By: Claude <noreply@anthropic.com>`
- Subjects are imperative, lowercase, and concise
- Bodies explain what and why; bullet points for changes
- Should read as a clean, natural progression
