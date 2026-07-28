# Mixture Selection — Developer Guide

## Overview

Research project for **sparse mixture selection** on Gaussian mixture models. Two
layouts are supported: a ring of 16 Gaussians and a 6×6 grid of 36 Gaussians.
Each layout defines 140 or 265 candidate mixtures of three types. The goal is to
study how optimization algorithms prune or select among these candidates under
sparsity constraints.

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

All mixtures are equally weighted internally.

### Theoretical Properties

- The **RKE optimum** is the uniform mixture over all base Gaussians (16 for
  ring, 36 for grid). It admits many equivalent decompositions: Type-1 singles,
  Type-2 combos, or any combination producing uniform mass.
- Under a **four-sparse constraint**, the only valid representation uses the
  four Type-3 mixtures in either layout.

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

All parameters live in `config.yaml`. Set `gmm.layout` to `"ring"` or `"grid"`
to select the geometry.

```yaml
gmm:
  layout: "ring"            # "ring" | "grid"

  sigma: 0.08               # Std dev (covariance = sigma²·I)
  random_seed: 42

  # --- Ring layout ---
  n_components: 16          # Number of base Gaussians on the circle
  radius: 1.0               # Circle radius for component means

  # --- Grid layout ---
  grid_rows: 6              # Rows in the 2D lattice
  grid_cols: 6              # Columns in the 2D lattice
  grid_spacing: 0.4         # Spacing between adjacent grid points

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
             src/mixtures.py  ──►  RingMixtureFactory
             src/grid_mixtures.py ─► GridMixtureFactory
                       │              │
                       │    BaseGaussian × 16 on ring / 36 on grid
                       │    Mixture × 140 (ring) / 265 (grid)
                       │
                       ▼
             src/sampling.py ──►  SampleManager
                       │
                       │    .generate(mixture) → (5000, 2) ndarray
                       │    .save(samples, mixture, path) → .npz
                       │    .generate_all() → manifest.json
                       │
                       ▼
          data/samples/{ring,grid}/type*/  .npz files + manifest.json
                       │
                       ▼
             src/visualize.py ──►  plot_mixture_selection()
                                   plot_reference()
                                   demo_weight_vectors()
                                   grid_demo_weight_vectors()
                       │
                       ▼
              figures/{ring,grid}/   .png output
```

### Key Classes

**`BaseGaussian`** (`gmm.py`) — Single 2D isotropic Gaussian.
- `mean` (2,), `sigma` (float), `index` (int)
- `cov` → σ²I, `sample(n, rng)` → (n, 2)

**`Mixture`** (`gmm.py`) — Weighted combination of `BaseGaussian`s.
- `components: list[tuple[float, BaseGaussian]]`
- `weights`, `component_indices`, `means`, `sigma`
- `sample(n, rng)` — choose component by weight, then draw from Gaussian
- `metadata()` — dict for .npz serialization

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

**`SampleManager`** (`sampling.py`) — Sample generation and .npz I/O.
- `generate(mixture, n)` → ndarray
- `save(samples, mixture, path)` — self-describing .npz
- `load(path)` → dict of arrays
- `generate_all(mixtures)` → manifest dict + manifest.json (in layout subdirectory)
- `load_manifest(dir)` → label → path mapping

**Visualization** (`visualize.py`) — Mixture-of-mixtures plotting.
- `plot_mixture_selection(weights, manifest, layout, ...)` — main function
- `plot_reference(manifest, layout, ...)` — all base Gaussians with X markers
- `build_mixture_index(manifest)` → simplex dimension → label
- `demo_weight_vectors(manifest)` → ring-specific predefined weight vectors
- `grid_demo_weight_vectors(manifest)` → grid-specific predefined weight vectors

### .npz File Format

Each file is self-describing:

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `samples` | float32 | (5000, 2) | i.i.d. draws |
| `mixture_type` | int32 | scalar | 1, 2, or 3 |
| `component_indices` | int32 | (K,) | Base indices used |
| `weights` | float64 | (K,) | Mixture weights |
| `means` | float64 | (K, 2) | Component means |
| `sigma` | float64 | scalar | Shared sigma |
| `label` | str | scalar | e.g. "type2_3_7" or "type2_0_1_2_3" |
| `n_components_base` | int32 | scalar | From config (always `n_components`, even for grid) |
| `radius` | float64 | scalar | From config (ring radius; present but unused for grid) |

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

Use `build_mixture_index(manifest)` to get the exact mapping for any layout.

### Label Convention

| Layout | Type 1 | Type 2 | Type 3 |
|--------|--------|--------|--------|
| Ring | `type1_0` | `type2_3_7` | `type3_0_1_2_3` |
| Grid | `type1_2_3` | `type2_0_1_2_3` | `type3_block_0` |

---

## Usage

### Generate Samples

```bash
# Ring layout (default): 140 candidates → data/samples/ring/
python scripts/generate_samples.py

# Grid layout: change gmm.layout to "grid" in config.yaml, then:
python scripts/generate_samples.py                 # 265 candidates → data/samples/grid/

# Custom config
python scripts/generate_samples.py --config custom.yaml
```

Samples are saved under `data/samples/{layout}/type{1,2,3}/` with a
`manifest.json` in the layout subdirectory.

### Visualize

```bash
# Demo — all interesting weight vectors (auto-detects layout from config)
python scripts/visualize.py --demo

# Reference plot — all base Gaussians with labels
python scripts/visualize.py --reference

# Built-in presets (work for both ring and grid layouts)
python scripts/visualize.py --preset four-sparse
python scripts/visualize.py --preset rke-optimum
python scripts/visualize.py --preset type1-only
python scripts/visualize.py --preset type2-only
python scripts/visualize.py --preset type3-only
python scripts/visualize.py --preset mixed
python scripts/visualize.py --preset single-type1
python scripts/visualize.py --preset single-type2
python scripts/visualize.py --preset single-type3

# Custom weight vector from file
python scripts/visualize.py --weights my_weights.npy
python scripts/visualize.py --weights my_weights.npz
```

### Programmatic API

```python
import numpy as np
from pathlib import Path
from src.config import load_config
from src.sampling import SampleManager
from src.visualize import plot_mixture_selection, plot_reference, build_mixture_index

config = load_config()
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
if layout == "ring":
    for qi in ["type3_0_1_2_3", "type3_4_5_6_7",
               "type3_8_9_10_11", "type3_12_13_14_15"]:
        w[label_to_idx[qi]] = 0.25
else:
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

---

## Commit Conventions

- Commits are co-authored with Claude — every commit includes:
  `Co-Authored-By: Claude <noreply@anthropic.com>`
- Subjects are imperative, lowercase, and concise
- Bodies explain what and why; bullet points for changes
- Should read as a clean, natural progression
