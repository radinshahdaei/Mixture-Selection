# Mixture Selection — Developer Guide

## Overview

Research project for **sparse mixture selection** on Gaussian mixture models. The
setup places 16 standard 2D Gaussians uniformly on a circle and defines 140
candidate mixtures of three types. The goal is to study how optimization
algorithms prune or select among these candidates under sparsity constraints.

### The Gaussian Ring

- **N = 16** isotropic 2D Gaussians with shared covariance σ²I.
- Component *k* has mean at angle θₖ = 2πk/16 on a circle of radius R:
  μₖ = R · (cos θₖ, sin θₖ).
- Defaults: R = 1.0, σ = 0.08.

### Mixture Types

| Type | Count | Description |
|------|-------|-------------|
| Type 1 | 16 | Single base Gaussian (one per component) |
| Type 2 | 120 | All two-component equally-weighted pairs C(16,2) |
| Type 3 | 4 | Four-component equally-weighted disjoint quartets |
| **Total** | **140** | |

All mixtures are equally weighted internally.

### Theoretical Properties

- The **RKE optimum** is the uniform mixture over all 16 base Gaussians. It
  admits many equivalent decompositions: 16 Type-1 mixtures, 8 Type-2 mixtures,
  or any combination producing uniform mass on the ring.
- Under a **four-sparse constraint**, the only valid representation uses the
  four Type-3 quartets.

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
conda install numpy scipy matplotlib seaborn pyyaml tqdm -y
```

---

## Configuration

All parameters live in `config.yaml`:

```yaml
gmm:
  n_components: 16        # Number of base Gaussians
  radius: 1.0             # Circle radius for component means
  sigma: 0.08             # Std dev (covariance = sigma²·I)
  random_seed: 42

sampling:
  n_samples: 5000         # Samples per mixture
  output_dir: "data/samples"

visualization:
  style: "light"          # "light" | "dark"
  dpi: 150
  figsize: [10, 10]
  figure_dir: "figures"
  total_samples: 8000     # Points to render in weighted-mixture view
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
              src/mixtures.py ──►  MixtureFactory
                        │              │
                        │    BaseGaussian × 16 on the ring
                        │    Mixture × 140 (all three types)
                        │
                        ▼
              src/sampling.py ──►  SampleManager
                        │
                        │    .generate(mixture) → (5000, 2) ndarray
                        │    .save(samples, mixture, path) → .npz
                        │    .generate_all() → manifest.json
                        │
                        ▼
           data/samples/type*/  140 .npz files + manifest.json
                        │
                        ▼
              src/visualize.py ──►  plot_mixture_selection()
                                    plot_reference()
                        │
                        ▼
                  figures/       .png output
```

### Key Classes

**`BaseGaussian`** (`gmm.py`) — Single 2D isotropic Gaussian.
- `mean` (2,), `sigma` (float), `index` (0..15)
- `cov` → σ²I, `sample(n, rng)` → (n, 2)

**`Mixture`** (`gmm.py`) — Weighted combination of `BaseGaussian`s.
- `components: list[tuple[float, BaseGaussian]]`
- `weights`, `component_indices`, `means`, `sigma`
- `sample(n, rng)` — choose component by weight, then draw from Gaussian
- `metadata()` — dict for .npz serialization

**`MixtureFactory`** (`mixtures.py`) — Builds all 140 candidates.
- `create_type1()` → 16 single-component mixtures
- `create_type2()` → 120 pair mixtures (all C(16,2) combinations)
- `create_type3()` → 4 quartet mixtures (consecutive disjoint groups)
- `create_all()` → dict[label → Mixture]
- `create_uniform_superposition()` → all 16 bases equally weighted

**`SampleManager`** (`sampling.py`) — Sample generation and .npz I/O.
- `generate(mixture, n)` → ndarray
- `save(samples, mixture, path)` — self-describing .npz
- `load(path)` → dict of arrays
- `generate_all(mixtures)` → manifest dict + manifest.json
- `load_manifest(dir)` → label → path mapping

**Visualization** (`visualize.py`) — Mixture-of-mixtures plotting.
- `plot_mixture_selection(weights, manifest, ...)` — main function
- `plot_reference(manifest, ...)` — all 16 bases with X markers
- `build_mixture_index(manifest)` → simplex dimension → label
- `demo_weight_vectors(manifest)` → predefined interesting weight vectors

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
| `label` | str | scalar | e.g. "type2_3_7" |
| `n_components_base` | int32 | scalar | N (16) |
| `radius` | float64 | scalar | Circle radius |

### Simplex Index Ordering

The 140-dim weight vector maps to mixtures as:
1. **Indices 0..15**: Type 1 — g₀, g₁, ..., g₁₅
2. **Indices 16..135**: Type 2 — all C(16,2) pairs in lexicographic order (g₀g₁, g₀g₂, ..., g₁₄g₁₅)
3. **Indices 136..139**: Type 3 — quartets (0,1,2,3), (4,5,6,7), (8,9,10,11), (12,13,14,15)

Use `build_mixture_index(manifest)` to get the exact mapping.

---

## Usage

### Generate Samples

```bash
python scripts/generate_samples.py                # default config.yaml
python scripts/generate_samples.py --config custom.yaml
```

Creates 140 `.npz` files in `data/samples/type{1,2,3}/` plus a `manifest.json`.

### Visualize

```bash
# Demo — all interesting weight vectors
python scripts/visualize.py --demo

# Reference plot — all 16 base Gaussians with labels
python scripts/visualize.py --reference

# Built-in presets
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
from src.sampling import SampleManager
from src.visualize import plot_mixture_selection, plot_reference

manifest = SampleManager.load_manifest(Path("data/samples"))

# Reference plot
plot_reference(manifest, save_path=Path("reference.png"))

# 4-sparse optimum — weight vector over 140 candidates
from src.visualize import build_mixture_index
idx_map = build_mixture_index(manifest)
label_to_idx = {v: k for k, v in idx_map.items()}

w = np.zeros(140)
for qi in ["type3_0_1_2_3", "type3_4_5_6_7",
           "type3_8_9_10_11", "type3_12_13_14_15"]:
    w[label_to_idx[qi]] = 0.25

plot_mixture_selection(
    weights=w,
    manifest=manifest,
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
- **Reference circle** — dashed gray circle at the component mean radius.
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
