# Mixture Selection

Sparse mixture selection on Gaussian mixture models. Three layouts are supported:
a **ring** of 16 Gaussians, a 6×6 **grid** of 36 Gaussians, and a **random 3D**
layout with full PSD covariances. Three families of equally-weighted candidate
mixtures are defined per layout, studied under sparsity constraints.

## Quick start

```bash
conda create -n mixture-selection python=3.11 -y && conda activate mixture-selection
pip install -r requirements.txt

# Generate samples (use --config to pick layout)
python scripts/generate_samples.py --config config_ring.yaml       # 140 candidates
python scripts/generate_samples.py --config config_grid.yaml       # 265 candidates
python scripts/generate_samples.py --config config_random_3d.yaml  # 140 candidates (3D)

# Visualize (ring and grid only; random_3d has no viz)
python scripts/visualize.py --config config_ring.yaml --demo --reference
python scripts/visualize.py --config config_grid.yaml --demo --reference

# Validate random_3d data
python scripts/validate_random_3d.py
```

## Layouts

| Layout     | Bases | Type 1 | Type 2 | Type 3 | Total | Dim |
|------------|-------|--------|--------|--------|-------|-----|
| Ring       |    16 |     16 |    120 |      4 | **140** | 2D isotropic |
| Grid       |    36 |     36 |    225 |      4 | **265** | 2D isotropic |
| Random 3D  |    16 |     16 |    120 |      4 | **140** | 3D full cov |

- **Ring**: 16 Gaussians on a circle of radius R=1.0. Type 2 = all C(16,2) pairs;
  Type 3 = four disjoint consecutive quartets.
- **Grid**: 36 Gaussians on a 6×6 lattice. Type 2 = all C(6,2)×C(6,2) 2-row,
  2-col intersection blocks; Type 3 = four disjoint 3×3 blocks.
- **Random 3D**: 16 Gaussians in 3D with means drawn uniformly from [-scale, scale]³
  and full random PSD covariances (A @ Aᵀ). Same Type 1/2/3 structure as ring.
  **No visualization** — data generation only.

The RKE optimum is the uniform mixture over all base Gaussians. Under a
four-sparse constraint the only valid representation uses the four Type-3
mixtures in any layout.

## Configuration

Dedicated config files for each layout:

| File | Layout |
|------|--------|
| `config_ring.yaml` | Ring (default) |
| `config_grid.yaml` | Grid |
| `config_random_3d.yaml` | Random 3D |

All share the same schema — only `gmm.layout` and layout-specific parameters
differ. See [DEVELOPER.md](DEVELOPER.md) for the full schema.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/generate_samples.py` | Generate .npz samples for any layout |
| `scripts/visualize.py` | Plot weighted mixture selections (2D only) |
| `scripts/validate_random_3d.py` | Validate random_3d data integrity |

## Visualization API

```python
from src.visualize import plot_mixture_selection
# w is a simplex vector over all candidate mixtures (140 or 265 dims)
plot_mixture_selection(weights=w, manifest=manifest, layout="ring", save_path="out.png")
```

See [DEVELOPER.md](DEVELOPER.md) for the full developer guide.
