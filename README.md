# Mixture Selection

Sparse mixture selection on Gaussian mixture models. Base Gaussians are placed on
either a **ring** (16 components) or a **grid** (36 components on a 6×6 lattice).
Three families of equally-weighted candidate mixtures are defined per layout,
studied under sparsity constraints.

## Quick start

```bash
conda create -n mixture-selection python=3.11 -y && conda activate mixture-selection
pip install -r requirements.txt

# Default layout (ring): 140 candidates × 5000 samples → data/samples/ring/
python scripts/generate_samples.py

# For grid layout, change gmm.layout to "grid" in config.yaml, then:
python scripts/generate_samples.py           # 265 candidates → data/samples/grid/

python scripts/visualize.py --demo           # all interesting weight vectors
python scripts/visualize.py --reference      # all base Gaussians with labels
```

## Layouts

| Layout  | Bases | Type 1 | Type 2 | Type 3 | Total |
|---------|-------|--------|--------|--------|-------|
| Ring    |    16 |     16 |    120 |      4 | **140** |
| Grid    |    36 |     36 |    225 |      4 | **265** |

- **Ring**: 16 Gaussians on a circle of radius R=1.0. Type 2 = all C(16,2) pairs;
  Type 3 = four disjoint consecutive quartets.
- **Grid**: 36 Gaussians on a 6×6 lattice. Type 2 = all C(6,2)×C(6,2) 2-row,
  2-col intersection blocks; Type 3 = four disjoint 3×3 blocks.

The RKE optimum is the uniform mixture over all base Gaussians. Under a
four-sparse constraint the only valid representation uses the four Type-3
mixtures in either layout.

## Configuration

All parameters live in `config.yaml` — set `gmm.layout` to `"ring"` or `"grid"`
to select the geometry. See [DEVELOPER.md](DEVELOPER.md) for the full schema.

## Visualization API

```python
from src.visualize import plot_mixture_selection
# w is a simplex vector over all candidate mixtures (140 or 265 dims)
plot_mixture_selection(weights=w, manifest=manifest, layout="ring", save_path="out.png")
```

See [DEVELOPER.md](DEVELOPER.md) for the full developer guide.
