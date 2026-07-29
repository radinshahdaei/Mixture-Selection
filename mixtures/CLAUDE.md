# Mixture Selection — Agentic Context

**Do not inspect images.** Figures are generated as PNG files — trust that they
render correctly and never attempt to Read or display them.

Research project for sparse mixture selection on Gaussian mixtures. Three layouts:

**Ring** (default): 16 standard 2D Gaussians placed uniformly on a circle (radius R,
isotropic covariance σ²I). Three candidate mixture families, all equally weighted:

- **Type 1** (16 candidates): single base Gaussian.
- **Type 2** (120 candidates): all two-component pairs — C(16,2).
- **Type 3** (4 candidates): four-component disjoint quartets — partition 0..3,
  4..7, 8..11, 12..15.
- **Total**: 140 candidate mixtures.

**Grid**: 36 Gaussians on a 6×6 lattice. Three candidate mixture families:

- **Type 1** (36 candidates): single base Gaussian.
- **Type 2** (225 candidates): four-component blocks — choose 2 rows + 2 cols,
  C(6,2)² intersections.
- **Type 3** (4 candidates): nine-component disjoint 3×3 blocks.
- **Total**: 265 candidate mixtures (2D isotropic only).

**Random 3D**: 20 Gaussians in 3D with random means and full PSD covariances:

- **Type 1** (20 candidates): single base Gaussian. No mixtures — each candidate
  is a standalone Gaussian.
- Means: uniform in [-scale, scale]³ (default scale=2.0).
- Covariances: Wishart-like A @ Aᵀ where A ~ N(0, cov_scale, (3,3)).
- **No visualization** — data generation only.

The RKE optimum is the uniform mixture over all base Gaussians. Under a four-sparse
constraint the only valid representation uses the four Type-3 mixtures.

## Project layout

```
mixtures/
  config.yaml               # Default config (ring)
  config_ring.yaml          # Ring layout config
  config_grid.yaml          # Grid layout config
  config_random_3d.yaml     # Random 3D layout config
  src/
    config.py               # Config dataclass + YAML loader
    gmm.py                  # BaseGaussian (isotropic or full-cov) + Mixture (sample, metadata)
    mixtures.py             # RingMixtureFactory — builds all 140 ring candidates
    grid_mixtures.py        # GridMixtureFactory — builds all 265 grid candidates
    random_mixtures.py      # RandomMixtureFactory — builds 20 random 3D Gaussians
    sampling.py             # SampleManager — generate + .npz I/O + manifest
    visualize.py            # plot_mixture_selection(), plot_reference(), demo vectors
  scripts/
    generate_samples.py     # Create all .npz files (140 or 265 depending on layout)
    visualize.py            # CLI: --demo, --preset, --reference, --weights
    validate_random_3d.py   # Validate random_3d data integrity
  data/samples/{ring,grid,random_3d}/  # .npz files + manifest.json
  figures/{ring,grid}/      # Output plots (2D only)
```

## Environment

Uses conda: `conda activate mixture-selection`. Python 3.11, dependencies in
`requirements.txt` (numpy, scipy, matplotlib, pyyaml, tqdm).

## Key workflows

```bash
cd mixtures

# Generate samples (use dedicated config files)
python scripts/generate_samples.py --config config_ring.yaml       # 140 → data/samples/ring/
python scripts/generate_samples.py --config config_grid.yaml       # 265 → data/samples/grid/
python scripts/generate_samples.py --config config_random_3d.yaml  # 20 → data/samples/random_3d/

# Visualize (2D only)
python scripts/visualize.py --config config_ring.yaml --demo --reference
python scripts/visualize.py --config config_grid.yaml --demo --reference

# Validate random_3d data
python scripts/validate_random_3d.py
```

## Core API

```python
from src.visualize import plot_mixture_selection
# w: simplex weight vector (140-dim ring, 265-dim grid, 20-dim random_3d)
# manifest: dict[label → Path] from SampleManager.load_manifest()
plot_mixture_selection(weights=w, manifest=manifest, layout="ring", save_path="out.png")
```

## Visualization conventions

- White background, vibrant HSV-generated per-mixture colors.
- Component mean X-markers are off by default; only shown on the reference plot.
- All plots built **from samples** (.npz), never from analytical PDF.
- Sigma default is 0.08 — tight clusters for visual distinguishability.
- Reference circle drawn for ring layout only; not drawn for grid.
- Visualization is 2D only — random_3d is data-generation only.

# Commit Conventions

- Commits are co-authored with Claude — every commit includes: `Co-Authored-By: Claude <noreply@anthropic.com>`
- Subjects are imperative, lowercase, and concise
- Bodies explain what and why; bullet points for changes
- Should read as a clean, natural progression
