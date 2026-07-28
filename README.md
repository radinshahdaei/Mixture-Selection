# Mixture Selection

Sparse mixture selection on a ring of 16 Gaussian components. Three families of
equally-weighted candidate mixtures — 140 total — used to study how optimization
prunes distributions under sparsity constraints.

## Quick start

```bash
conda create -n mixture-selection python=3.11 -y && conda activate mixture-selection
conda install numpy scipy matplotlib pyyaml tqdm -y

python scripts/generate_samples.py   # 140 × 5000 samples → data/samples/
python scripts/visualize.py --demo   # demo all interesting weight vectors
python scripts/visualize.py --reference  # all 16 bases with labels
```

## Mixture types

| Type   | Count | Description                          |
|--------|-------|--------------------------------------|
| Type 1 | 16    | Single base Gaussian                 |
| Type 2 | 120   | All two-component pairs C(16,2)      |
| Type 3 | 4     | Four-component disjoint quartets     |

The RKE optimum is the uniform mixture over all 16 bases. Under a four-sparse
constraint the only valid representation uses the four Type-3 mixtures.

## Visualization API

```python
from src.visualize import plot_mixture_selection
# w is a 140-dim simplex vector over all candidate mixtures
plot_mixture_selection(weights=w, manifest=manifest, save_path="out.png")
```

See [DEVELOPER.md](DEVELOPER.md) for full documentation.
