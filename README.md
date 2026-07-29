# Mixture Selection

Two related projects on sparse mixture selection:

| Project | Directory | Description |
|---------|-----------|-------------|
| Mixture Selection (GMM) | `mixtures/` | Gaussian mixture model generation, sampling, and visualization |
| Sparse Mixture-UCB | `sparse_mixture_ucb/` | Bandit algorithms for online sparse mixture optimization |

---

## Mixture Selection (GMM)

Sparse mixture selection on Gaussian mixture models. Three layouts are supported:
a **ring** of 16 Gaussians, a 6×6 **grid** of 36 Gaussians, and a **random 3D**
layout with full PSD covariances.

### Quick start

```bash
cd mixtures
conda create -n mixture-selection python=3.11 -y && conda activate mixture-selection
pip install -r requirements.txt

# Generate samples (use --config to pick layout)
python scripts/generate_samples.py --config config_ring.yaml       # 140 candidates
python scripts/generate_samples.py --config config_grid.yaml       # 265 candidates
python scripts/generate_samples.py --config config_random_3d.yaml  # 20 candidates (3D singles)

# Visualize (ring and grid only; random_3d has no viz)
python scripts/visualize.py --config config_ring.yaml --demo --reference
python scripts/visualize.py --config config_grid.yaml --demo --reference

# Validate random_3d data
python scripts/validate_random_3d.py
```

### Layouts

| Layout     | Bases | Type 1 | Type 2 | Type 3 | Total | Dim |
|------------|-------|--------|--------|--------|-------|-----|
| Ring       |    16 |     16 |    120 |      4 | **140** | 2D isotropic |
| Grid       |    36 |     36 |    225 |      4 | **265** | 2D isotropic |
| Random 3D  |    20 |     20 |      0 |      0 |  **20** | 3D full cov |

See `mixtures/DEVELOPER.md` for the full developer guide.

---

## Sparse Mixture-UCB

Bandit algorithms for online sparse mixture selection with quadratic objectives (RKE, MMD):

- Fully-Corrective Sparse Mixture-UCB
- Coherence-Structured FC with Forced Exploration
- Non-sparse Mixture-UCB and standalone baselines

```bash
cd sparse_mixture_ucb
pip install -r requirements.txt
python -m sparse_ucb.check_solvers
python -m sparse_ucb.run_experiment --objective both --m 8 --s 3 --T 2000 --reps 10 --out results
```

Full experiment report in `sparse_mixture_ucb/experiment_report.tex`.
