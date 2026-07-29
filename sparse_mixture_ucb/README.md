# Sparse Mixture-UCB Toy 2D Gaussian Experiments

This repository contains a small, reproducible experiment for comparing:

1. Exact Sparse-Mixture-UCB
2. Frank-Wolfe Sparse-Mixture-UCB
3. Fully-Corrective Sparse-Mixture-UCB
4. Non-sparse Mixture-UCB baseline
5. Oracle best standalone generator bandit

The toy problem uses fixed 2D Gaussian generators

```math
P_i = \mathcal N(\mu_i, 0.16 I_2), \quad i=1,\ldots,m.
```

The code supports candidate sets with `m=8`, `m=12`, or `m=16`.  The smaller settings are prefixes of the fixed 16-candidate design, so the experiments are nested and easy to compare.  You can also sweep multiple sparsity levels with `--s-values`.

The two objectives are:

- `rke`: target-free RKE-style diversity objective, `L(alpha)=alpha^T K alpha`.
- `mmd`: squared MMD to a fixed 3-component Gaussian-mixture target, `Q = 0.45 P_3 + 0.35 P_5 + 0.20 P_7`.

All regret curves use the sparse-output convention from the main paper:

```math
\hat\alpha_t = q_\tau, \qquad \tau \sim \mathrm{Unif}\{1,\ldots,t\},
```

so the plotted regret is

```math
R_t = \frac{1}{t}\sum_{r=1}^t L(q_r) - L(\alpha_s^\star).
```

The non-sparse baseline is also compared against the same sparse benchmark `alpha_s_star`, so it may go below zero.

## Installation

Create an environment with Python 3.10+ and install:

```bash
pip install -r requirements.txt
```

The default code only depends on NumPy, pandas, matplotlib, and tqdm. Optional CVXPY solver hooks are included for users who want to compare against external QP/MIQP backends.

## Solver sanity check

The package includes explicit continuous QP and sparse MIQP solver code in `sparse_ucb/solvers.py`. To verify the solvers on a small random convex instance, run:

```bash
python -m sparse_ucb.check_solvers
```

Optional CVXPY hooks are also provided. To use them, install:

```bash
pip install -r requirements-cvxpy.txt
```

The default experiment does **not** require CVXPY or a commercial MIQP solver.

## Quick smoke tests

Original small setting:

```bash
python -m sparse_ucb.run_experiment --objective both --m 8 --s 3 --T 80 --reps 2 --out results_smoke
```

More candidates:

```bash
python -m sparse_ucb.run_experiment --objective both --m 16 --s 3 --T 80 --reps 2 --out results_m16_smoke
```

Sweep sparsity levels:

```bash
python -m sparse_ucb.run_experiment --objective both --m 16 --s-values 1,2,3,4 --T 80 --reps 2 --out results_s_sweep_smoke
```

## Full default experiment

The default design is `T=2000`, `reps=50`, `m=8`, `s=3`, `beta=4`:

```bash
python -m sparse_ucb.run_experiment --objective both --out results
```

## More candidates and multiple sparsity levels

Use 12 or 16 candidates with:

```bash
python -m sparse_ucb.run_experiment --objective both --m 12 --s 3 --out results_m12_s3
python -m sparse_ucb.run_experiment --objective both --m 16 --s 3 --out results_m16_s3
```

Sweep several values of `s`:

```bash
python -m sparse_ucb.run_experiment --objective both --m 16 --s-values 1,2,3,4,5 --out results_m16_sweep
```

When `--s-values` contains more than one value, the code writes one subdirectory per sparsity level, for example `m16_s01`, `m16_s02`, etc., and also writes a combined `sweep_summary.csv` plus final-regret-vs-`s` plots.

Exact Sparse-Mixture-UCB solves the sparse MIQP by exact support enumeration.  The number of supports checked per exact sparse optimization is

```math
\sum_{k=1}^s {m \choose k}.
```

Examples:

| m | s | supports |
|---|---:|---------:|
| 8 | 3 | 92 |
| 12 | 3 | 298 |
| 16 | 3 | 696 |
| 16 | 4 | 2516 |
| 16 | 5 | 6884 |

Large `m`/`s` sweeps can therefore be slow for the exact sparse method.

## Outputs

For each objective and each `(m,s)` setting, the output directory contains:

- `{objective}_regret_curves.csv`: mean and standard-error regret curves.
- `{objective}_final_weights.csv`: final mixture weights averaged over repetitions.
- `{objective}_final_counts.csv`: final arm-pull counts averaged over repetitions. For the standalone bandit, all `T` pulls are on the oracle-best arm.
- `{objective}_oracle.json`: true sparse optimum, full optimum, standalone baseline metadata, and the number of exact sparse supports checked.
- `{objective}_regret.png`: regret curves.
- `{objective}_geometry.png`: generator geometry.
- `{objective}_oracle_sparse.png`: true sparse oracle mixture.
- `{objective}_final_mixtures.png`: average final mixtures for all baselines, including the standalone bandit.

When using `--s-values`, the root output directory also contains:

- `sweep_summary.csv`: final regret at horizon `T` for every objective, method, and `s`.
- `{objective}_s_sweep_final_regret.png`: final regret versus sparsity.

## Fixed means

The maximum 16-candidate generator set is fixed as follows.  Using `m=8` or `m=12` selects the first 8 or 12 rows.

| i | mu_i |
|---|------|
| 1 | (-3.0, -2.0) |
| 2 | (-2.0, 1.5) |
| 3 | (-0.8, -1.2) |
| 4 | (0.0, 2.4) |
| 5 | (1.2, 0.2) |
| 6 | (2.2, -1.8) |
| 7 | (3.0, 1.4) |
| 8 | (0.9, 3.2) |
| 9 | (-3.2, 0.3) |
| 10 | (-1.4, 3.3) |
| 11 | (0.2, -3.0) |
| 12 | (1.8, 2.8) |
| 13 | (3.5, -0.6) |
| 14 | (-3.6, 2.8) |
| 15 | (3.7, 3.1) |
| 16 | (-0.1, 0.4) |

All generators use covariance `0.16 I_2`.

For MMD, the target remains

```math
Q = 0.45 P_3 + 0.35 P_5 + 0.20 P_7.
```

## Best standalone bandit baseline

The best standalone generator is implemented as an actual oracle bandit policy. For each objective, the code first computes

```math
i^\star \in \arg\min_{i \in [m]} L(e_i).
```

Then, during every repetition and every round, the standalone baseline deploys

```math
q_t=e_{i^\star}, \qquad A_t=i^\star, \qquad t=1,\ldots,T,
```

and consumes samples from that generator's sample stream. Its regret curve is horizontal because it always deploys the same distribution, but it is still represented in the code as a bandit with final counts equal to `T` on the best arm and zero on all other arms.

## Notes on optimization

The optimization layer is explicit and lives in `sparse_ucb/solvers.py`. It contains:

- `solve_simplex_qp(...)`: continuous QP over the full simplex. The `auto` backend uses exact active-set enumeration for `m<=12` and projected gradient for larger full-simplex QPs.
- `solve_qp_on_fixed_support_kkt(...)`: KKT solver for a fixed support/simplex face.
- `solve_sparse_miqp(...)`: sparse cardinality-constrained MIQP solver. The default backend is exact support enumeration.
- Optional CVXPY backends: `solve_qp_cvxpy(...)` and `solve_miqp_sparse_cvxpy(...)`.

The exact sparse method remains a true exact sparse optimizer for the toy setting because it enumerates all supports of size at most `s` and solves the support-restricted convex QP on each one.
