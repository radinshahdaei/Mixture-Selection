# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Verify QP/MIQP solvers on a random instance
python -m sparse_ucb.check_solvers

# Quick smoke test (m=8, s=3, 80 rounds, 2 reps)
bash scripts/run_smoke.sh

# Full experiment (m=8, s=3, 2000 rounds, 50 reps)
bash scripts/run_full.sh

# Custom experiments
python -m sparse_ucb.run_experiment --objective both --m 8 --s 3 --out results
python -m sparse_ucb.run_experiment --objective both --m 16 --s-values 1,2,3,4,5 --out results_sweep
```

Dependencies: `numpy`, `pandas`, `matplotlib`, `tqdm` (`pip install -r requirements.txt`). Optional CVXPY: `pip install -r requirements-cvxpy.txt`.

## Architecture

### Module dependency graph

```
config.py          — fixed generator means, ExperimentConfig dataclass
objectives.py      — ObjectiveSpec (true K, f, oracles), RBF kernel, MMD target
solvers.py         — QP/MIQP solvers (KKT, active-set, projected gradient, CVXPY)
optim.py           — thin backward-compat wrappers around solvers.py
algorithms.py      — EmpiricalState (incremental V-statistic), bandit loop, optimizers
run_experiment.py  — CLI, experiment orchestration, CSV/JSON output
plotting.py        — matplotlib figures (regret, geometry, mixtures, s-sweep)
```

### Data flow during an experiment

1. `config.py`: `ExperimentConfig` holds means, `sigma_g`, kernel bandwidth, horizon, reps. `fixed_means(m)` returns the first `m` rows of a fixed 16-generator 2D design.
2. `objectives.py`: `make_objective(name, config)` precomputes closed-form true kernel matrix `K` (expectation of RBF kernel under Gaussian draws), `f_true` (zero for RKE; `-2 * E_Q[k(x,Y)]` for MMD), and true oracles — it calls `optim.py/solvers.py` to solve the true optimization problems.
3. `run_experiment.py`: generates a sample bank per repetition (shared across all methods), then calls `run_one_algorithm` for each method and `run_standalone_bandit`.
4. `algorithms.py`: The bandit loop initializes by pulling each arm once. At each round `t`, it builds optimistic `Khat` and `fhat` from `EmpiricalState` (incremental V-statistic updates), computes UCB via `epsilon(t)`, then calls the method's optimizer to get `alpha`, samples an arm from `alpha`, and observes a sample.

### Key design decisions

- **Sample bank sharing**: Within each repetition, all methods share the same pre-generated sample bank `(m, T+5, d)`, so differences in regret come from the bandit policy, not sample noise.
- **Exact support enumeration for sparse MIQP**: The default backend enumerates all `sum_{k=1}^s C(m,k)` supports and solves a KKT system on each. This is exact but exponential — for `m=16, s=5`, that's 6884 supports per optimization.
- **Sparse-output convention**: Regret uses `R_t = (1/t)*sum_{r=1}^t L(q_r) - L(alpha_s_star)`, where `q_r` is the deployed distribution at round `r`, and `alpha_s_star` is the true s-sparse optimum. The non-sparse baseline is measured against the same sparse benchmark, so its regret can go negative.
- **Auto full-QP backend**: For `m <= 12`, full-simplex QP uses exact active-set enumeration (all `2^m - 1` supports). For `m = 16`, it switches to deterministic projected gradient to avoid the `2^16` cost.
- **Standalone baseline is a real policy**: It always deploys `e_{i*}` and pulls arm `i*`, consuming actual samples. Its regret curve is horizontal.
- **Frank-Wolfe `gamma_r = 2/(r+2)`**: Starts from the best vertex, runs `s` updates. Since `gamma_0 = 1`, the first update resets to a vertex; after `s` updates, support ≤ `s`.
- **Fully-corrective FW**: At each iteration, adds the vertex with smallest gradient to the active set, then re-solves the QP restricted to that support (via active-set KKT enumeration). After convergence, prunes near-zero entries.
- **All RKE/MMD expectations are closed-form**: No Monte Carlo integration — `gaussian_rbf_expectation` computes exact kernel expectations under isotropic Gaussians, so oracle computations are exact.

### Solver backends

| Backend | When used | Method |
|----------|-----------|--------|
| `kkt-fixed-support` | One fixed support face | Solves `[2Q 1; 1^T 0][x; nu] = [-c; 1]` |
| `active-set-enumeration` | Full QP (m ≤ 12) or sparse MIQP | Enumerates subsets, solves KKT on each |
| `projected-gradient` | Full QP (m > 12) | Deterministic, simplex-projected GD with backtracking |
| `miqp-support-enumeration` | Default sparse MIQP | Enumerates supports of size ≤ s |
| `cvxpy` | Optional, requires CVXPY | Delegates to OSQP/CLARABEL (QP) or GUROBI/MOSEK (MIQP) |

### ObjectiveSpec fields

Each `ObjectiveSpec` (created by `make_objective`) carries the true problem data and oracles:
- `K`: true `m×m` kernel matrix (closed-form expectation, not empirical)
- `f_true`: linear term vector
- `const`: additive constant
- `alpha_sparse_star`, `loss_sparse_star`: true s-sparse optimum (regret benchmark)
- `alpha_full_star`, `loss_full_star`: true full-simplex optimum
- `best_standalone_index`, `best_standalone_loss`: best single-arm policy
- `delta_kappa`, `delta_f`, `delta_L`: concentration bounds for UCB

### `--s-values` sweep behavior

When `--s-values 1,2,3,4,5` is passed, the code creates subdirectories `m16_s01/`, `m16_s02/`, etc., each with its own config, CSVs, and figures. It also writes a combined `sweep_summary.csv` and final-regret-vs-`s` plots at the root output directory.
