from __future__ import annotations

"""Small sanity check for the QP and MIQP solvers.

Run from the repository root with:

    python -m sparse_ucb.check_solvers
"""

import numpy as np

from .solvers import solve_simplex_qp, solve_sparse_miqp


def main() -> None:
    rng = np.random.default_rng(20260522)
    m = 6
    s = 3
    A = rng.normal(size=(m, m))
    Q = A.T @ A / m
    c = rng.normal(size=m)

    qp = solve_simplex_qp(Q, c, backend="active-set")
    miqp = solve_sparse_miqp(Q, c, s=s, backend="enumeration")

    print("Full-simplex QP")
    print(f"  status: {qp.status}")
    print(f"  backend: {qp.backend}")
    print(f"  value: {qp.value:.8f}")
    print(f"  alpha: {np.round(qp.alpha, 5)}")
    print(f"  active set: {[i + 1 for i in qp.active_set]}")
    print()

    print("Sparse MIQP")
    print(f"  status: {miqp.status}")
    print(f"  backend: {miqp.backend}")
    print(f"  supports checked: {miqp.num_supports_checked}")
    print(f"  value: {miqp.value:.8f}")
    print(f"  alpha: {np.round(miqp.alpha, 5)}")
    print(f"  active set: {[i + 1 for i in miqp.active_set]}")

    assert np.all(qp.alpha >= -1e-9) and abs(qp.alpha.sum() - 1.0) < 1e-8
    assert np.all(miqp.alpha >= -1e-9) and abs(miqp.alpha.sum() - 1.0) < 1e-8
    assert np.count_nonzero(miqp.alpha > 1e-9) <= s
    print("\nSolver sanity checks passed.")


if __name__ == "__main__":
    main()
