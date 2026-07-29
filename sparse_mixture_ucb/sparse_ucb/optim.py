from __future__ import annotations

"""Backward-compatible optimization helpers.

The explicit QP and MIQP implementations live in :mod:`sparse_ucb.solvers`.
This module keeps the shorter helper names used by the experiment code.
"""

import numpy as np

from .solvers import (
    active_subsets,
    best_vertex,
    num_active_subsets,
    simplex_project,
    solve_qp_on_fixed_support_kkt,
    solve_qp_simplex_active_set,
    solve_simplex_qp,
    solve_sparse_miqp,
)


def qp_on_active_set(
    Q: np.ndarray,
    c: np.ndarray,
    active: tuple[int, ...],
    tol: float = 1e-10,
) -> tuple[np.ndarray | None, float]:
    """Compatibility wrapper around the fixed-support KKT QP solver."""

    res = solve_qp_on_fixed_support_kkt(Q, c, active, tol=tol)
    if res.status != "optimal":
        return None, np.inf
    return res.alpha, res.value


def solve_simplex_qp_by_active_sets(
    Q: np.ndarray,
    c: np.ndarray,
    max_support: int | None = None,
    tol: float = 1e-9,
) -> tuple[np.ndarray, float]:
    """Compatibility wrapper for the simplex QP solver.

    With ``max_support`` this is exact sparse support enumeration.  Without a
    support cap it uses the auto full-QP backend, which switches to projected
    gradient for larger candidate sets.
    """

    backend = "active-set" if max_support is not None else "auto"
    res = solve_simplex_qp(Q, c, max_support=max_support, backend=backend, tol=tol)
    return res.alpha, res.value
