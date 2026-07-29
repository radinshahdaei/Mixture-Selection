from __future__ import annotations

"""Small QP/MIQP solvers for the toy Sparse Mixture-UCB experiment.

The sparse MIQP backend is exact support enumeration: enumerate all supports of
size at most s and solve the convex QP on each support.  This remains practical
for the intended toy sizes, e.g. m=8,12,16 and modest s.

For the non-sparse full-simplex QP, the solver supports exact active-set
enumeration for small m and a lightweight projected-gradient backend for larger
candidate sets.  Optional CVXPY hooks are included for external QP/MIQP solvers.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import comb
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class QPResult:
    """Result returned by the QP and MIQP helper solvers."""

    alpha: np.ndarray
    value: float
    status: str
    backend: str
    active_set: tuple[int, ...]
    num_supports_checked: int = 0
    message: str = ""


# -------------------------------------------------------------------------
# Basic utilities
# -------------------------------------------------------------------------


def symmetrize(Q: np.ndarray) -> np.ndarray:
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix")
    return 0.5 * (Q + Q.T)


def check_qp_inputs(Q: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Qs = symmetrize(Q)
    c = np.asarray(c, dtype=float)
    if c.shape != (Qs.shape[0],):
        raise ValueError(f"c must have shape ({Qs.shape[0]},), got {c.shape}")
    return Qs, c


def simplex_project(y: np.ndarray) -> np.ndarray:
    """Euclidean projection of ``y`` onto the probability simplex."""

    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("simplex_project expects a 1D array")
    n = y.size
    u = np.sort(y)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = u * np.arange(1, n + 1) > (cssv - 1.0)
    if not np.any(rho_candidates):
        return np.ones(n) / n
    rho = np.nonzero(rho_candidates)[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1)
    return np.maximum(y - theta, 0.0)


@lru_cache(maxsize=None)
def active_subsets(m: int, max_size: int | None = None) -> tuple[tuple[int, ...], ...]:
    """All nonempty subsets of ``{0,...,m-1}`` up to ``max_size``."""

    if max_size is None:
        max_size = m
    max_size = min(int(max_size), int(m))
    out: list[tuple[int, ...]] = []
    for k in range(1, max_size + 1):
        out.extend(combinations(range(m), k))
    return tuple(out)


def num_active_subsets(m: int, max_size: int | None = None) -> int:
    if max_size is None:
        max_size = m
    max_size = min(int(max_size), int(m))
    return int(sum(comb(m, k) for k in range(1, max_size + 1)))


def objective_value(Q: np.ndarray, c: np.ndarray, alpha: np.ndarray) -> float:
    Qs, c = check_qp_inputs(Q, c)
    alpha = np.asarray(alpha, dtype=float)
    return float(alpha @ Qs @ alpha + c @ alpha)


def best_vertex(Q: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Return the simplex vertex minimizing ``alpha^T Q alpha + c^T alpha``."""

    Qs, c = check_qp_inputs(Q, c)
    scores = np.diag(Qs) + c
    j = int(np.argmin(scores))
    x = np.zeros(Qs.shape[0], dtype=float)
    x[j] = 1.0
    return x


# -------------------------------------------------------------------------
# Continuous QP over a simplex face
# -------------------------------------------------------------------------


def solve_qp_on_fixed_support_kkt(
    Q: np.ndarray,
    c: np.ndarray,
    support: Iterable[int],
    tol: float = 1e-10,
) -> QPResult:
    """Solve the equality-constrained QP on one fixed support.

    This solves ``min x^T Q_SS x + c_S^T x`` subject to ``1^T x = 1`` on the
    specified support. If the solution has negative coordinates, the support is
    not an interior optimum of that face; the boundary is handled by smaller
    supports in the active-set enumeration solver.
    """

    Qs, c = check_qp_inputs(Q, c)
    idx = tuple(int(i) for i in support)
    if len(idx) == 0:
        raise ValueError("support must be nonempty")
    if min(idx) < 0 or max(idx) >= Qs.shape[0]:
        raise ValueError("support contains an invalid index")

    id_arr = np.array(idx, dtype=int)
    k = id_arr.size
    Qsub = Qs[np.ix_(id_arr, id_arr)]
    csub = c[id_arr]

    # KKT system for 2 Qsub x + csub + nu 1 = 0, 1^T x = 1.
    A = np.block(
        [
            [2.0 * Qsub, np.ones((k, 1))],
            [np.ones((1, k)), np.zeros((1, 1))],
        ]
    )
    b = np.concatenate([-csub, np.array([1.0])])

    try:
        sol = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(A, b, rcond=None)[0]

    xsub = sol[:k]
    if np.any(xsub < -tol):
        alpha = np.zeros(Qs.shape[0], dtype=float)
        return QPResult(
            alpha=alpha,
            value=np.inf,
            status="infeasible_negative_coordinate",
            backend="kkt-fixed-support",
            active_set=idx,
            message="KKT solution on this support has a negative coordinate.",
        )

    xsub = np.maximum(xsub, 0.0)
    if xsub.sum() <= tol:
        alpha = np.zeros(Qs.shape[0], dtype=float)
        return QPResult(
            alpha=alpha,
            value=np.inf,
            status="infeasible_zero_sum",
            backend="kkt-fixed-support",
            active_set=idx,
        )

    xsub = xsub / xsub.sum()
    alpha = np.zeros(Qs.shape[0], dtype=float)
    alpha[id_arr] = xsub
    val = objective_value(Qs, c, alpha)
    active = tuple(int(i) for i in np.flatnonzero(alpha > tol))
    return QPResult(alpha=alpha, value=val, status="optimal", backend="kkt-fixed-support", active_set=active)


def solve_qp_simplex_active_set(
    Q: np.ndarray,
    c: np.ndarray,
    max_support: int | None = None,
    tol: float = 1e-9,
) -> QPResult:
    """Solve a small convex QP over the simplex by active-set enumeration.

    If ``max_support=s``, this is also an exact sparse MIQP solver for the toy
    setting because it enumerates all binary support choices.
    """

    Qs, c = check_qp_inputs(Q, c)
    m = Qs.shape[0]
    if max_support is None:
        max_support = m
    max_support = min(int(max_support), m)

    best: QPResult | None = None
    subsets = active_subsets(m, max_support)
    for support in subsets:
        res = solve_qp_on_fixed_support_kkt(Qs, c, support, tol=tol)
        if res.status == "optimal" and (best is None or res.value < best.value):
            best = res

    if best is None:
        alpha = best_vertex(Qs, c)
        best = QPResult(
            alpha=alpha,
            value=objective_value(Qs, c, alpha),
            status="fallback_vertex",
            backend="active-set-enumeration",
            active_set=tuple(np.flatnonzero(alpha > tol).tolist()),
            num_supports_checked=len(subsets),
            message="No interior active-set candidate found; returned best vertex.",
        )
    else:
        alpha = best.alpha.copy()
        alpha[alpha < tol] = 0.0
        alpha = alpha / alpha.sum()
        best = QPResult(
            alpha=alpha,
            value=objective_value(Qs, c, alpha),
            status="optimal",
            backend="active-set-enumeration",
            active_set=tuple(np.flatnonzero(alpha > tol).tolist()),
            num_supports_checked=len(subsets),
        )

    return best


def solve_qp_simplex_projected_gradient(
    Q: np.ndarray,
    c: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> QPResult:
    """Projected-gradient solver for the full-simplex convex QP.

    This backend is useful when the candidate set is larger and exact full
    active-set enumeration would require checking all 2^m-1 supports.  It is
    deterministic and self-contained; CVXPY is not required.
    """

    Qs, c = check_qp_inputs(Q, c)
    m = Qs.shape[0]
    alpha = simplex_project(best_vertex(Qs, c) + 1e-12)

    try:
        lam_max = float(np.linalg.eigvalsh(Qs).max())
    except np.linalg.LinAlgError:
        lam_max = float(np.linalg.norm(Qs, ord=2))
    L = max(2.0 * lam_max, 1e-8)
    step = 1.0 / L

    prev_val = objective_value(Qs, c, alpha)
    status = "max_iter"
    for it in range(1, max_iter + 1):
        grad = 2.0 * Qs @ alpha + c
        new_alpha = simplex_project(alpha - step * grad)
        new_val = objective_value(Qs, c, new_alpha)

        # Monotone backtracking if numerical PSD/noise makes a step too large.
        local_step = step
        while new_val > prev_val + 1e-12 and local_step > 1e-12:
            local_step *= 0.5
            new_alpha = simplex_project(alpha - local_step * grad)
            new_val = objective_value(Qs, c, new_alpha)

        if np.linalg.norm(new_alpha - alpha, ord=1) < tol:
            alpha = new_alpha
            prev_val = new_val
            status = "optimal_tol"
            break
        alpha = new_alpha
        prev_val = new_val
    else:
        it = max_iter

    alpha[alpha < 1e-9] = 0.0
    alpha = alpha / alpha.sum()
    return QPResult(
        alpha=alpha,
        value=objective_value(Qs, c, alpha),
        status=status,
        backend="projected-gradient",
        active_set=tuple(np.flatnonzero(alpha > 1e-9).tolist()),
        message=f"Projected-gradient full-simplex QP, iterations={it}.",
    )


# -------------------------------------------------------------------------
# Sparse MIQP wrappers
# -------------------------------------------------------------------------


def solve_miqp_sparse_enumeration(Q: np.ndarray, c: np.ndarray, s: int, tol: float = 1e-9) -> QPResult:
    """Exact sparse MIQP solver by support enumeration."""

    res = solve_qp_simplex_active_set(Q, c, max_support=s, tol=tol)
    return QPResult(
        alpha=res.alpha,
        value=res.value,
        status=res.status,
        backend="miqp-support-enumeration",
        active_set=res.active_set,
        num_supports_checked=res.num_supports_checked,
        message=(
            f"Solved sparse MIQP exactly by enumerating {res.num_supports_checked} "
            f"supports of size <= {s}."
        ),
    )


# -------------------------------------------------------------------------
# Optional CVXPY solvers
# -------------------------------------------------------------------------


def solve_qp_cvxpy(
    Q: np.ndarray,
    c: np.ndarray,
    support: Iterable[int] | None = None,
    solver: str | None = None,
    verbose: bool = False,
) -> QPResult:
    """Optional CVXPY continuous QP solver."""

    try:
        import cvxpy as cp  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "CVXPY is not installed. Install optional dependencies with "
            "`pip install -r requirements-cvxpy.txt`."
        ) from exc

    Qs, c = check_qp_inputs(Q, c)
    m = Qs.shape[0]
    if support is None:
        idx = np.arange(m, dtype=int)
    else:
        idx = np.array(tuple(int(i) for i in support), dtype=int)
    Qsub = Qs[np.ix_(idx, idx)]
    csub = c[idx]

    x = cp.Variable(idx.size)
    objective = cp.Minimize(cp.quad_form(x, cp.psd_wrap(Qsub)) + csub @ x)
    constraints = [x >= 0, cp.sum(x) == 1]
    prob = cp.Problem(objective, constraints)

    if solver is None:
        installed = set(cp.installed_solvers())
        for candidate in ["OSQP", "CLARABEL", "ECOS", "SCS"]:
            if candidate in installed:
                solver = candidate
                break

    prob.solve(solver=solver, verbose=verbose) if solver else prob.solve(verbose=verbose)
    if x.value is None:
        raise RuntimeError(f"CVXPY QP failed with status {prob.status}")

    alpha = np.zeros(m, dtype=float)
    alpha[idx] = np.maximum(np.asarray(x.value, dtype=float), 0.0)
    alpha = alpha / alpha.sum()
    return QPResult(
        alpha=alpha,
        value=objective_value(Qs, c, alpha),
        status=str(prob.status),
        backend=f"cvxpy-{solver or 'default'}",
        active_set=tuple(np.flatnonzero(alpha > 1e-9).tolist()),
        message="Solved continuous simplex QP with CVXPY.",
    )


def solve_miqp_sparse_cvxpy(
    Q: np.ndarray,
    c: np.ndarray,
    s: int,
    solver: str | None = None,
    verbose: bool = False,
) -> QPResult:
    """Optional CVXPY mixed-integer QP solver for the sparse simplex problem."""

    try:
        import cvxpy as cp  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "CVXPY is not installed. Install optional dependencies with "
            "`pip install -r requirements-cvxpy.txt`."
        ) from exc

    Qs, c = check_qp_inputs(Q, c)
    m = Qs.shape[0]
    alpha = cp.Variable(m)
    z = cp.Variable(m, boolean=True)
    objective = cp.Minimize(cp.quad_form(alpha, cp.psd_wrap(Qs)) + c @ alpha)
    constraints = [alpha >= 0, cp.sum(alpha) == 1, alpha <= z, cp.sum(z) <= int(s)]
    prob = cp.Problem(objective, constraints)

    if solver is None:
        installed = set(cp.installed_solvers())
        for candidate in ["GUROBI", "MOSEK", "CPLEX", "XPRESS", "SCIP", "ECOS_BB"]:
            if candidate in installed:
                solver = candidate
                break
        if solver is None:
            raise RuntimeError(
                "No CVXPY mixed-integer QP solver found. Install GUROBI, MOSEK, "
                "CPLEX, XPRESS, SCIP, or ECOS_BB, or use enumeration."
            )

    prob.solve(solver=solver, verbose=verbose)
    if alpha.value is None:
        raise RuntimeError(f"CVXPY MIQP failed with status {prob.status}")

    out = np.maximum(np.asarray(alpha.value, dtype=float), 0.0)
    out[out < 1e-9] = 0.0
    out = out / out.sum()
    return QPResult(
        alpha=out,
        value=objective_value(Qs, c, out),
        status=str(prob.status),
        backend=f"cvxpy-miqp-{solver}",
        active_set=tuple(np.flatnonzero(out > 1e-9).tolist()),
        message="Solved sparse MIQP with CVXPY mixed-integer backend.",
    )


# -------------------------------------------------------------------------
# User-facing dispatchers
# -------------------------------------------------------------------------


def solve_simplex_qp(
    Q: np.ndarray,
    c: np.ndarray,
    max_support: int | None = None,
    backend: str = "auto",
    **kwargs,
) -> QPResult:
    """Solve a simplex QP with a selected backend.

    Backends:
    - ``auto``: active-set for small exact full QPs; projected-gradient for
      larger full QPs; exact enumeration for sparse support-capped QPs.
    - ``active-set``: exact active-set enumeration.
    - ``projected-gradient``: deterministic full-simplex projected-gradient QP.
    - ``cvxpy``: optional continuous QP solver using CVXPY.
    """

    Qs, c = check_qp_inputs(Q, c)
    m = Qs.shape[0]

    if backend == "auto":
        if max_support is not None:
            return solve_qp_simplex_active_set(Qs, c, max_support=max_support, **kwargs)
        # Full active-set enumeration checks 2^m-1 supports.  This is fine for
        # m <= 12 in this toy package, but for m=16 we switch to projected GD.
        if m <= 12:
            return solve_qp_simplex_active_set(Qs, c, max_support=None, **kwargs)
        return solve_qp_simplex_projected_gradient(Qs, c, **kwargs)

    if backend == "active-set":
        return solve_qp_simplex_active_set(Qs, c, max_support=max_support, **kwargs)
    if backend == "projected-gradient":
        if max_support is not None:
            raise ValueError("projected-gradient backend only solves the full-simplex QP; use sparse MIQP enumeration for support caps.")
        return solve_qp_simplex_projected_gradient(Qs, c, **kwargs)
    if backend == "cvxpy":
        if max_support is not None:
            return solve_miqp_sparse_cvxpy(Qs, c, int(max_support), **kwargs)
        return solve_qp_cvxpy(Qs, c, **kwargs)
    raise ValueError(f"Unknown QP backend '{backend}'")


def solve_sparse_miqp(
    Q: np.ndarray,
    c: np.ndarray,
    s: int,
    backend: str = "enumeration",
    **kwargs,
) -> QPResult:
    """Solve the sparse simplex MIQP with a selected backend.

    Backends:
    - ``enumeration``: exact support enumeration; default.
    - ``cvxpy``: optional CVXPY mixed-integer solver, if installed.
    """

    if backend == "enumeration":
        return solve_miqp_sparse_enumeration(Q, c, s=s, **kwargs)
    if backend == "cvxpy":
        return solve_miqp_sparse_cvxpy(Q, c, s=s, **kwargs)
    raise ValueError(f"Unknown MIQP backend '{backend}'")
