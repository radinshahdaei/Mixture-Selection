from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import ExperimentConfig
from .optim import solve_simplex_qp_by_active_sets


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    K: np.ndarray
    f_true: np.ndarray
    const: float
    delta_kappa: float
    delta_f: float
    delta_L: float
    alpha_sparse_star: np.ndarray
    loss_sparse_star: float
    alpha_full_star: np.ndarray
    loss_full_star: float
    best_standalone_index: int
    best_standalone_loss: float

    def true_loss(self, alpha: np.ndarray) -> float:
        alpha = np.asarray(alpha, dtype=float)
        return float(alpha @ self.K @ alpha + self.f_true @ alpha + self.const)

    def true_losses(self, alphas: np.ndarray) -> np.ndarray:
        alphas = np.asarray(alphas, dtype=float)
        return np.einsum("bi,ij,bj->b", alphas, self.K, alphas) + alphas @ self.f_true + self.const

    @property
    def standalone_regret(self) -> float:
        return float(self.best_standalone_loss - self.loss_sparse_star)


# -------------------------------------------------------------------------
# RBF kernel and closed-form Gaussian expectations
# -------------------------------------------------------------------------

def rbf_kernel_matrix(X: np.ndarray, Y: np.ndarray, h: float) -> np.ndarray:
    """Pairwise Gaussian RBF kernel matrix between two point clouds."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    sqdist = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    return np.exp(-sqdist / (2.0 * h * h))


def gaussian_rbf_expectation(mu_a: np.ndarray, mu_b: np.ndarray, sigma_a: float, sigma_b: float, h: float, d: int = 2) -> float:
    """E exp(-||X-Y||^2/(2 h^2)) for isotropic Gaussians in d dimensions."""
    mu_a = np.asarray(mu_a, dtype=float)
    mu_b = np.asarray(mu_b, dtype=float)
    var_sum = sigma_a ** 2 + sigma_b ** 2
    denom = h * h + var_sum
    prefactor = (h * h / denom) ** (d / 2.0)
    sqdist = float(np.sum((mu_a - mu_b) ** 2))
    return float(prefactor * np.exp(-sqdist / (2.0 * denom)))


def build_true_K(config: ExperimentConfig) -> np.ndarray:
    m = config.m
    K = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(m):
            K[i, j] = gaussian_rbf_expectation(
                config.means[i],
                config.means[j],
                config.sigma_g,
                config.sigma_g,
                config.kernel_bandwidth,
                d=config.d,
            )
    return K


def mmd_b_and_const(config: ExperimentConfig) -> tuple[np.ndarray, float]:
    m = config.m
    idx = np.array(config.target_indices, dtype=int)
    w = np.array(config.target_weights, dtype=float)
    h = config.kernel_bandwidth

    b = np.zeros(m, dtype=float)
    for i in range(m):
        val = 0.0
        for weight, j in zip(w, idx):
            val += weight * gaussian_rbf_expectation(
                config.means[i],
                config.means[j],
                config.sigma_g,
                config.sigma_q,
                h,
                d=config.d,
            )
        b[i] = val

    c = 0.0
    for wa, ia in zip(w, idx):
        for wb, ib in zip(w, idx):
            c += wa * wb * gaussian_rbf_expectation(
                config.means[ia],
                config.means[ib],
                config.sigma_q,
                config.sigma_q,
                h,
                d=config.d,
            )
    return b, float(c)


def make_objective(name: str, config: ExperimentConfig) -> ObjectiveSpec:
    name = name.lower()
    K = build_true_K(config)

    if name == "rke":
        f_true = np.zeros(config.m, dtype=float)
        const = 0.0
        delta_kappa = 1.0
        delta_f = 0.0
        delta_L = 2.0
    elif name == "mmd":
        b, const = mmd_b_and_const(config)
        f_true = -2.0 * b
        delta_kappa = 1.0
        # Safe range for f(x)=-2 E_Q[k(x,Y)] is [-2, 0].
        delta_f = 2.0
        delta_L = 4.0
    else:
        raise ValueError(f"Unknown objective '{name}'. Use 'rke' or 'mmd'.")

    alpha_sparse, _ = solve_simplex_qp_by_active_sets(K, f_true, max_support=config.s)
    loss_sparse = float(alpha_sparse @ K @ alpha_sparse + f_true @ alpha_sparse + const)

    alpha_full, _ = solve_simplex_qp_by_active_sets(K, f_true, max_support=None)
    loss_full = float(alpha_full @ K @ alpha_full + f_true @ alpha_full + const)

    eye = np.eye(config.m)
    standalone_losses = np.array([float(e @ K @ e + f_true @ e + const) for e in eye])
    best_idx = int(np.argmin(standalone_losses))

    return ObjectiveSpec(
        name=name,
        K=K,
        f_true=f_true,
        const=const,
        delta_kappa=delta_kappa,
        delta_f=delta_f,
        delta_L=delta_L,
        alpha_sparse_star=alpha_sparse,
        loss_sparse_star=loss_sparse,
        alpha_full_star=alpha_full,
        loss_full_star=loss_full,
        best_standalone_index=best_idx,
        best_standalone_loss=float(standalone_losses[best_idx]),
    )


def f_sample_values(X: np.ndarray, objective: ObjectiveSpec, config: ExperimentConfig) -> np.ndarray:
    """Evaluate f(x) for sample points X under the chosen objective."""
    X = np.asarray(X, dtype=float)
    if objective.name == "rke":
        return np.zeros(X.shape[0], dtype=float)

    if objective.name != "mmd":
        raise ValueError(f"Unsupported objective {objective.name}")

    # f(x) = -2 E_{Y~Q} k(x,Y), where Q is a Gaussian mixture.
    h = config.kernel_bandwidth
    idx = np.array(config.target_indices, dtype=int)
    weights = np.array(config.target_weights, dtype=float)
    means_q = config.means[idx]
    denom = h * h + config.sigma_q ** 2
    prefactor = (h * h / denom) ** (config.d / 2.0)

    sqdist = np.sum((X[:, None, :] - means_q[None, :, :]) ** 2, axis=2)
    vals = prefactor * np.exp(-sqdist / (2.0 * denom))
    return -2.0 * (vals @ weights)
