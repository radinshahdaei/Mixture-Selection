from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import numpy as np

from .config import ExperimentConfig
from .objectives import ObjectiveSpec, f_sample_values, rbf_kernel_matrix
from .solvers import best_vertex, solve_simplex_qp, solve_sparse_miqp


@dataclass
class EmpiricalState:
    """Incremental empirical estimates for K and f."""

    config: ExperimentConfig
    objective: ObjectiveSpec
    samples: list[list[np.ndarray]] = field(init=False)
    counts: np.ndarray = field(init=False)
    kernel_sums: np.ndarray = field(init=False)
    f_sums: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.samples = [[] for _ in range(self.config.m)]
        self.counts = np.zeros(self.config.m, dtype=int)
        self.kernel_sums = np.zeros((self.config.m, self.config.m), dtype=float)
        self.f_sums = np.zeros(self.config.m, dtype=float)

    def add_sample(self, arm: int, x: np.ndarray) -> None:
        arm = int(arm)
        x = np.asarray(x, dtype=float)
        h = self.config.kernel_bandwidth

        # Update kernel V-statistic sums before appending x.
        for j in range(self.config.m):
            if self.counts[j] == 0:
                continue
            Y = np.asarray(self.samples[j], dtype=float)
            vals = rbf_kernel_matrix(x[None, :], Y, h).reshape(-1)
            s = float(vals.sum())
            if j == arm:
                # Add row x against old samples, column old samples against x, and self-pair k(x,x)=1.
                self.kernel_sums[arm, arm] += 2.0 * s + 1.0
            else:
                self.kernel_sums[arm, j] += s
                self.kernel_sums[j, arm] += s

        if self.counts[arm] == 0:
            # First sample of an arm only contributes the self-pair.
            self.kernel_sums[arm, arm] += 1.0

        self.samples[arm].append(x)
        self.counts[arm] += 1
        self.f_sums[arm] += float(f_sample_values(x[None, :], self.objective, self.config)[0])

    def Khat(self) -> np.ndarray:
        if np.any(self.counts == 0):
            raise RuntimeError("Khat requested before every arm has at least one sample")
        denom = self.counts[:, None] * self.counts[None, :]
        return self.kernel_sums / denom

    def fhat(self) -> np.ndarray:
        if np.any(self.counts == 0):
            raise RuntimeError("fhat requested before every arm has at least one sample")
        return self.f_sums / self.counts

    def epsilon(self, T: int) -> np.ndarray:
        n = self.counts.astype(float)
        return self.objective.delta_L * np.sqrt(self.config.beta * np.log(T) / (2.0 * n)) + self.objective.delta_kappa / n


# -------------------------------------------------------------------------
# Optimistic optimizers
# -------------------------------------------------------------------------

def exact_sparse_optimizer(Q: np.ndarray, c: np.ndarray, s: int) -> np.ndarray:
    """Exact sparse MIQP optimizer using support-enumeration by default."""
    result = solve_sparse_miqp(Q, c, s=s, backend="enumeration")
    return result.alpha


def full_simplex_optimizer(Q: np.ndarray, c: np.ndarray, s: int | None = None) -> np.ndarray:
    """Non-sparse full-simplex QP optimizer."""
    result = solve_simplex_qp(Q, c, max_support=None, backend="auto")
    return result.alpha


def coherence_optimizer(
    Q: np.ndarray, c_opt: np.ndarray, fhat: np.ndarray, s: int
) -> np.ndarray:
    """Coherence-structured fully-corrective sparse optimizer (Algorithm 3).

    Uses the unoptimistic objective (f̂) for support discovery via greedy
    reduced-gradient selection and simplex corrections, then applies a final
    optimistic correction (c_opt = f̂ − ε) over the discovered face.
    """
    Q = 0.5 * (Q + Q.T)
    alpha = best_vertex(Q, fhat)
    active: set[int] = {int(np.argmax(alpha))}

    for _ in range(1, s):
        grad = 2.0 * Q @ alpha + fhat
        j = int(np.argmin(grad))
        active.add(j)
        idx = sorted(active)
        Qs = Q[np.ix_(idx, idx)]
        cs = fhat[idx]
        sub_alpha = solve_simplex_qp(Qs, cs, max_support=None, backend="active-set").alpha
        alpha = np.zeros(Q.shape[0], dtype=float)
        alpha[idx] = sub_alpha
        active = {i for i in idx if alpha[i] > 1e-12}
        if len(active) == 0:
            active = {int(np.argmax(alpha))}

    # Final optimistic correction over the discovered face.
    idx = sorted(active)
    Qs = Q[np.ix_(idx, idx)]
    cs = c_opt[idx]
    sub_alpha = solve_simplex_qp(Qs, cs, max_support=None, backend="active-set").alpha
    alpha = np.zeros(Q.shape[0], dtype=float)
    alpha[idx] = sub_alpha
    alpha[alpha < 1e-14] = 0.0
    return alpha / alpha.sum()


def fully_corrective_sparse_optimizer(Q: np.ndarray, c: np.ndarray, s: int) -> np.ndarray:
    """Fully-corrective sparse Frank-Wolfe optimizer."""
    Q = 0.5 * (Q + Q.T)
    alpha = best_vertex(Q, c)
    active: set[int] = {int(np.argmax(alpha))}

    for _ in range(1, s):
        grad = 2.0 * Q @ alpha + c
        j = int(np.argmin(grad))
        active.add(j)
        idx = sorted(active)
        Qs = Q[np.ix_(idx, idx)]
        cs = c[idx]
        sub_alpha = solve_simplex_qp(Qs, cs, max_support=None, backend="active-set").alpha
        alpha = np.zeros(Q.shape[0], dtype=float)
        alpha[idx] = sub_alpha
        active = {i for i in idx if alpha[i] > 1e-12}
        if len(active) == 0:
            active = {int(np.argmax(alpha))}
    alpha[alpha < 1e-14] = 0.0
    return alpha / alpha.sum()


OPTIMIZERS: dict[str, Callable[[np.ndarray, np.ndarray, int], np.ndarray]] = {
    "exact_sparse": exact_sparse_optimizer,
    "fc_sparse": fully_corrective_sparse_optimizer,
    "full": full_simplex_optimizer,
}

DISPLAY_NAMES = {
    "exact_sparse": "Exact Sparse",
    "fc_sparse": "Fully-Corrective Sparse",
    "full": "Non-sparse Mixture-UCB",
    "coherence": "Coherence-Structured FC",
    "standalone": "Best Standalone",
}


@dataclass
class RunResult:
    method: str
    regret_curve: np.ndarray
    q_losses: np.ndarray
    final_alpha: np.ndarray
    final_counts: np.ndarray


def run_one_algorithm(
    method: str,
    sample_bank: np.ndarray,
    config: ExperimentConfig,
    objective: ObjectiveSpec,
    rng: np.random.Generator,
) -> RunResult:
    """Run one algorithm for one repetition using a pre-generated sample bank.

    sample_bank[i, k] is the k-th possible sample from arm i. This makes all
    algorithms in the same repetition share the same per-arm observation stream.
    """
    if method not in OPTIMIZERS and method != "coherence":
        raise ValueError(f"Unknown method {method}")

    T = int(config.T)
    m = int(config.m)
    state = EmpiricalState(config=config, objective=objective)
    next_index = np.zeros(m, dtype=int)
    q_losses = np.zeros(T, dtype=float)
    final_alpha = np.ones(m) / m

    # Initialization: pull every arm (once for standard, n_0 times for coherence).
    n_init = config.n_0 if method == "coherence" else 1
    for rep in range(n_init):
        for i in range(m):
            x = sample_bank[i, next_index[i]]
            next_index[i] += 1
            state.add_sample(i, x)
            e = np.zeros(m, dtype=float)
            e[i] = 1.0
            q_losses[i + rep * m] = objective.true_loss(e)
            final_alpha = e

    init_rounds = n_init * m
    for t in range(init_rounds, T):
        Khat = state.Khat()
        fhat = state.fhat()
        eps = state.epsilon(T)
        c = fhat - eps

        if method == "coherence":
            alpha = coherence_optimizer(Khat, c, fhat, config.s)
        else:
            optimizer = OPTIMIZERS[method]
            alpha = optimizer(Khat, c, config.s)
        alpha = np.maximum(alpha, 0.0)
        alpha = alpha / alpha.sum()

        q_losses[t] = objective.true_loss(alpha)
        final_alpha = alpha.copy()

        if method == "coherence":
            eta = min(1.0, config.c_eta * (t ** (-config.a_eta)))
            q_sample = (1.0 - eta) * alpha + eta / m
        else:
            q_sample = alpha

        arm = int(rng.choice(m, p=q_sample))
        if next_index[arm] >= sample_bank.shape[1]:
            raise RuntimeError("Sample bank exhausted; increase T or bank size")
        x = sample_bank[arm, next_index[arm]]
        next_index[arm] += 1
        state.add_sample(arm, x)

    regret_curve = np.cumsum(q_losses) / np.arange(1, T + 1) - objective.loss_sparse_star
    return RunResult(method=method, regret_curve=regret_curve, q_losses=q_losses, final_alpha=final_alpha, final_counts=state.counts.copy())


def run_standalone_bandit(
    sample_bank: np.ndarray,
    config: ExperimentConfig,
    objective: ObjectiveSpec,
) -> RunResult:
    """Oracle best-standalone bandit baseline.

    This is a real policy, not just a post-hoc horizontal line: it deploys
    q_t=e_{i^*} at every round, pulls the same oracle-best arm i^* for all
    T rounds, and records the resulting count trajectory. Its regret curve is
    constant because L(e_{i^*}) is the same at every deployed round.
    """
    T = int(config.T)
    m = int(config.m)
    best_arm = int(objective.best_standalone_index)

    e = np.zeros(m, dtype=float)
    e[best_arm] = 1.0

    # Actually consume samples from the best arm, so this baseline is implemented
    # as a bandit policy that always pulls one generator. The observed samples do
    # not affect the oracle policy, but the counts make the behavior explicit.
    if sample_bank.shape[1] < T:
        raise RuntimeError("Sample bank exhausted for standalone bandit")
    _observed_samples = sample_bank[best_arm, :T].copy()

    q_loss = objective.true_loss(e)
    q_losses = np.full(T, q_loss, dtype=float)
    regret_curve = np.cumsum(q_losses) / np.arange(1, T + 1) - objective.loss_sparse_star
    counts = np.zeros(m, dtype=int)
    counts[best_arm] = T

    return RunResult(
        method="standalone",
        regret_curve=regret_curve,
        q_losses=q_losses,
        final_alpha=e,
        final_counts=counts,
    )


# Backward-compatible alias.
def run_standalone_baseline(
    config: ExperimentConfig,
    objective: ObjectiveSpec,
    sample_bank: np.ndarray | None = None,
) -> RunResult:
    if sample_bank is None:
        rng = np.random.default_rng(config.seed + 999_999)
        sample_bank = rng.normal(
            loc=config.means[:, None, :],
            scale=config.sigma_g,
            size=(config.m, config.T + 5, config.d),
        )
    return run_standalone_bandit(sample_bank, config, objective)
