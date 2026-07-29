from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# A fixed master list of candidate generator means.  The experiment can use
# the first 8, 12, or 16 entries.  The original m=8 design is the prefix.
MASTER_MEANS_16 = np.array(
    [
        [-3.0, -2.0],
        [-2.0, 1.5],
        [-0.8, -1.2],
        [0.0, 2.4],
        [1.2, 0.2],
        [2.2, -1.8],
        [3.0, 1.4],
        [0.9, 3.2],
        [-3.2, 0.3],
        [-1.4, 3.3],
        [0.2, -3.0],
        [1.8, 2.8],
        [3.5, -0.6],
        [-3.6, 2.8],
        [3.7, 3.1],
        [-0.1, 0.4],
    ],
    dtype=float,
)


VALID_NUM_CANDIDATES = (8, 12, 16)


def fixed_means(num_candidates: int = 8) -> np.ndarray:
    """Return a fixed set of 2D Gaussian means.

    Supported candidate-set sizes are 8, 12, and 16.  The smaller settings are
    prefixes of the 16-candidate design, so runs are nested and easy to compare.
    """

    num_candidates = int(num_candidates)
    if num_candidates not in VALID_NUM_CANDIDATES:
        raise ValueError(
            f"num_candidates must be one of {VALID_NUM_CANDIDATES}, got {num_candidates}."
        )
    return MASTER_MEANS_16[:num_candidates].copy()


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the toy 2D Gaussian experiment."""

    means: np.ndarray
    sigma_g: float = 0.40
    sigma_q: float = 0.40
    kernel_bandwidth: float = 1.25
    s: int = 3
    beta: float = 4.0
    T: int = 2000
    reps: int = 50
    seed: int = 12345
    n_0: int = 2
    c_eta: float = 0.05
    a_eta: float = 0.0

    # MMD target uses generators 3, 5, and 7 in 1-indexed notation.
    # Stored here as zero-indexed generator indices.  These are present for
    # all supported candidate-set sizes 8, 12, and 16.
    target_indices: tuple[int, int, int] = (2, 4, 6)
    target_weights: tuple[float, float, float] = (0.45, 0.35, 0.20)

    @property
    def m(self) -> int:
        return int(self.means.shape[0])

    @property
    def d(self) -> int:
        return int(self.means.shape[1])

    @property
    def cov_g(self) -> np.ndarray:
        return (self.sigma_g ** 2) * np.eye(self.d)

    def validate(self) -> None:
        if self.T < self.m:
            raise ValueError(f"Need T >= m for initialization; got T={self.T}, m={self.m}.")
        if not (1 <= self.s <= self.m):
            raise ValueError(f"Need 1 <= s <= m; got s={self.s}, m={self.m}.")
        if max(self.target_indices) >= self.m:
            raise ValueError(
                f"MMD target indices {self.target_indices} require at least {max(self.target_indices)+1} candidates."
            )


def default_config(
    T: int = 2000,
    reps: int = 50,
    s: int = 3,
    beta: float = 4.0,
    seed: int = 12345,
    m: int = 8,
) -> ExperimentConfig:
    """Build the default experiment configuration.

    Parameters
    ----------
    m:
        Number of candidate Gaussian generators.  Supported values are 8, 12,
        and 16.  The 8-candidate setting is the original toy design.
    """

    cfg = ExperimentConfig(means=fixed_means(m), T=T, reps=reps, s=s, beta=beta, seed=seed)
    cfg.validate()
    return cfg
