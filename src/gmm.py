"""Core GMM types: BaseGaussian and Mixture."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class BaseGaussian:
    """A single 2D isotropic Gaussian component.

    Parameters
    ----------
    mean : (2,) ndarray
        The mean vector.
    sigma : float
        Standard deviation (covariance = sigma² * I).
    index : int
        Identifier (0..15 for the 16 base components).
    """

    def __init__(self, mean: NDArray[np.floating], sigma: float, index: int):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.sigma = float(sigma)
        self.index = int(index)

    @property
    def cov(self) -> NDArray[np.floating]:
        """2x2 isotropic covariance matrix: sigma² * I."""
        return np.eye(2, dtype=np.float64) * (self.sigma**2)

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.floating]:
        """Draw *n* i.i.d. samples from this Gaussian.

        Returns (n, 2) array.
        """
        return rng.multivariate_normal(self.mean, self.cov, size=n)

    def __repr__(self) -> str:
        return f"BaseGaussian(index={self.index}, mean={self.mean.round(4)}, sigma={self.sigma})"


class Mixture:
    """A weighted mixture of Gaussian components.

    Parameters
    ----------
    components : list of (weight, BaseGaussian)
        Weighted components. Weights are normalised automatically.
    mixture_type : int
        1 (single base), 2 (pair), or 3 (quartet).
    label : str
        Human-readable identifier, e.g. ``"type2_3_7"``.
    """

    def __init__(
        self,
        components: list[tuple[float, BaseGaussian]],
        mixture_type: int,
        label: str,
    ):
        self._raw_components = components
        self.mixture_type = int(mixture_type)
        self.label = str(label)

        # Normalise weights
        total = sum(w for w, _ in components)
        self.components: list[tuple[float, BaseGaussian]] = [
            (w / total, g) for w, g in components
        ]

    # -- Derived properties ---------------------------------------------------

    @property
    def num_components(self) -> int:
        """Number of Gaussian components in this mixture."""
        return len(self.components)

    @property
    def weights(self) -> NDArray[np.floating]:
        """(K,) normalised weight vector."""
        return np.array([w for w, _ in self.components], dtype=np.float64)

    @property
    def component_indices(self) -> NDArray[np.integer]:
        """(K,) array of base-Gaussian indices used by this mixture."""
        return np.array([g.index for _, g in self.components], dtype=np.int32)

    @property
    def means(self) -> NDArray[np.floating]:
        """(K, 2) array of component means."""
        return np.array([g.mean for _, g in self.components], dtype=np.float64)

    @property
    def sigma(self) -> float:
        """Sigma of the first component (all share the same sigma by design)."""
        return float(self.components[0][1].sigma)

    # -- Sampling -------------------------------------------------------------

    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.floating]:
        """Draw *n* i.i.d. samples from the mixture.

        1. Choose a component according to the weights.
        2. Sample from the selected Gaussian.

        Returns (n, 2) array.
        """
        if rng is None:
            rng = np.random.default_rng()

        weights = self.weights
        # How many samples to draw from each component
        counts = rng.multinomial(n, weights)
        samples = np.empty((n, 2), dtype=np.float64)
        pos = 0
        for k, (_, gaussian) in enumerate(self.components):
            c = counts[k]
            if c > 0:
                samples[pos : pos + c] = gaussian.sample(c, rng)
                pos += c
        return samples

    # -- Metadata for .npz ----------------------------------------------------

    def metadata(self) -> dict:
        """Return a dict suitable for storing alongside samples in an .npz."""
        return {
            "mixture_type": np.array(self.mixture_type, dtype=np.int32),
            "component_indices": self.component_indices,
            "weights": self.weights,
            "means": self.means,
            "sigma": np.array(self.sigma, dtype=np.float64),
            "label": np.array(self.label, dtype=str),
        }

    def __repr__(self) -> str:
        idx_str = "_".join(str(i) for i in self.component_indices)
        return f"Mixture(type={self.mixture_type}, label={self.label}, indices=[{idx_str}])"
