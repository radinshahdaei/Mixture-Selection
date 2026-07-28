"""Core GMM types: BaseGaussian and Mixture."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class BaseGaussian:
    """A single multivariate Gaussian component.

    Supports both isotropic (scalar sigma) and full-covariance construction.
    Dimensionality is inferred from the mean vector.

    Parameters
    ----------
    mean : (D,) ndarray
        The mean vector.
    sigma : float, optional
        Standard deviation for isotropic covariance (sigma² * I).
        Must be provided if ``cov`` is not.
    cov : (D, D) ndarray, optional
        Full covariance matrix. Takes precedence over ``sigma`` when provided.
    index : int
        Identifier for this component.
    """

    def __init__(
        self,
        mean: NDArray[np.floating],
        sigma: float | None = None,
        cov: NDArray[np.floating] | None = None,
        index: int = 0,
    ):
        if sigma is None and cov is None:
            raise ValueError("Either sigma or cov must be provided.")
        self.mean = np.asarray(mean, dtype=np.float64)
        self._cov = np.asarray(cov, dtype=np.float64) if cov is not None else None
        self.sigma = float(sigma) if sigma is not None else None
        self.index = int(index)

    @property
    def ndim(self) -> int:
        """Dimensionality of the Gaussian."""
        return len(self.mean)

    @property
    def cov(self) -> NDArray[np.floating]:
        """Covariance matrix (D×D).

        Returns the stored full covariance if provided, otherwise the
        isotropic sigma² * I matrix computed from ``sigma``.
        """
        if self._cov is not None:
            return self._cov
        if self.sigma is not None:
            return np.eye(self.ndim, dtype=np.float64) * (self.sigma**2)
        raise RuntimeError("Neither cov nor sigma is set.")

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.floating]:
        """Draw *n* i.i.d. samples from this Gaussian.

        Returns (n, D) array where D is the Gaussian dimensionality.
        """
        return rng.multivariate_normal(self.mean, self.cov, size=n)

    def __repr__(self) -> str:
        if self.sigma is not None:
            return f"BaseGaussian(index={self.index}, mean={self.mean.round(4)}, sigma={self.sigma})"
        else:
            return f"BaseGaussian(index={self.index}, mean={self.mean.round(4)}, cov_shape={self._cov.shape})"


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
        """(K, D) array of component means (D = dimensionality)."""
        return np.array([g.mean for _, g in self.components], dtype=np.float64)

    @property
    def sigma(self) -> float | None:
        """Sigma of the first component, or None if components use full covariances."""
        g = self.components[0][1]
        return float(g.sigma) if g.sigma is not None else None

    @property
    def ndim(self) -> int:
        """Dimensionality of this mixture's components."""
        return self.components[0][1].ndim

    # -- Sampling -------------------------------------------------------------

    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.floating]:
        """Draw *n* i.i.d. samples from the mixture.

        1. Choose a component according to the weights.
        2. Sample from the selected Gaussian.

        Returns (n, D) array where D is the component dimensionality.
        """
        if rng is None:
            rng = np.random.default_rng()

        weights = self.weights
        # How many samples to draw from each component
        counts = rng.multinomial(n, weights)
        ndim = self.ndim
        samples = np.empty((n, ndim), dtype=np.float64)
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
        meta = {
            "mixture_type": np.array(self.mixture_type, dtype=np.int32),
            "component_indices": self.component_indices,
            "weights": self.weights,
            "means": self.means,
            "label": np.array(self.label, dtype=str),
        }
        if self.sigma is not None:
            meta["sigma"] = np.array(self.sigma, dtype=np.float64)
        else:
            # Store full covariances for non-isotropic components
            covs = np.stack(
                [g._cov for _, g in self.components], axis=0
            )
            meta["covariances"] = covs.astype(np.float64)
        return meta

    def __repr__(self) -> str:
        idx_str = "_".join(str(i) for i in self.component_indices)
        return f"Mixture(type={self.mixture_type}, label={self.label}, indices=[{idx_str}])"
