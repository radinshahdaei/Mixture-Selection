"""Random 3D factory — creates N standalone 3D Gaussians with random means and
random full PSD covariances. No mixtures — each candidate is a single Gaussian."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import GMMConfig
from .gmm import BaseGaussian, Mixture


class RandomMixtureFactory:
    """Creates N standalone 3D Gaussians with random means and covariances.

    Each base Gaussian is independently sampled: its mean is drawn uniformly
    from [-scale, scale]^3, and its covariance is a random positive-definite
    matrix generated via A @ A^T where A ~ N(0, cov_scale, (3,3)).

    Unlike the ring and grid factories, this produces only Type-1 candidates
    (single Gaussians). There are no Type-2 pairs or Type-3 quartets.

    Parameters
    ----------
    config : GMMConfig
        Configuration specifying n_components, scale, cov_scale, seed.
    """

    def __init__(self, config: GMMConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.scale = config.random_3d_scale
        self.cov_scale = config.random_3d_cov_scale

        # Build the N base Gaussians with random means and covariances
        self.base_gaussians = self._create_base_gaussians()

    # ------------------------------------------------------------------
    # Base Gaussians
    # ------------------------------------------------------------------

    def _create_base_gaussians(self) -> list[BaseGaussian]:
        """Generate N 3D Gaussians with random means and full PSD covariances."""
        n = self.config.n_components
        scale = self.scale

        gaussians = []
        for k in range(n):
            # Random mean in [-scale, scale]^3
            mean = self.rng.uniform(-scale, scale, size=3)
            # Random PSD covariance matrix
            cov = self._random_psd_matrix()
            gaussians.append(BaseGaussian(mean=mean, cov=cov, index=k))
        return gaussians

    def _random_psd_matrix(self) -> NDArray[np.floating]:
        """Generate a random 3x3 positive-definite covariance matrix.

        Uses the Wishart-like construction: A @ A^T where A ~ N(0, s, (3,3)).
        """
        A = self.rng.normal(0.0, self.cov_scale, size=(3, 3))
        return A @ A.T

    # ------------------------------------------------------------------
    # Candidates — N single Gaussians (no mixtures)
    # ------------------------------------------------------------------

    def create_all(self) -> dict[str, Mixture]:
        """Return all N candidates keyed by label.

        Each candidate is a single Gaussian wrapped as a Mixture with weight 1.0.

        Returns
        -------
        dict[str, Mixture]
            Keys like ``"type1_0"``, ``"type1_1"``, ..., ``"type1_{N-1}"``.
        """
        all_mixtures: dict[str, Mixture] = {}

        for g in self.base_gaussians:
            m = Mixture(
                components=[(1.0, g)],
                mixture_type=1,
                label=f"type1_{g.index}",
            )
            all_mixtures[m.label] = m

        return all_mixtures

    # ------------------------------------------------------------------
    # Convenience: uniform superposition (all N bases, equal weight)
    # ------------------------------------------------------------------

    def create_uniform_superposition(self) -> Mixture:
        """The uniform mixture over all N base Gaussians (the RKE optimum)."""
        n = self.config.n_components
        w = 1.0 / n
        return Mixture(
            components=[(w, g) for g in self.base_gaussians],
            mixture_type=0,  # special type for the superposition
            label="uniform_superposition",
        )
