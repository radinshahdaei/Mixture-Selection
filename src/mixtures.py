"""Ring mixture factory — creates all 140 candidate mixtures from 16 base
Gaussians placed uniformly on a circle."""

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .config import GMMConfig
from .gmm import BaseGaussian, Mixture


class RingMixtureFactory:
    """Creates the three mixture types from base Gaussians on a ring.

    Parameters
    ----------
    config : GMMConfig
        Configuration specifying N, radius, sigma, seed.
    """

    def __init__(self, config: GMMConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

        # Build the N base Gaussians on the circle
        self.base_gaussians = self._create_base_gaussians()

    # ------------------------------------------------------------------
    # Base Gaussians
    # ------------------------------------------------------------------

    def _create_base_gaussians(self) -> list[BaseGaussian]:
        """Place N Gaussians uniformly on a circle of radius R."""
        n = self.config.n_components
        R = self.config.radius
        sigma = self.config.sigma

        gaussians = []
        for k in range(n):
            theta = 2.0 * np.pi * k / n
            mean = np.array([R * np.cos(theta), R * np.sin(theta)], dtype=np.float64)
            gaussians.append(BaseGaussian(mean=mean, sigma=sigma, index=k))
        return gaussians

    # ------------------------------------------------------------------
    # Type 1 — N single-component mixtures
    # ------------------------------------------------------------------

    def create_type1(self) -> list[Mixture]:
        """Each base Gaussian as its own single-component mixture.

        Returns 16 Mixture objects (one per base Gaussian).
        """
        mixtures = []
        for g in self.base_gaussians:
            m = Mixture(
                components=[(1.0, g)],
                mixture_type=1,
                label=f"type1_{g.index}",
            )
            mixtures.append(m)
        return mixtures

    # ------------------------------------------------------------------
    # Type 2 — all C(N, 2) two-component mixtures
    # ------------------------------------------------------------------

    def create_type2(self) -> list[Mixture]:
        """All two-component equally-weighted mixtures.

        Returns C(N,2) = 120 mixtures for N=16.
        """
        n = self.config.n_components
        mixtures = []
        for i, j in itertools.combinations(range(n), 2):
            g_i = self.base_gaussians[i]
            g_j = self.base_gaussians[j]
            m = Mixture(
                components=[(0.5, g_i), (0.5, g_j)],
                mixture_type=2,
                label=f"type2_{i}_{j}",
            )
            mixtures.append(m)
        return mixtures

    # ------------------------------------------------------------------
    # Type 3 — N/4 four-component mixtures (disjoint quartets)
    # ------------------------------------------------------------------

    def create_type3(self) -> list[Mixture]:
        """Four-component mixtures from disjoint consecutive quartets.

        Partitions indices 0..N-1 into N/4 groups of 4 consecutive indices.
        For N=16: {0,1,2,3}, {4,5,6,7}, {8,9,10,11}, {12,13,14,15}.
        Returns 4 mixtures.
        """
        n = self.config.n_components
        q = 4  # quartet size
        n_quartets = n // q

        if n % q != 0:
            raise ValueError(
                f"n_components ({n}) must be divisible by 4 for Type 3 mixtures."
            )

        mixtures = []
        for t in range(n_quartets):
            indices = list(range(t * q, (t + 1) * q))
            comps = [(0.25, self.base_gaussians[idx]) for idx in indices]
            idx_str = "_".join(str(i) for i in indices)
            m = Mixture(
                components=comps,
                mixture_type=3,
                label=f"type3_{idx_str}",
            )
            mixtures.append(m)
        return mixtures

    # ------------------------------------------------------------------
    # All mixtures
    # ------------------------------------------------------------------

    def create_all(self) -> dict[str, Mixture]:
        """Return all 140 mixtures keyed by label.

        Returns
        -------
        dict[str, Mixture]
            Keys like ``"type1_0"``, ``"type2_3_7"``, ``"type3_0_1_2_3"``.
        """
        all_mixtures: dict[str, Mixture] = {}

        for m in self.create_type1():
            all_mixtures[m.label] = m
        for m in self.create_type2():
            all_mixtures[m.label] = m
        for m in self.create_type3():
            all_mixtures[m.label] = m

        return all_mixtures

    # ------------------------------------------------------------------
    # Convenience: uniform superposition (all 16 bases, equal weight)
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


# Backward-compatible alias
MixtureFactory = RingMixtureFactory
