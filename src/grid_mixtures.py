"""Grid mixture factory — creates all 265 candidate mixtures from 36 base
Gaussians arranged on a 6x6 lattice.

Type 1: 36 single-component mixtures (each base Gaussian).
Type 2: 225 four-component mixtures — all C(6,2) x C(6,2) combinations
        (choose 2 rows + 2 cols, the 4 intersection points).
Type 3:   4 nine-component mixtures — disjoint 3x3 blocks partitioning the grid.
Total:  265 candidates.
"""

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .config import GMMConfig
from .gmm import BaseGaussian, Mixture


class GridMixtureFactory:
    """Creates the three mixture types from base Gaussians on a 2D lattice.

    Parameters
    ----------
    config : GMMConfig
        Configuration specifying grid_rows, grid_cols, spacing, sigma, seed.
    """

    def __init__(self, config: GMMConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.n_rows = config.grid_rows
        self.n_cols = config.grid_cols
        self.spacing = config.grid_spacing

        # Build the N = n_rows * n_cols base Gaussians on the grid
        self.base_gaussians = self._create_base_gaussians()

    # ------------------------------------------------------------------
    # Base Gaussians
    # ------------------------------------------------------------------

    def _create_base_gaussians(self) -> list[BaseGaussian]:
        """Place N Gaussians on a centered n_rows x n_cols lattice.

        Row-major indexing: index = row * n_cols + col.
        The grid is centered at the origin.
        """
        sigma = self.config.sigma

        # Offsets to center the grid
        x_offset = (self.n_cols - 1) * self.spacing / 2.0
        y_offset = (self.n_rows - 1) * self.spacing / 2.0

        gaussians = []
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                x = col * self.spacing - x_offset
                y = row * self.spacing - y_offset
                idx = row * self.n_cols + col
                gaussians.append(
                    BaseGaussian(
                        mean=np.array([x, y], dtype=np.float64),
                        sigma=sigma,
                        index=idx,
                    )
                )
        return gaussians

    # ------------------------------------------------------------------
    # Type 1 — single-component mixtures (N = n_rows * n_cols)
    # ------------------------------------------------------------------

    def create_type1(self) -> list[Mixture]:
        """Each base Gaussian as its own single-component mixture.

        Returns n_rows * n_cols Mixture objects (36 for a 6x6 grid).
        """
        mixtures = []
        for g in self.base_gaussians:
            row = g.index // self.n_cols
            col = g.index % self.n_cols
            m = Mixture(
                components=[(1.0, g)],
                mixture_type=1,
                label=f"type1_{row}_{col}",
            )
            mixtures.append(m)
        return mixtures

    # ------------------------------------------------------------------
    # Type 2 — all C(n_rows,2) x C(n_cols,2) four-component mixtures
    # ------------------------------------------------------------------

    def create_type2(self) -> list[Mixture]:
        """All four-component mixtures formed by choosing 2 rows + 2 columns.

        For a 6x6 grid: C(6,2) x C(6,2) = 15 x 15 = 225 mixtures.
        Each mixture uses the 4 Gaussians at the intersection points,
        equally weighted at 0.25.
        """
        mixtures = []
        for r1, r2 in itertools.combinations(range(self.n_rows), 2):
            for c1, c2 in itertools.combinations(range(self.n_cols), 2):
                idx_11 = r1 * self.n_cols + c1
                idx_12 = r1 * self.n_cols + c2
                idx_21 = r2 * self.n_cols + c1
                idx_22 = r2 * self.n_cols + c2

                g_11 = self.base_gaussians[idx_11]
                g_12 = self.base_gaussians[idx_12]
                g_21 = self.base_gaussians[idx_21]
                g_22 = self.base_gaussians[idx_22]

                m = Mixture(
                    components=[(0.25, g_11), (0.25, g_12),
                                (0.25, g_21), (0.25, g_22)],
                    mixture_type=2,
                    label=f"type2_{r1}_{r2}_{c1}_{c2}",
                )
                mixtures.append(m)
        return mixtures

    # ------------------------------------------------------------------
    # Type 3 — disjoint 3x3 blocks (4 blocks for a 6x6 grid)
    # ------------------------------------------------------------------

    def create_type3(self) -> list[Mixture]:
        """Four disjoint 3x3 blocks partitioning the grid.

        Requires n_rows and n_cols to be divisible by 3.
        Blocks are indexed in row-major order over the 2x2 block grid.
        Each block contains 9 components, equally weighted at 1/9.
        """
        if self.n_rows % 3 != 0 or self.n_cols % 3 != 0:
            raise ValueError(
                f"grid_rows ({self.n_rows}) and grid_cols ({self.n_cols}) "
                f"must both be divisible by 3 for Type 3 mixtures."
            )

        block_rows = self.n_rows // 3
        block_cols = self.n_cols // 3

        mixtures = []
        for br in range(block_rows):
            for bc in range(block_cols):
                row_start = br * 3
                col_start = bc * 3

                comps = []
                for dr in range(3):
                    for dc in range(3):
                        r = row_start + dr
                        c = col_start + dc
                        idx = r * self.n_cols + c
                        comps.append((1.0 / 9.0, self.base_gaussians[idx]))

                block_idx = br * block_cols + bc
                m = Mixture(
                    components=comps,
                    mixture_type=3,
                    label=f"type3_block_{block_idx}",
                )
                mixtures.append(m)
        return mixtures

    # ------------------------------------------------------------------
    # All mixtures
    # ------------------------------------------------------------------

    def create_all(self) -> dict[str, Mixture]:
        """Return all 265 mixtures keyed by label.

        Returns
        -------
        dict[str, Mixture]
            Keys like ``"type1_0_0"``, ``"type2_0_1_2_3"``,
            ``"type3_block_0"``.
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
    # Convenience: uniform superposition (all N bases, equal weight)
    # ------------------------------------------------------------------

    def create_uniform_superposition(self) -> Mixture:
        """The uniform mixture over all N base Gaussians (the RKE optimum)."""
        n = len(self.base_gaussians)
        w = 1.0 / n
        return Mixture(
            components=[(w, g) for g in self.base_gaussians],
            mixture_type=0,  # special type for the superposition
            label="uniform_superposition",
        )
