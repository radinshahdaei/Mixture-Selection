"""Sample generation and .npz I/O."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from .config import Config
from .gmm import Mixture


class SampleManager:
    """Generates samples from mixtures and persists them as .npz files.

    Parameters
    ----------
    config : Config
        Full project configuration.
    """

    def __init__(self, config: Config):
        self.config = config
        self.rng = np.random.default_rng(config.gmm.random_seed)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def generate(
        self,
        mixture: Mixture,
        n_samples: Optional[int] = None,
    ) -> NDArray[np.floating]:
        """Draw i.i.d. samples from a mixture.

        Parameters
        ----------
        mixture : Mixture
            The mixture to sample from.
        n_samples : int, optional
            Number of samples (defaults to config value).

        Returns
        -------
        samples : (n_samples, 2) ndarray
        """
        if n_samples is None:
            n_samples = self.config.sampling.n_samples
        return mixture.sample(n_samples, rng=self.rng)

    # ------------------------------------------------------------------
    # .npz I/O
    # ------------------------------------------------------------------

    def save(
        self,
        samples: NDArray[np.floating],
        mixture: Mixture,
        filepath: Path,
    ) -> None:
        """Save samples and metadata to a .npz file.

        The file is self-describing: it includes all metadata needed to
        reconstruct the geometry without the original config.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        meta = mixture.metadata()
        # Add extra context
        meta["samples"] = samples.astype(np.float32)
        meta["n_components_base"] = np.array(self.config.gmm.n_components, dtype=np.int32)

        layout = self.config.gmm.layout
        meta["layout"] = np.array(layout, dtype=str)
        if layout == "ring":
            meta["radius"] = np.array(self.config.gmm.radius, dtype=np.float64)
        elif layout == "random_3d":
            meta["scale"] = np.array(self.config.gmm.random_3d_scale, dtype=np.float64)

        np.savez_compressed(filepath, **meta)

    def load(self, filepath: Path) -> dict[str, np.ndarray]:
        """Load a .npz file and return its contents as a dict."""
        data = np.load(filepath, allow_pickle=True)
        return dict(data)

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def generate_all(
        self,
        mixtures: dict[str, Mixture],
        output_dir: Optional[Path] = None,
    ) -> dict[str, Path]:
        """Generate and save samples for all mixtures.

        Parameters
        ----------
        mixtures : dict[str, Mixture]
            All mixtures keyed by label.
        output_dir : Path, optional
            Root output directory (defaults to config value).

        Returns
        -------
        manifest : dict[str, Path]
            Dictionary mapping mixture labels to their .npz file paths.
        """
        if output_dir is None:
            output_dir = Path(self.config.sampling.output_dir)

        n_samples = self.config.sampling.n_samples
        manifest: dict[str, Path] = {}

        # Group by type for organized output
        type_dirs = {1: "type1", 2: "type2", 3: "type3"}
        mixtures_by_type: dict[int, list[tuple[str, Mixture]]] = {1: [], 2: [], 3: []}
        for label, m in mixtures.items():
            mixtures_by_type[m.mixture_type].append((label, m))

        total = sum(len(v) for v in mixtures_by_type.values())
        pbar = tqdm(total=total, desc="Generating samples", unit="mixture")

        for mtype, dirname in type_dirs.items():
            type_dir = output_dir / dirname
            type_dir.mkdir(parents=True, exist_ok=True)
            for label, mixture in mixtures_by_type[mtype]:
                samples = self.generate(mixture, n_samples)
                filepath = type_dir / f"{label}.npz"
                self.save(samples, mixture, filepath)
                manifest[label] = filepath
                pbar.update(1)

        pbar.close()

        # Write manifest
        manifest_path = output_dir / "manifest.json"
        manifest_serializable = {k: str(v) for k, v in manifest.items()}
        manifest_path.write_text(json.dumps(manifest_serializable, indent=2))

        return manifest

    @staticmethod
    def load_manifest(samples_dir: Path) -> dict[str, Path]:
        """Load the manifest file, returning label → path mapping."""
        manifest_path = samples_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest found at {manifest_path}. Run sample generation first.")
        raw = json.loads(manifest_path.read_text())
        return {k: Path(v) for k, v in raw.items()}
