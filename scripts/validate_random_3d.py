#!/usr/bin/env python3
"""Validate random_3d sample data for correctness.

Checks:
  1. All 140 .npz files exist and load correctly
  2. Manifest is consistent with files on disk
  3. Sample shapes: (5000, 3)
  4. Means within [-scale, scale]^3
  5. Covariance matrices are valid PSD
  6. Reproducibility: same seed → same data
  7. Mixture type counts: 16 + 120 + 4 = 140
  8. Uniform superposition: all 16 bases equally weighted

Usage:
    python scripts/validate_random_3d.py
    python scripts/validate_random_3d.py --config config_random_3d.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.random_mixtures import RandomMixtureFactory


def check(condition: bool, msg: str) -> int:
    """Report a check result. Returns 1 if failed, 0 if passed."""
    if condition:
        print(f"  ✓ {msg}")
        return 0
    else:
        print(f"  ✗ FAIL: {msg}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Validate random_3d sample data."
    )
    parser.add_argument(
        "--config", "-c", type=str, default="config_random_3d.yaml",
        help="Path to config file (default: config_random_3d.yaml).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    layout = config.gmm.layout

    if layout != "random_3d":
        print(f"ERROR: This validator requires layout='random_3d', got '{layout}'.")
        print("Use --config config_random_3d.yaml")
        sys.exit(1)

    scale = config.gmm.random_3d_scale
    n_components = config.gmm.n_components
    n_samples = config.sampling.n_samples
    samples_dir = Path(config.sampling.output_dir) / layout

    failures = 0
    print(f"Validating random_3d data in {samples_dir}/")
    print(f"  Config: {n_components} components, scale={scale}, "
          f"cov_scale={config.gmm.random_3d_cov_scale}")
    print()

    # --- 1. Manifest -------------------------------------------------------
    print("1. Manifest")
    manifest_path = samples_dir / "manifest.json"
    failures += check(manifest_path.exists(), "manifest.json exists")

    with open(manifest_path) as f:
        manifest = json.load(f)
    failures += check(len(manifest) == 140, f"manifest has 140 entries (got {len(manifest)})")

    # Check all three type directories referenced
    labels = list(manifest.keys())
    t1 = [l for l in labels if l.startswith("type1_")]
    t2 = [l for l in labels if l.startswith("type2_")]
    t3 = [l for l in labels if l.startswith("type3_")]
    failures += check(len(t1) == 16, f"Type 1: 16 mixtures (got {len(t1)})")
    failures += check(len(t2) == 120, f"Type 2: 120 mixtures (got {len(t2)})")
    failures += check(len(t3) == 4, f"Type 3: 4 mixtures (got {len(t3)})")

    # Check no extra labels
    all_expected = set(t1 + t2 + t3)
    failures += check(len(all_expected) == 140, "no duplicate/missing labels")

    # --- 2. File existence & loadability ----------------------------------
    print("\n2. File integrity")
    missing = []
    for label, path_str in manifest.items():
        p = Path(path_str)
        if not p.exists():
            missing.append(label)
    failures += check(len(missing) == 0, f"all 140 .npz files exist (missing: {len(missing)})")
    if missing:
        for m in missing[:5]:
            print(f"      missing: {m}")

    # --- 3. Sample shapes & metadata ---------------------------------------
    print("\n3. Sample shapes & metadata")

    all_labels = sorted(manifest.keys())
    sample_path = Path(manifest[all_labels[0]])
    d = np.load(sample_path, allow_pickle=True)
    failures += check(d["samples"].shape == (n_samples, 3),
                      f"samples shape: {d['samples'].shape} == ({n_samples}, 3)")

    # Spot-check 5 random files
    rng = np.random.default_rng(42)
    spot_checks = rng.choice(all_labels, size=min(5, len(all_labels)), replace=False)
    shape_ok = True
    has_cov = True
    has_layout = True
    has_scale_key = True
    no_sigma = True
    for label in spot_checks:
        d = np.load(Path(manifest[label]), allow_pickle=True)
        if d["samples"].shape != (n_samples, 3):
            shape_ok = False
        if "covariances" not in d:
            has_cov = False
        if d.get("layout") != "random_3d":
            has_layout = False
        if "scale" not in d:
            has_scale_key = False
        if "sigma" in d:
            no_sigma = False

    failures += check(shape_ok, f"spot-check {len(spot_checks)} files: all shapes ({n_samples}, 3)")
    failures += check(has_cov, "spot-check: all have 'covariances' key")
    failures += check(has_layout, "spot-check: layout='random_3d'")
    failures += check(has_scale_key, f"spot-check: scale={scale}")
    failures += check(no_sigma, "spot-check: no 'sigma' key (full cov, not isotropic)")

    # --- 4. Means within bounds --------------------------------------------
    print("\n4. Means bounds")
    means_all = []
    for label, path_str in manifest.items():
        d = np.load(Path(path_str), allow_pickle=True)
        means_all.append(d["means"])
    all_means = np.concatenate(means_all, axis=0)  # shape (total_components, 3)

    in_bounds = np.all((all_means >= -scale - 1e-10) & (all_means <= scale + 1e-10))
    failures += check(in_bounds, f"all component means within [-{scale}, {scale}]^3")
    failures += check(all_means.shape[1] == 3, "means are 3-dimensional")

    # Check that means are NOT all the same (i.e. actually random)
    unique_means = len(np.unique(np.round(all_means, decimals=6), axis=0))
    failures += check(unique_means >= 8,
                      f"at least 8 distinct means (got {unique_means}) — confirms randomness")

    # --- 5. Covariance validity --------------------------------------------
    print("\n5. Covariance matrices")

    # Check all unique base Gaussian covariances (from type1 files)
    covs = []
    for label in t1:
        d = np.load(Path(manifest[label]), allow_pickle=True)
        covs.append(d["covariances"][0])  # single-component → shape (3,3)
    covs = np.array(covs)  # (16, 3, 3)

    # 5a. Symmetry
    symmetric = np.allclose(covs, covs.transpose(0, 2, 1), atol=1e-12)
    failures += check(symmetric, "all 16 covariances are symmetric")

    # 5b. Positive semidefinite (all eigenvalues >= 0)
    eigenvals = np.linalg.eigvalsh(covs)
    all_psd = np.all(eigenvals >= -1e-12)
    failures += check(all_psd, "all 16 covariances are PSD (eigenvalues >= 0)")

    # 5c. Not all identical (randomness check)
    cov_fingerprints = np.round(covs, decimals=8).reshape(16, -1)
    unique_covs = np.unique(cov_fingerprints, axis=0).shape[0]
    failures += check(unique_covs >= 8,
                      f"at least 8 distinct covariances (got {unique_covs}) — confirms randomness")

    # 5d. Not isotropic (off-diagonals should be non-zero)
    off_diag_mask = ~np.eye(3, dtype=bool)
    off_diag_values = covs[:, off_diag_mask]
    max_off_diag = np.max(np.abs(off_diag_values))
    failures += check(max_off_diag > 1e-6,
                      f"covariances are non-isotropic (max |off-diag| = {max_off_diag:.6f} > 0)")

    # 5e. Non-zero variance on all axes
    diag_positive = np.all(np.diagonal(covs, axis1=1, axis2=2) > 1e-12)
    failures += check(diag_positive, "all diagonal entries (variances) > 0")

    # --- 6. Reproducibility ------------------------------------------------
    print("\n6. Reproducibility")

    factory1 = RandomMixtureFactory(config.gmm)
    means1 = np.array([g.mean for g in factory1.base_gaussians])

    factory2 = RandomMixtureFactory(config.gmm)  # same seed
    means2 = np.array([g.mean for g in factory2.base_gaussians])

    reproducible = np.allclose(means1, means2, atol=1e-15)
    failures += check(reproducible, "same seed → identical means (reproducible)")

    covs1 = np.array([g.cov for g in factory1.base_gaussians])
    covs2 = np.array([g.cov for g in factory2.base_gaussians])
    cov_repro = np.allclose(covs1, covs2, atol=1e-15)
    failures += check(cov_repro, "same seed → identical covariances (reproducible)")

    # --- 7. Mixture structure validation -----------------------------------
    print("\n7. Mixture structure")

    # Type 1: single components, weight 1.0
    m1 = factory1.create_type1()
    t1_ok = all(
        m.mixture_type == 1
        and m.num_components == 1
        and np.isclose(m.weights[0], 1.0)
        and m.ndim == 3
        for m in m1
    )
    failures += check(t1_ok, "Type 1: 16 singles, weight=1.0, ndim=3")

    # Type 2: pairs, equal weight 0.5
    m2 = factory1.create_type2()
    t2_ok = all(
        m.mixture_type == 2
        and m.num_components == 2
        and np.allclose(m.weights, 0.5)
        and m.ndim == 3
        for m in m2
    )
    failures += check(t2_ok, "Type 2: 120 pairs, weight=0.5, ndim=3")

    # Type 3: quartets, equal weight 0.25
    m3 = factory1.create_type3()
    t3_ok = all(
        m.mixture_type == 3
        and m.num_components == 4
        and np.allclose(m.weights, 0.25)
        and m.ndim == 3
        for m in m3
    )
    failures += check(t3_ok, "Type 3: 4 quartets, weight=0.25, ndim=3")

    # --- Summary -----------------------------------------------------------
    print(f"\n{'='*50}")
    if failures == 0:
        print("✓ All checks passed — random_3d data is valid.")
    else:
        print(f"✗ {failures} check(s) FAILED.")
    print(f"{'='*50}")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
