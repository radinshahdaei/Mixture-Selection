from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from .config import ExperimentConfig
from .objectives import ObjectiveSpec
from .algorithms import DISPLAY_NAMES


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_geometry(config: ExperimentConfig, objective: ObjectiveSpec, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    means = config.means
    ax.scatter(means[:, 0], means[:, 1], s=80, label="Generators")
    for i, mu in enumerate(means, start=1):
        ax.text(mu[0] + 0.06, mu[1] + 0.06, str(i), fontsize=10)
        ax.add_patch(Circle(mu, config.sigma_g, fill=False, alpha=0.35))

    if objective.name == "mmd":
        idx = np.array(config.target_indices, dtype=int)
        w = np.array(config.target_weights, dtype=float)
        ax.scatter(
            means[idx, 0],
            means[idx, 1],
            s=700 * w + 80,
            marker="*",
            label="MMD target components",
        )

    ax.set_title(f"2D Gaussian generator geometry ({objective.name.upper()})")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_weighted_mixture(config: ExperimentConfig, weights: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    means = config.means
    sizes = 80 + 1800 * weights
    ax.scatter(means[:, 0], means[:, 1], s=sizes)
    for i, (mu, wi) in enumerate(zip(means, weights), start=1):
        ax.text(mu[0] + 0.06, mu[1] + 0.06, f"{i}: {wi:.2f}", fontsize=9)
        ax.add_patch(Circle(mu, config.sigma_g, fill=False, alpha=0.25))
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_regret_curves(regret_df: pd.DataFrame, objective: ObjectiveSpec, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method, sub in regret_df.groupby("method", sort=False):
        label = DISPLAY_NAMES.get(method, method)
        x = sub["t"].to_numpy()
        mean = sub["mean_regret"].to_numpy()
        se = sub["se_regret"].to_numpy()
        ax.plot(x, mean, label=label)
        if method != "standalone":
            ax.fill_between(x, mean - se, mean + se, alpha=0.18)
    ax.axhline(0.0, linestyle="--", linewidth=1.0, label="Sparse oracle")
    ax.set_title(f"Regret curves ({objective.name.upper()})")
    ax.set_xlabel("horizon t")
    ax.set_ylabel(r"$\frac{1}{t}\sum_{r=1}^t L(q_r)-L(\alpha_s^\star)$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_final_mixtures(config: ExperimentConfig, final_df: pd.DataFrame, objective: ObjectiveSpec, out_path: Path) -> None:
    methods = list(final_df["method"].unique())
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), squeeze=False)
    means = config.means
    for ax, method in zip(axes[0], methods):
        sub = final_df[final_df["method"] == method].sort_values("arm")
        weights = sub["mean_weight"].to_numpy()
        sizes = 50 + 1200 * weights
        ax.scatter(means[:, 0], means[:, 1], s=sizes)
        for i, (mu, wi) in enumerate(zip(means, weights), start=1):
            ax.text(mu[0] + 0.06, mu[1] + 0.06, f"{i}\n{wi:.2f}", fontsize=8)
        ax.set_title(DISPLAY_NAMES.get(method, method))
        ax.axis("equal")
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"Average final deployed mixtures ({objective.name.upper()})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_s_sweep_summary(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Plot final regret at horizon T as a function of sparsity s."""

    if summary_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for method, sub in summary_df.groupby("method", sort=False):
        sub = sub.sort_values("s")
        x = sub["s"].to_numpy()
        mean = sub["final_mean_regret"].to_numpy()
        se = sub["final_se_regret"].to_numpy()
        ax.plot(x, mean, marker="o", label=DISPLAY_NAMES.get(method, method))
        if method != "standalone":
            ax.fill_between(x, mean - se, mean + se, alpha=0.15)
    objective = str(summary_df["objective"].iloc[0]).upper()
    m = int(summary_df["m"].iloc[0])
    T = int(summary_df["T"].iloc[0])
    ax.axhline(0.0, linestyle="--", linewidth=1.0, label="Sparse oracle")
    ax.set_title(f"Final regret vs sparsity ({objective}, m={m}, T={T})")
    ax.set_xlabel("sparsity level s")
    ax.set_ylabel("final mean regret")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
