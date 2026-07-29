from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from .algorithms import DISPLAY_NAMES, run_one_algorithm, run_standalone_bandit
from .config import default_config, VALID_NUM_CANDIDATES
from .objectives import make_objective
from .plotting import (
    plot_final_mixtures,
    plot_geometry,
    plot_regret_curves,
    plot_s_sweep_summary,
    plot_weighted_mixture,
)
from .solvers import num_active_subsets


METHODS = ["fc_sparse", "full", "coherence"]


def sample_bank_for_rep(config, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Slightly larger than T to be safe. In T total pulls, no arm can need more than T samples.
    return rng.normal(
        loc=config.means[:, None, :],
        scale=config.sigma_g,
        size=(config.m, config.T + 5, config.d),
    )


def summarize_regrets(regret_runs: dict[str, list[np.ndarray]], T: int, m: int, s: int) -> pd.DataFrame:
    rows = []
    for method, curves in regret_runs.items():
        arr = np.vstack(curves)
        mean = arr.mean(axis=0)
        se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros(T)
        for t in range(1, T + 1):
            rows.append(
                {
                    "m": m,
                    "s": s,
                    "t": t,
                    "method": method,
                    "display_name": DISPLAY_NAMES.get(method, method),
                    "mean_regret": mean[t - 1],
                    "se_regret": se[t - 1],
                    "n_reps": arr.shape[0],
                }
            )
    return pd.DataFrame(rows)


def summarize_final_weights(final_runs: dict[str, list[np.ndarray]], m: int, s: int) -> pd.DataFrame:
    rows = []
    for method, weights in final_runs.items():
        arr = np.vstack(weights)
        mean = arr.mean(axis=0)
        se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros(m)
        for arm in range(m):
            rows.append(
                {
                    "m": m,
                    "s": s,
                    "method": method,
                    "display_name": DISPLAY_NAMES.get(method, method),
                    "arm": arm + 1,
                    "mean_weight": mean[arm],
                    "se_weight": se[arm],
                    "n_reps": arr.shape[0],
                }
            )
    return pd.DataFrame(rows)


def summarize_final_counts(count_runs: dict[str, list[np.ndarray]], m: int, s: int) -> pd.DataFrame:
    rows = []
    for method, counts in count_runs.items():
        arr = np.vstack(counts)
        mean = arr.mean(axis=0)
        se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros(m)
        for arm in range(m):
            rows.append(
                {
                    "m": m,
                    "s": s,
                    "method": method,
                    "display_name": DISPLAY_NAMES.get(method, method),
                    "arm": arm + 1,
                    "mean_final_count": mean[arm],
                    "se_final_count": se[arm],
                    "n_reps": arr.shape[0],
                }
            )
    return pd.DataFrame(rows)


def run_objective(objective_name: str, config, out_dir: Path) -> pd.DataFrame:
    objective = make_objective(objective_name, config)
    sparse_supports = num_active_subsets(config.m, config.s)
    print(f"\n=== Objective: {objective.name.upper()} | m={config.m}, s={config.s} ===")
    print(f"Exact sparse MIQP enumeration checks {sparse_supports} supports per optimization.")
    print(f"Sparse oracle loss: {objective.loss_sparse_star:.6f}")
    print(f"Sparse oracle alpha: {np.round(objective.alpha_sparse_star, 4)}")
    print(f"Full oracle loss:   {objective.loss_full_star:.6f}")
    print(f"Full oracle alpha:  {np.round(objective.alpha_full_star, 4)}")
    print(
        f"Best standalone: arm {objective.best_standalone_index + 1}, "
        f"loss={objective.best_standalone_loss:.6f}, "
        f"regret={objective.standalone_regret:.6f}"
    )

    regret_runs: dict[str, list[np.ndarray]] = {m: [] for m in METHODS + ["standalone"]}
    final_runs: dict[str, list[np.ndarray]] = {m: [] for m in METHODS + ["standalone"]}
    count_runs: dict[str, list[np.ndarray]] = {m: [] for m in METHODS + ["standalone"]}

    start = time.time()
    for rep in tqdm(range(config.reps), desc=f"{objective.name.upper()} m={config.m} s={config.s}"):
        bank_seed = config.seed + 10_000 * rep
        bank = sample_bank_for_rep(config, bank_seed)

        for method_index, method in enumerate(METHODS):
            alg_seed = config.seed + 10_000 * rep + 100 * (method_index + 1)
            rng = np.random.default_rng(alg_seed)
            result = run_one_algorithm(method, bank, config, objective, rng)
            regret_runs[method].append(result.regret_curve)
            final_runs[method].append(result.final_alpha)
            count_runs[method].append(result.final_counts)

        standalone = run_standalone_bandit(bank, config, objective)
        regret_runs["standalone"].append(standalone.regret_curve)
        final_runs["standalone"].append(standalone.final_alpha)
        count_runs["standalone"].append(standalone.final_counts)

    elapsed = time.time() - start
    print(f"Finished {objective.name.upper()} m={config.m} s={config.s} in {elapsed:.1f} seconds")

    regret_df = summarize_regrets(regret_runs, config.T, config.m, config.s)
    final_df = summarize_final_weights(final_runs, config.m, config.s)
    counts_df = summarize_final_counts(count_runs, config.m, config.s)

    out_dir.mkdir(parents=True, exist_ok=True)
    regret_csv = out_dir / f"{objective.name}_regret_curves.csv"
    final_csv = out_dir / f"{objective.name}_final_weights.csv"
    counts_csv = out_dir / f"{objective.name}_final_counts.csv"
    oracle_json = out_dir / f"{objective.name}_oracle.json"

    regret_df.to_csv(regret_csv, index=False)
    final_df.to_csv(final_csv, index=False)
    counts_df.to_csv(counts_csv, index=False)

    oracle_payload = {
        "objective": objective.name,
        "m": config.m,
        "s": config.s,
        "num_sparse_supports_checked_per_exact_optimization": sparse_supports,
        "delta_kappa": objective.delta_kappa,
        "delta_f": objective.delta_f,
        "delta_L": objective.delta_L,
        "alpha_sparse_star": objective.alpha_sparse_star.tolist(),
        "loss_sparse_star": objective.loss_sparse_star,
        "alpha_full_star": objective.alpha_full_star.tolist(),
        "loss_full_star": objective.loss_full_star,
        "best_standalone_index_1based": objective.best_standalone_index + 1,
        "best_standalone_loss": objective.best_standalone_loss,
        "best_standalone_regret": objective.standalone_regret,
        "K": objective.K.tolist(),
        "f_true": objective.f_true.tolist(),
        "const": objective.const,
    }
    oracle_json.write_text(json.dumps(oracle_payload, indent=2))

    plot_geometry(config, objective, out_dir / f"{objective.name}_geometry.png")
    plot_weighted_mixture(
        config,
        objective.alpha_sparse_star,
        f"True sparse oracle mixture ({objective.name.upper()}, m={config.m}, s={config.s})",
        out_dir / f"{objective.name}_oracle_sparse.png",
    )
    plot_regret_curves(regret_df, objective, out_dir / f"{objective.name}_regret.png")
    plot_final_mixtures(config, final_df, objective, out_dir / f"{objective.name}_final_mixtures.png")

    print(f"Wrote {regret_csv}")
    print(f"Wrote {final_csv}")
    print(f"Wrote {counts_csv}")
    print(f"Wrote figures to {out_dir}")

    final_rows = []
    for method, curves in regret_runs.items():
        arr = np.vstack(curves)
        vals = arr[:, -1]
        final_rows.append(
            {
                "objective": objective.name,
                "m": config.m,
                "s": config.s,
                "method": method,
                "display_name": DISPLAY_NAMES.get(method, method),
                "T": config.T,
                "reps": config.reps,
                "final_mean_regret": float(vals.mean()),
                "final_se_regret": float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0,
                "loss_sparse_star": objective.loss_sparse_star,
                "loss_full_star": objective.loss_full_star,
                "best_standalone_index_1based": objective.best_standalone_index + 1,
            }
        )
    return pd.DataFrame(final_rows)


def parse_s_values(text: str | None, fallback_s: int) -> list[int]:
    if text is None or str(text).strip() == "":
        return [int(fallback_s)]
    vals = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    if not vals:
        return [int(fallback_s)]
    return sorted(set(vals))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Sparse Mixture-UCB 2D Gaussian toy experiments.")
    parser.add_argument("--objective", choices=["rke", "mmd", "both"], default="both")
    parser.add_argument("--T", type=int, default=2000, help="Horizon")
    parser.add_argument("--reps", type=int, default=50, help="Monte Carlo repetitions")
    parser.add_argument("--m", type=int, choices=list(VALID_NUM_CANDIDATES), default=8, help="Number of Gaussian candidates: 8, 12, or 16")
    parser.add_argument("--s", type=int, default=3, help="Single sparsity level, used when --s-values is omitted")
    parser.add_argument("--s-values", type=str, default=None, help="Comma-separated sparsity sweep, e.g. '1,2,3,4,5'")
    parser.add_argument("--beta", type=float, default=4.0, help="Confidence parameter")
    parser.add_argument("--seed", type=int, default=12345, help="Base random seed")
    parser.add_argument("--out", type=str, default="results", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    s_values = parse_s_values(args.s_values, args.s)
    objectives = ["rke", "mmd"] if args.objective == "both" else [args.objective]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    sweep_rows: list[pd.DataFrame] = []
    multi_s = len(s_values) > 1
    for s in s_values:
        config = default_config(T=args.T, reps=args.reps, s=s, beta=args.beta, seed=args.seed, m=args.m)
        this_out = out_root / f"m{config.m:02d}_s{s:02d}" if multi_s else out_root
        this_out.mkdir(parents=True, exist_ok=True)

        config_payload = {
            "T": config.T,
            "reps": config.reps,
            "m": config.m,
            "s": config.s,
            "beta": config.beta,
            "seed": config.seed,
            "sigma_g": config.sigma_g,
            "sigma_q": config.sigma_q,
            "kernel_bandwidth": config.kernel_bandwidth,
            "means": config.means.tolist(),
            "target_indices_1based": [i + 1 for i in config.target_indices],
            "target_weights": list(config.target_weights),
            "methods": METHODS + ["standalone"],
            "num_sparse_supports_checked_per_exact_optimization": num_active_subsets(config.m, config.s),
        }
        (this_out / "config.json").write_text(json.dumps(config_payload, indent=2))

        for obj in objectives:
            summary = run_objective(obj, config, this_out)
            sweep_rows.append(summary)

    if sweep_rows:
        sweep_df = pd.concat(sweep_rows, ignore_index=True)
        sweep_csv = out_root / "sweep_summary.csv"
        sweep_df.to_csv(sweep_csv, index=False)
        print(f"\nWrote sweep summary to {sweep_csv}")
        if len(s_values) > 1:
            for obj in objectives:
                plot_s_sweep_summary(sweep_df[sweep_df["objective"] == obj], out_root / f"{obj}_s_sweep_final_regret.png")


if __name__ == "__main__":
    main()
