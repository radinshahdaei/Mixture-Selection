#!/usr/bin/env bash
set -euo pipefail
python -m sparse_ucb.run_experiment --objective both --m 8 --s 3 --T 80 --reps 2 --out results_smoke
python -m sparse_ucb.run_experiment --objective both --m 16 --s-values 1,2,3 --T 40 --reps 1 --out results_sweep_smoke
