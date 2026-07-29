#!/usr/bin/env bash
set -euo pipefail
python -m sparse_ucb.run_experiment --objective both --m 8 --s 3 --T 2000 --reps 50 --out results
# Larger candidate-set sweep; can be slow because exact sparse enumerates supports.
# python -m sparse_ucb.run_experiment --objective both --m 16 --s-values 1,2,3,4,5 --T 2000 --reps 50 --out results_m16_sweep
