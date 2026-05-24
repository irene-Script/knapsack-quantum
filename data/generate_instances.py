"""
generate_instances.py
---------------------
Generates 0-1 Knapsack Problem instances and saves them as JSON files.

Instance naming: kp_n{n}_id{id}.json
Sizes (normal): n in [10, 15, 20, 30, 50, 100], 5 instances each
Sizes (light):  n in [5, 8, 10],                3 instances each  (--light flag)

Usage:
    python data/generate_instances.py           # full dataset
    python data/generate_instances.py --light   # lightweight for limited hardware
"""

import argparse
import json
import os
import random
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────
SIZES_FULL  = [10, 15, 20, 30, 50, 100]   # full benchmark
SIZES_LIGHT = [5, 8, 10]                   # lightweight (all quantum solvers feasible)
N_INSTS_FULL  = 5                          # instances per size (full)
N_INSTS_LIGHT = 3                          # instances per size (light)
SEED    = 42                               # reproducibility
W_MIN, W_MAX = 1, 100                      # weight range
P_MIN, P_MAX = 1, 100                      # profit range
CAPACITY_RATIO = 0.5                       # W = ratio * sum(weights)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "instances")


def generate_instance(n: int, seed: int) -> dict:
    """
    Generate a random KP instance.

    Args:
        n    : number of items
        seed : random seed for reproducibility

    Returns:
        dict with keys: n, weights, profits, capacity, optimal (None until solved)
    """
    rng     = np.random.default_rng(seed)
    weights = rng.integers(W_MIN, W_MAX + 1, size=n).tolist()
    profits = rng.integers(P_MIN, P_MAX + 1, size=n).tolist()
    capacity = int(CAPACITY_RATIO * sum(weights))

    return {
        "n"        : n,
        "weights"  : weights,
        "profits"  : profits,
        "capacity" : capacity,
        "optimal"  : None,   # filled by dp_solver.py when run
        "seed"     : seed
    }


def solve_dp(instance: dict) -> int:
    """
    Solve the KP instance with dynamic programming (exact).
    Used to pre-fill the 'optimal' field in each instance file.

    Time: O(n * W)
    """
    n  = instance["n"]
    W  = instance["capacity"]
    w  = instance["weights"]
    p  = instance["profits"]

    # dp[i][c] = max profit using items 0..i-1 with capacity c
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for c in range(W + 1):
            dp[i][c] = dp[i-1][c]
            if w[i-1] <= c:
                dp[i][c] = max(dp[i][c], dp[i-1][c - w[i-1]] + p[i-1])
    return dp[n][W]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate KP instances (full or lightweight)"
    )
    parser.add_argument(
        '--light', action='store_true',
        help='Generate a small dataset (n in [5,8,10], 3 instances each) '
             'suitable for PCs with limited memory / time'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.light:
        sizes   = SIZES_LIGHT
        n_insts = N_INSTS_LIGHT
        print("Mode LIGHT — instances réduits (n ∈ {5,8,10}, 3 par taille)")
    else:
        sizes   = SIZES_FULL
        n_insts = N_INSTS_FULL
        print("Mode FULL  — benchmark complet (n ∈ {10,15,20,30,50,100}, 5 par taille)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng_master = random.Random(SEED)

    summary = []

    for n in sizes:
        for inst_id in range(n_insts):
            seed = rng_master.randint(0, 10_000_000)
            inst = generate_instance(n, seed)

            # Solve exactly with DP and store the optimal value
            inst["optimal"] = solve_dp(inst)

            fname = f"kp_n{n:03d}_id{inst_id:02d}.json"
            fpath = os.path.join(OUTPUT_DIR, fname)
            with open(fpath, "w") as f:
                json.dump(inst, f, indent=2)

            summary.append({
                "file"    : fname,
                "n"       : n,
                "capacity": inst["capacity"],
                "optimal" : inst["optimal"]
            })
            print(f"  Generated {fname}  |  W={inst['capacity']:5d}  |  opt={inst['optimal']:6d}")

    # Save summary CSV
    import csv
    csv_path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "n", "capacity", "optimal"])
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nDone. {len(summary)} instances saved to '{OUTPUT_DIR}/'")
    print(f"Summary written to '{csv_path}'")


if __name__ == "__main__":
    main()
