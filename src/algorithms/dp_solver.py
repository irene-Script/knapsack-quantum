"""
dp_solver.py
------------
Exact 0-1 Knapsack solver using Dynamic Programming.
Used as the optimal baseline for comparison.

Time complexity : O(n * W)
Space complexity: O(n * W)
"""

import time
import numpy as np


class DPSolver:
    """
    Exact DP solver for the 0-1 Knapsack Problem.

    Args:
        profits  : list[int] – profit of each item
        weights  : list[int] – weight of each item
        capacity : int       – knapsack capacity W
    """

    def __init__(self, profits: list, weights: list, capacity: int):
        self.profits  = profits
        self.weights  = weights
        self.capacity = capacity
        self.n        = len(profits)

    def solve(self) -> dict:
        """
        Run DP and return a result dict.

        Returns:
            {
              'best_value'   : int,
              'best_solution': list[int],   # binary selection vector
              'time_sec'     : float,
              'method'       : 'DP'
            }
        """
        n, W = self.n, self.capacity
        w, p = self.weights, self.profits

        t0 = time.perf_counter()

        # Build DP table
        dp = [[0] * (W + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for c in range(W + 1):
                dp[i][c] = dp[i-1][c]
                if w[i-1] <= c:
                    dp[i][c] = max(dp[i][c], dp[i-1][c - w[i-1]] + p[i-1])

        best_value = dp[n][W]

        # Backtrack to find selected items
        solution = [0] * n
        c = W
        for i in range(n, 0, -1):
            if dp[i][c] != dp[i-1][c]:
                solution[i-1] = 1
                c -= w[i-1]

        elapsed = time.perf_counter() - t0

        return {
            "best_value"    : best_value,
            "best_solution" : solution,
            "time_sec"      : elapsed,
            "method"        : "DP"
        }
