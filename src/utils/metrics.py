"""
metrics.py
----------
Evaluation metrics for comparing KP solvers.
"""

import numpy as np


def approximation_ratio(found: float, optimal: float) -> float:
    """AR = found / optimal  (1.0 = exact solution)."""
    if optimal == 0:
        return 1.0
    return found / optimal


def gap_percent(found: float, optimal: float) -> float:
    """Optimality gap in percent: (optimal - found) / optimal * 100."""
    if optimal == 0:
        return 0.0
    return (optimal - found) / optimal * 100.0


def aggregate_runs(results: list, optimal: float) -> dict:
    """
    Aggregate statistics over multiple independent runs of a solver.

    Args:
        results : list of result dicts (each with 'best_value', 'time_sec')
        optimal : exact optimal value (from DP)

    Returns:
        dict with: best, mean, std, ar_mean, gap_mean, time_mean, n_optimal
    """
    values = np.array([r["best_value"] for r in results])
    times  = np.array([r["time_sec"]   for r in results])

    n_exact = int(np.sum(np.isclose(values, optimal, rtol=1e-4)))

    return {
        "best"      : float(np.max(values)),
        "mean"      : float(np.mean(values)),
        "std"       : float(np.std(values)),
        "ar_mean"   : float(np.mean(values) / optimal) if optimal > 0 else 1.0,
        "gap_mean"  : gap_percent(float(np.mean(values)), optimal),
        "time_mean" : float(np.mean(times)),
        "n_runs"    : len(results),
        "n_optimal" : n_exact,
        "pct_optimal": 100.0 * n_exact / len(results)
    }
