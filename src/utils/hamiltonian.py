"""
hamiltonian.py
--------------
Builds the Ising Hamiltonian for the 0-1 Knapsack Problem.

The KP maximization:
    max  sum_i p_i * x_i
    s.t. sum_i w_i * x_i <= W,  x_i in {0,1}

is converted to QUBO minimization:
    min  -sum_i p_i * x_i  +  lambda * (sum_i w_i * x_i - W)^2

Then substituting x_i = (1 - z_i) / 2  with z_i in {-1, +1}:
    H_C = sum_i h_i * Z_i  +  sum_{i<j} J_ij * Z_i Z_j  + constant

Returned as a Qiskit SparsePauliOp.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp


def build_hamiltonian(
    profits: list,
    weights: list,
    capacity: int,
    lam: float = None
) -> tuple:
    """
    Build the Ising Hamiltonian H_C as a SparsePauliOp.

    Args:
        profits  : list[float] – p_i
        weights  : list[float] – w_i
        capacity : float       – W
        lam      : float       – penalty coefficient lambda
                                 Default: 1.5 * sum(profits)  (ensures lam > max feasible profit)

    Returns:
        (hamiltonian: SparsePauliOp, offset: float)
        offset = constant term (does not affect optimization, kept for energy reference)
    """
    n  = len(profits)
    p  = np.array(profits,  dtype=float)
    w  = np.array(weights,  dtype=float)
    W  = float(capacity)

    # Auto-tune lambda: must dominate the maximum achievable profit (sum of all profits)
    # so that any infeasible solution has higher energy than any feasible one.
    # Sufficient condition: lam > sum(profits).  Using 1.5x as safety margin.
    if lam is None:
        lam = 1.5 * float(np.sum(p))

    # ── Linear coefficients h_i ──────────────────────────────────────────────
    # Derivation: substitute x_i = (1 - z_i)/2 into QUBO Q(x).
    #
    # QUBO linear term a_i = -p_i + lam*w_i^2 - 2*lam*W*w_i  →  contributes
    # -a_i/2 to z_i, and cross-term (λ/2)Σ_{i<j}w_i w_j contributes -(λ/2)w_i(sum_w-w_i)/2.
    # Both together give:
    #   h_i = p_i/2 + lam * w_i * (W - sum_w/2)
    #
    # Sign check: if W > sum_w/2 and p_i > 0 then h_i > 0, so z_i = -1
    # (i.e. x_i = 1, item selected) lowers energy ↔ increases profit ✓
    sum_w = float(np.sum(w))
    h = p / 2.0 + lam * w * (W - sum_w / 2.0)

    # ── Quadratic coefficients J_ij ──────────────────────────────────────────
    # From (λ/2) * Σ_{i<j} w_i * w_j * z_i * z_j
    #   J_ij = lam * w_i * w_j / 2   (i < j)
    J = np.outer(w, w) * lam / 2.0   # full matrix; use upper triangle

    # ── Constant (energy offset) ─────────────────────────────────────────────
    offset = (lam * (sum_w / 2.0 - W) ** 2
              - np.sum(p) / 2.0
              + lam * np.sum(w ** 2) / 4.0)

    # ── Assemble SparsePauliOp ───────────────────────────────────────────────
    pauli_list = []

    # Linear terms: h_i * Z_i
    for i in range(n):
        if abs(h[i]) > 1e-10:
            # Qiskit convention: rightmost character = qubit 0
            ops          = ['I'] * n
            ops[n-1 - i] = 'Z'
            pauli_list.append((''.join(ops), h[i]))

    # Quadratic terms: J_ij * Z_i Z_j  (i < j)
    for i in range(n):
        for j in range(i + 1, n):
            coeff = J[i, j]
            if abs(coeff) > 1e-10:
                ops          = ['I'] * n
                ops[n-1 - i] = 'Z'
                ops[n-1 - j] = 'Z'
                pauli_list.append((''.join(ops), coeff))

    hamiltonian = SparsePauliOp.from_list(pauli_list)
    return hamiltonian, offset


def energy_to_profit(energy: float, offset: float) -> float:
    """
    Convert a raw Ising energy (SparsePauliOp expectation) to the equivalent
    KP profit.  Only valid for feasible solutions (penalty = 0).

    Derivation: Q(x) = H_Ising(z) + offset  and  Q(x) = -profit (feasible)
    => profit = -(energy - offset) * 2     [factor 2 from x=(1-z)/2 scaling]

    In practice callers read the bitstring and compute profit directly.
    This helper is used for numerical sanity checks.
    """
    return -(energy - offset) * 2.0


def bitstring_to_solution(bitstring: str) -> list:
    """
    Convert a Qiskit measurement bitstring to a binary list.

    Qiskit returns bitstrings with qubit 0 on the RIGHT.
    e.g. '1010' -> [0, 1, 0, 1] (qubit 0 first)

    Args:
        bitstring: str, e.g. '01101'
    Returns:
        list[int] in item order (item 0 first)
    """
    return [int(b) for b in reversed(bitstring)]


def solution_profit(solution: list, profits: list) -> float:
    """Compute the total profit of a binary selection vector."""
    return float(sum(p * x for p, x in zip(profits, solution)))


def solution_weight(solution: list, weights: list) -> float:
    """Compute the total weight of a binary selection vector."""
    return float(sum(w * x for w, x in zip(weights, solution)))


def is_feasible(solution: list, weights: list, capacity: int) -> bool:
    """Check if a solution respects the capacity constraint."""
    return solution_weight(solution, weights) <= capacity
