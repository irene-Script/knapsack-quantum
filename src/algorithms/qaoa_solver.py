"""
qaoa_solver.py
--------------
QAOA (Quantum Approximate Optimization Algorithm) solver for the 0-1 Knapsack Problem.

Circuit structure (p layers):
    |psi(gamma, beta)> = prod_{k=1}^{p}  exp(-i*beta_k * H_M) * exp(-i*gamma_k * H_C)  |+>^n

where:
    H_C = Ising cost Hamiltonian (from KP penalty formulation)
    H_M = sum_i X_i               (mixer / driver Hamiltonian)
    |+>^n = H^{otimes n} |0>^n    (uniform superposition)

Reference:
    Farhi, E., Goldstone, J., & Gutmann, S. (2014).
    A quantum approximate optimization algorithm.
    arXiv:1411.4028
"""

import time
import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.hamiltonian import (
    build_hamiltonian, bitstring_to_solution,
    solution_profit, is_feasible
)


def _build_qaoa_circuit(n: int, hamiltonian: SparsePauliOp, p: int) -> QuantumCircuit:
    """
    Build the QAOA circuit manually for full transparency.

    Args:
        n           : number of qubits
        hamiltonian : SparsePauliOp cost Hamiltonian
        p           : number of QAOA layers

    Returns:
        Parametric QuantumCircuit with 2*p parameters (gamma_0..p-1, beta_0..p-1)
    """
    gamma = ParameterVector('gamma', p)
    beta  = ParameterVector('beta',  p)

    qc = QuantumCircuit(n)

    # ── Initial state: uniform superposition |+>^n ────────────────────────
    qc.h(range(n))

    for k in range(p):
        # ── Cost unitary: exp(-i * gamma_k * H_C) ────────────────────────
        # Decompose H_C into Pauli terms and apply rotations
        for pauli_str, coeff in zip(
            hamiltonian.paulis.to_labels(),
            hamiltonian.coeffs
        ):
            coeff_real = float(np.real(coeff))
            if abs(coeff_real) < 1e-10:
                continue

            # Identify which qubits have Z (or ZZ) operators
            z_qubits = [
                n - 1 - idx
                for idx, ch in enumerate(pauli_str)
                if ch == 'Z'
            ]

            if len(z_qubits) == 1:
                # Single Z: Rz rotation
                qc.rz(2.0 * coeff_real * gamma[k], z_qubits[0])

            elif len(z_qubits) == 2:
                # ZZ interaction: CNOT + Rz + CNOT
                qi, qj = z_qubits
                qc.cx(qi, qj)
                qc.rz(2.0 * coeff_real * gamma[k], qj)
                qc.cx(qi, qj)
            # Identity term: global phase, skip

        # ── Mixer unitary: exp(-i * beta_k * H_M) ────────────────────────
        # H_M = sum_i X_i  =>  exp(-i*beta*X) = Rx(2*beta) on each qubit
        for i in range(n):
            qc.rx(2.0 * beta[k], i)

    return qc


class QAOASolver:
    """
    QAOA solver for the 0-1 KP.

    Args:
        profits   : list[float]
        weights   : list[float]
        capacity  : int
        p         : int   – circuit depth / number of layers (default 2)
        lam       : float – penalty coefficient (auto if None)
        max_iter  : int   – max COBYLA iterations (default 300)
        shots     : int   – shots for final sampling (default 8192)
        seed      : int   – random seed
    """

    def __init__(
        self,
        profits:  list,
        weights:  list,
        capacity: int,
        p:        int   = 2,
        lam:      float = None,
        max_iter: int   = 300,
        shots:    int   = 8192,
        seed:     int   = None
    ):
        self.profits  = profits
        self.weights  = weights
        self.capacity = capacity
        self.n        = len(profits)
        self.p        = p
        self.max_iter = max_iter
        self.shots    = shots
        self.seed     = seed

        self.hamiltonian, self.offset = build_hamiltonian(
            profits, weights, capacity, lam
        )

    def run(self) -> dict:
        """
        Run QAOA.

        Returns:
            {
              'best_value'    : float,
              'best_solution' : list[int],
              'history'       : list[float],
              'opt_energy'    : float,
              'time_sec'      : float,
              'method'        : f'QAOA_p{self.p}'
            }
        """
        t0  = time.perf_counter()
        qc  = _build_qaoa_circuit(self.n, self.hamiltonian, self.p)
        n_params = qc.num_parameters   # 2 * p

        estimator = StatevectorEstimator()
        history   = []

        # ── Objective ─────────────────────────────────────────────────────
        def objective(params):
            pub    = (qc, self.hamiltonian, params)
            result = estimator.run([pub]).result()
            energy = float(result[0].data.evs)
            history.append(energy)
            return energy

        # ── Initialisation: gamma ~ U[0, pi], beta ~ U[0, pi/2] ──────────
        rng   = np.random.default_rng(self.seed)
        x0    = np.concatenate([
            rng.uniform(0.0, np.pi,     self.p),   # gamma
            rng.uniform(0.0, np.pi/2.0, self.p)    # beta
        ])

        # ── Classical optimisation ────────────────────────────────────────
        result = minimize(
            objective, x0,
            method  = 'COBYLA',
            options = {'maxiter': self.max_iter, 'rhobeg': 0.5}
        )

        opt_params = result.x
        opt_energy = result.fun

        # ── Sample optimised circuit ──────────────────────────────────────
        bound_circuit = qc.assign_parameters(opt_params)
        bound_circuit.measure_all()

        sampler = StatevectorSampler()
        job     = sampler.run([bound_circuit], shots=self.shots)
        counts  = job.result()[0].data.meas.get_counts()

        best_value    = 0.0
        best_solution = [0] * self.n

        # Scan ALL measured bitstrings; keep the one with highest feasible profit
        # (not just the most frequent one – frequency and profit are uncorrelated)
        for bitstr in counts:
            sol = bitstring_to_solution(bitstr)
            if len(sol) != self.n:
                continue
            if is_feasible(sol, self.weights, self.capacity):
                val = solution_profit(sol, self.profits)
                if val > best_value:
                    best_value    = val
                    best_solution = sol

        elapsed = time.perf_counter() - t0

        return {
            "best_value"    : best_value,
            "best_solution" : best_solution,
            "history"       : history,
            "opt_energy"    : opt_energy,
            "time_sec"      : elapsed,
            "method"        : f"QAOA_p{self.p}"
        }
