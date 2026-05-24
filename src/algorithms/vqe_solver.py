"""
vqe_solver.py
-------------
VQE (Variational Quantum Eigensolver) solver for the 0-1 Knapsack Problem.

Workflow:
    1. Build Ising Hamiltonian H_C from KP instance
    2. Prepare parametric ansatz |psi(gamma)>  (RealAmplitudes)
    3. Minimize <psi(gamma)|H_C|psi(gamma)> with classical optimizer (COBYLA)
    4. Sample the optimised circuit -> decode best bitstring -> check feasibility

Reference:
    Peruzzo et al. (2014). A variational eigenvalue solver on a photonic chip.
    Nature Communications, 5:4213.
"""

import time
import numpy as np
from scipy.optimize import minimize

from qiskit.circuit.library import real_amplitudes as _real_amplitudes_fn
from qiskit.primitives    import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info  import SparsePauliOp
from qiskit import QuantumCircuit

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.hamiltonian import (
    build_hamiltonian, bitstring_to_solution,
    solution_profit, solution_weight, is_feasible
)


class VQESolver:
    """
    VQE solver for 0-1 KP.

    Args:
        profits   : list[float]
        weights   : list[float]
        capacity  : int
        reps      : int   – number of ansatz layers (default 2)
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
        reps:     int   = 2,
        lam:      float = None,
        max_iter: int   = 300,
        shots:    int   = 8192,
        seed:     int   = None
    ):
        self.profits  = profits
        self.weights  = weights
        self.capacity = capacity
        self.n        = len(profits)
        self.reps     = reps
        self.max_iter = max_iter
        self.shots    = shots
        self.seed     = seed

        # Build Hamiltonian
        self.hamiltonian, self.offset = build_hamiltonian(
            profits, weights, capacity, lam
        )

    def _build_ansatz(self) -> QuantumCircuit:
        """
        RealAmplitudes ansatz: Ry rotations + linear entanglement.
        Number of parameters: n * (reps + 1)
        """
        return _real_amplitudes_fn(
            num_qubits   = self.n,
            reps         = self.reps,
            entanglement = 'linear'
        )

    def run(self) -> dict:
        """
        Run VQE.

        Returns:
            {
              'best_value'    : float,
              'best_solution' : list[int],
              'history'       : list[float],  # energy per COBYLA iteration
              'opt_energy'    : float,
              'time_sec'      : float,
              'method'        : 'VQE'
            }
        """
        t0      = time.perf_counter()
        ansatz  = self._build_ansatz()
        n_params = ansatz.num_parameters

        estimator = StatevectorEstimator()
        history   = []

        # ── Objective function ────────────────────────────────────────────
        def objective(params):
            pub    = (ansatz, self.hamiltonian, params)
            result = estimator.run([pub]).result()
            energy = float(result[0].data.evs)
            history.append(energy)
            return energy

        # ── Initial parameters: small random ─────────────────────────────
        rng = np.random.default_rng(self.seed)
        x0  = rng.uniform(-np.pi, np.pi, n_params)

        # ── Classical optimisation (COBYLA) ───────────────────────────────
        result = minimize(
            objective, x0,
            method  = 'COBYLA',
            options = {'maxiter': self.max_iter, 'rhobeg': 0.5}
        )

        opt_params = result.x
        opt_energy = result.fun

        # ── Sample the optimised circuit ──────────────────────────────────
        bound_circuit = ansatz.assign_parameters(opt_params)
        bound_circuit.measure_all()

        sampler = StatevectorSampler()
        job     = sampler.run([bound_circuit], shots=self.shots)
        counts  = job.result()[0].data.meas.get_counts()

        # Find best feasible bitstring by profit
        best_value    = 0.0
        best_solution = [0] * self.n

        for bitstr, count in counts.items():
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
            "method"        : "VQE"
        }
