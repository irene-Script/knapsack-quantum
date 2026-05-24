"""
test_all.py
-----------
Unit tests for all KP solvers.
Run with: python -m pytest tests/test_all.py -v

Tests cover:
  - DP exact solution
  - QIGA feasibility + profit improvement
  - Hamiltonian structure (Ising coefficients)
  - VQE/QAOA circuit construction (no execution, just structure)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

from algorithms.dp_solver   import DPSolver
from algorithms.qiga        import QIGA
from utils.hamiltonian      import (
    build_hamiltonian, bitstring_to_solution,
    solution_profit, solution_weight, is_feasible
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_instance():
    """Tiny KP instance with known optimal solution."""
    # items: (profit, weight)  = (6,2), (10,4), (12,6)
    # capacity = 10
    # optimal: items 0+1 -> profit 16, weight 6 <= 10
    #      or  items 0+2 -> profit 18, weight 8 <= 10  ← optimal
    #      or  items 1+2 -> profit 22, weight 10 <= 10 ← actually better!
    return {
        'profits' : [6, 10, 12],
        'weights' : [2,  4,  6],
        'capacity': 10,
        'optimal' : 22   # items 1 and 2 (0-indexed)
    }


@pytest.fixture
def medium_instance():
    """Randomly generated medium instance (n=15)."""
    rng = np.random.default_rng(42)
    n   = 15
    w   = rng.integers(1, 50, n).tolist()
    p   = rng.integers(1, 50, n).tolist()
    W   = int(0.5 * sum(w))
    dp  = DPSolver(p, w, W).solve()
    return {'profits': p, 'weights': w, 'capacity': W, 'optimal': dp['best_value']}


# ── DP tests ──────────────────────────────────────────────────────────────────

class TestDPSolver:

    def test_small_optimal(self, small_instance):
        inst = small_instance
        res  = DPSolver(inst['profits'], inst['weights'], inst['capacity']).solve()
        assert res['best_value'] == inst['optimal']

    def test_feasibility(self, small_instance):
        inst = small_instance
        res  = DPSolver(inst['profits'], inst['weights'], inst['capacity']).solve()
        total_w = sum(w * x for w, x in zip(inst['weights'], res['best_solution']))
        assert total_w <= inst['capacity']

    def test_solution_length(self, small_instance):
        inst = small_instance
        res  = DPSolver(inst['profits'], inst['weights'], inst['capacity']).solve()
        assert len(res['best_solution']) == len(inst['profits'])

    def test_binary_solution(self, small_instance):
        inst = small_instance
        res  = DPSolver(inst['profits'], inst['weights'], inst['capacity']).solve()
        assert all(x in (0, 1) for x in res['best_solution'])

    def test_method_label(self, small_instance):
        inst = small_instance
        res  = DPSolver(inst['profits'], inst['weights'], inst['capacity']).solve()
        assert res['method'] == 'DP'


# ── Hamiltonian tests ─────────────────────────────────────────────────────────

class TestHamiltonian:

    def test_construction(self, small_instance):
        inst = small_instance
        ham, offset = build_hamiltonian(
            inst['profits'], inst['weights'], inst['capacity']
        )
        assert ham.num_qubits == len(inst['profits'])

    def test_pauli_terms_nonzero(self, small_instance):
        inst = small_instance
        ham, _ = build_hamiltonian(
            inst['profits'], inst['weights'], inst['capacity']
        )
        assert len(ham) > 0

    def test_bitstring_conversion(self):
        # '1010' -> qubit 0 is rightmost -> [0, 1, 0, 1]
        assert bitstring_to_solution('1010') == [0, 1, 0, 1]
        assert bitstring_to_solution('0000') == [0, 0, 0, 0]
        assert bitstring_to_solution('1111') == [1, 1, 1, 1]

    def test_profit_calculation(self, small_instance):
        inst = small_instance
        sol  = [0, 1, 1]   # items 1 and 2
        val  = solution_profit(sol, inst['profits'])
        assert val == pytest.approx(22.0)

    def test_feasibility_check(self, small_instance):
        inst = small_instance
        assert is_feasible([0, 1, 1], inst['weights'], inst['capacity'])   # 10 <= 10
        assert not is_feasible([1, 1, 1], inst['weights'], inst['capacity'])  # 12 > 10


# ── QIGA tests ────────────────────────────────────────────────────────────────

class TestQIGA:

    def test_runs_without_error(self, small_instance):
        inst = small_instance
        qiga = QIGA(
            inst['profits'], inst['weights'], inst['capacity'],
            pop_size=10, max_gen=50, seed=0
        )
        res = qiga.run()
        assert 'best_value' in res
        assert 'best_solution' in res
        assert 'history' in res

    def test_feasibility(self, small_instance):
        inst = small_instance
        qiga = QIGA(
            inst['profits'], inst['weights'], inst['capacity'],
            pop_size=20, max_gen=100, seed=1
        )
        res = qiga.run()
        total_w = sum(w * x for w, x in zip(inst['weights'], res['best_solution']))
        assert total_w <= inst['capacity'], \
            f"QIGA returned infeasible solution (weight={total_w} > {inst['capacity']})"

    def test_solution_nonnegative(self, small_instance):
        inst = small_instance
        qiga = QIGA(
            inst['profits'], inst['weights'], inst['capacity'],
            pop_size=10, max_gen=50, seed=2
        )
        res = qiga.run()
        assert res['best_value'] >= 0

    def test_history_length(self, small_instance):
        inst = small_instance
        max_gen = 50
        qiga = QIGA(
            inst['profits'], inst['weights'], inst['capacity'],
            pop_size=10, max_gen=max_gen, seed=3
        )
        res = qiga.run()
        assert len(res['history']) == max_gen

    def test_history_nondecreasing(self, small_instance):
        inst = small_instance
        qiga = QIGA(
            inst['profits'], inst['weights'], inst['capacity'],
            pop_size=20, max_gen=100, seed=4
        )
        res = qiga.run()
        hist = res['history']
        # Best fitness should never decrease
        for i in range(1, len(hist)):
            assert hist[i] >= hist[i-1] - 1e-9, \
                f"Fitness decreased at gen {i}: {hist[i-1]} -> {hist[i]}"

    def test_reaches_good_solution_medium(self, medium_instance):
        """QIGA should reach at least 90% of optimal on medium instances."""
        inst = medium_instance
        qiga = QIGA(
            inst['profits'], inst['weights'], inst['capacity'],
            pop_size=30, max_gen=300, p_cross=0.75, p_mut=0.03, seed=42
        )
        res = qiga.run()
        ar  = res['best_value'] / inst['optimal']
        assert ar >= 0.90, f"AR={ar:.4f} < 0.90 on medium instance"

    def test_method_label(self, small_instance):
        inst = small_instance
        res  = QIGA(inst['profits'], inst['weights'], inst['capacity'],
                    pop_size=5, max_gen=10, seed=0).run()
        assert res['method'] == 'QIGA'


# ── VQE/QAOA circuit tests (lightweight – no full optimisation) ───────────────

class TestCircuitStructure:
    """
    Tests that check circuit construction without running the full optimiser.
    """

    def test_hamiltonian_num_qubits(self):
        profits  = [3, 5, 7]
        weights  = [2, 3, 4]
        capacity = 6
        ham, _ = build_hamiltonian(profits, weights, capacity)
        assert ham.num_qubits == 3

    def test_qaoa_circuit_parameters(self):
        """QAOA circuit with p layers should have 2*p parameters."""
        from algorithms.qaoa_solver import _build_qaoa_circuit
        profits  = [3, 5, 7]
        weights  = [2, 3, 4]
        capacity = 6
        p        = 2
        ham, _   = build_hamiltonian(profits, weights, capacity)
        qc       = _build_qaoa_circuit(3, ham, p)
        assert qc.num_parameters == 2 * p

    def test_vqe_ansatz_parameters(self):
        """RealAmplitudes with n qubits and reps layers."""
        from qiskit.circuit.library import real_amplitudes
        n    = 4
        reps = 2
        ans  = real_amplitudes(num_qubits=n, reps=reps)
        # n*(reps+1) parameters
        assert ans.num_parameters == n * (reps + 1)
