"""
qiga.py
-------
Quantum-Inspired Genetic Algorithm (QIGA) for the 0-1 Knapsack Problem.

References:
    Han, K.-H. & Kim, J.-H. (2002).
    Quantum-inspired evolutionary algorithm for a class of combinatorial optimization.
    IEEE Transactions on Evolutionary Computation, 6(6), 580-593.

    Galavani, Younesi & Ansari (2025).
    QIGA: Quantum-Inspired Genetic Algorithm for Dynamic Scheduling in MEC.
    IEEE CSICC 2025.

Operators (aligned with 2025 paper):
    - Measurement       : P(x_i=1) = sin²(θ_i)           [Han & Kim]
    - Crossover         : rotation gate R(±π/4) on angles  [Algorithm 3, 2025]
    - Mutation          : NOT gate  θ' = π/2 − θ           [Algorithm 4, 2025]
    - Quantum Interference: Q'_i = cos(α_i)Q_i + sin(α_i)Q_best  [Algorithm 5, 2025]
    - Q-gate            : rotation toward best solution     [Han & Kim Table I]
    - Elitism           : best individual preserved each gen
"""

import time
import numpy as np


class QIGA:
    """
    Quantum-Inspired Genetic Algorithm for 0-1 KP.

    Args:
        profits   : list[float] – p_i
        weights   : list[float] – w_i
        capacity  : int         – W
        pop_size  : int         – number of Q-chromosomes (default 50)
        max_gen   : int         – maximum generations (default 1000)
        p_cross   : float       – crossover probability (default 0.75)
        p_mut     : float       – mutation probability per gene (default 0.03)
        delta     : float       – Q-gate rotation step (default 0.05)
        n_elite   : int         – number of elite individuals kept (default 1)
        seed      : int|None    – random seed
    """

    def __init__(
        self,
        profits:  list,
        weights:  list,
        capacity: int,
        pop_size: int   = 50,
        max_gen:  int   = 1000,
        p_cross:  float = 0.75,
        p_mut:    float = 0.03,
        delta:    float = 0.05,
        n_elite:  int   = 1,
        seed:     int   = None
    ):
        self.p       = np.array(profits,  dtype=float)
        self.w       = np.array(weights,  dtype=float)
        self.W       = float(capacity)
        self.n       = len(profits)
        self.m       = pop_size
        self.g_max   = max_gen
        self.p_cross = p_cross
        self.p_mut   = p_mut
        self.delta   = delta
        self.n_elite = n_elite
        self.rng     = np.random.default_rng(seed)

    # ── Quantum operators ─────────────────────────────────────────────────────

    def _measure(self, theta: np.ndarray) -> np.ndarray:
        """Collapse Q-chromosome to binary: P(x_i=1) = sin²(θ_i)."""
        u = self.rng.uniform(0.0, 1.0, self.n)
        return (u < np.sin(theta) ** 2).astype(int)

    def _collapse(self) -> np.ndarray:
        """Project to |0...0>: called when constraint is violated."""
        return np.zeros(self.n)

    def _crossover(self, ta: np.ndarray, tb: np.ndarray):
        """
        Quantum crossover via rotation gate R(θ_c) (Algorithm 3, 2025 paper).

        One parent is rotated by +π/4, the other by -π/4 in angle space.
        This is equivalent to applying R(θ_c) and R(-θ_c) to the qubit vectors:
            q = [cos θ, sin θ]^T  =>  R(θ_c) q = [cos(θ+θ_c), sin(θ+θ_c)]^T
        """
        theta_c = np.pi / 4.0
        da = np.clip(ta + theta_c, 0.0, np.pi / 2.0)
        db = np.clip(tb - theta_c, 0.0, np.pi / 2.0)
        return da, db

    def _mutate(self, theta: np.ndarray) -> np.ndarray:
        """
        Quantum mutation via NOT gate (Algorithm 4, 2025 paper).

        NOT gate on [cos θ, sin θ]^T = [sin θ, cos θ]^T = [cos(π/2−θ), sin(π/2−θ)]^T
        This flips the measurement probability: P(1) = sin²θ  →  cos²θ = sin²(π/2−θ).
        Applied with probability p_mut per gene.
        """
        mask = self.rng.uniform(0.0, 1.0, self.n) < self.p_mut
        theta_new = theta.copy()
        theta_new[mask] = np.pi / 2.0 - theta_new[mask]
        return theta_new

    def _quantum_interference(
        self,
        pop: np.ndarray,
        fitness: list,
        b_theta: np.ndarray,
        f_star: float
    ) -> np.ndarray:
        """
        Quantum interference step (Algorithm 5, 2025 paper).

        Q'_i = cos(α_i) Q_i + sin(α_i) Q_best
        where α_i = (π/2) * (f_best - f_i) / f_best

        In angle space, mixing two qubits [cos θ_a, sin θ_a] and [cos θ_b, sin θ_b]
        with weights (cos α, sin α) gives a new qubit whose angle is atan2 of the
        combined beta/alpha components.

        When f_i ≈ f_best: α_i ≈ 0 → Q'_i ≈ Q_i   (no change)
        When f_i ≪ f_best: α_i ≈ π/2 → Q'_i ≈ Q_best (pulled toward best)
        """
        pop_new = pop.copy()
        if f_star <= 0:
            return pop_new

        for j in range(self.m):
            fi = fitness[j]
            alpha_i = (np.pi / 2.0) * max(0.0, (f_star - fi) / f_star)
            cos_a = np.cos(alpha_i)
            sin_a = np.sin(alpha_i)
            new_alpha = cos_a * np.cos(pop[j]) + sin_a * np.cos(b_theta)
            new_beta  = cos_a * np.sin(pop[j]) + sin_a * np.sin(b_theta)
            pop_new[j] = np.clip(np.arctan2(new_beta, new_alpha), 0.0, np.pi / 2.0)

        return pop_new

    def _qgate(self, theta: np.ndarray, x: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Q-gate rotation guided by current vs best binary solution (Han & Kim 2002).

            x_i=0, b_i=1 => rotate +delta  (push toward 1)
            x_i=1, b_i=0 => rotate -delta  (push toward 0)
        """
        theta_new = theta.copy()
        for i in range(self.n):
            if x[i] == 0 and b[i] == 1:
                theta_new[i] += self.delta
            elif x[i] == 1 and b[i] == 0:
                theta_new[i] -= self.delta
        return np.clip(theta_new, 0.0, np.pi / 2.0)

    # ── Fitness ───────────────────────────────────────────────────────────────

    def _fitness(self, x: np.ndarray) -> float:
        """Return profit if feasible, else 0."""
        if np.dot(self.w, x) > self.W:
            return 0.0
        return float(np.dot(self.p, x))

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Run QIGA.

        Returns:
            {
              'best_value'    : float,
              'best_solution' : list[int],
              'history'       : list[float],   # best fitness per generation
              'time_sec'      : float,
              'method'        : 'QIGA'
            }
        """
        t0 = time.perf_counter()

        # Init: θ = π/4 → P(0) = P(1) = 0.5 (maximum diversity)
        pop     = np.full((self.m, self.n), np.pi / 4.0)
        b_star  = np.zeros(self.n, dtype=int)   # best binary solution
        b_theta = np.full(self.n, np.pi / 4.0)  # angles of best individual
        f_star  = 0.0
        history = []

        for gen in range(self.g_max):
            solutions = []
            fitness   = []

            # ── Step 1: measure all Q-chromosomes ─────────────────────────
            for j in range(self.m):
                x = self._measure(pop[j])
                f = self._fitness(x)

                if f == 0.0 and np.dot(self.w, x) > self.W:
                    pop[j] = self._collapse()
                    solutions.append(np.zeros(self.n, dtype=int))
                    fitness.append(0.0)
                else:
                    solutions.append(x)
                    fitness.append(f)
                    if f > f_star:
                        f_star  = f
                        b_star  = x.copy()
                        b_theta = pop[j].copy()

            history.append(f_star)

            # ── Step 2: rank by fitness (elitism: keep best n_elite) ───────
            order     = np.argsort(fitness)[::-1]
            pop       = pop[order]
            solutions = [solutions[i] for i in order]
            fitness   = [fitness[i]   for i in order]

            # Save elite individuals before operators modify them
            elite_pop = pop[:self.n_elite].copy()

            # ── Step 3: crossover (rotation gate ±π/4) ────────────────────
            for j in range(0, self.m - 1, 2):
                if self.rng.uniform() < self.p_cross:
                    pop[j], pop[j + 1] = self._crossover(pop[j], pop[j + 1])

            # ── Step 4: mutation (NOT gate) ────────────────────────────────
            for j in range(self.n_elite, self.m):   # protect elite
                pop[j] = self._mutate(pop[j])

            # ── Step 5: Q-gate update ──────────────────────────────────────
            for j in range(self.m):
                pop[j] = self._qgate(pop[j], solutions[j], b_star)

            # ── Step 6: quantum interference (Algorithm 5) ────────────────
            pop = self._quantum_interference(pop, fitness, b_theta, f_star)

            # ── Step 7: restore elite individuals ─────────────────────────
            pop[:self.n_elite] = elite_pop

        elapsed = time.perf_counter() - t0

        return {
            "best_value"    : f_star,
            "best_solution" : b_star.tolist(),
            "history"       : history,
            "time_sec"      : elapsed,
            "method"        : "QIGA"
        }
