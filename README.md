# Knapsack Quantum Solvers

Comparative study of four approaches for solving the **0-1 Knapsack Problem**:

| Method | Type | Backend |
|--------|------|---------|
| `QIGA` | Quantum-Inspired Genetic Algorithm | Classical (NumPy) |
| `VQE`  | Variational Quantum Eigensolver | Qiskit AerSimulator |
| `QAOA` | Quantum Approximate Optimization | Qiskit AerSimulator |
| `DP`   | Dynamic Programming (exact baseline) | Classical |

> Based on: Han & Kim, *IEEE Trans. Evol. Comput.*, 2002  
> Circuits: Qiskit 1.x — IBM Quantum compatible

---

## Project Structure

```
knapsack_quantum/
├── data/
│   ├── instances/          # Generated KP instances (.json)
│   └── generate_instances.py
├── src/
│   ├── algorithms/
│   │   ├── qiga.py         # QIGA implementation
│   │   ├── vqe_solver.py   # VQE solver
│   │   ├── qaoa_solver.py  # QAOA solver
│   │   └── dp_solver.py    # Dynamic programming (baseline)
│   └── utils/
│       ├── hamiltonian.py  # Ising Hamiltonian builder
│       ├── metrics.py      # Evaluation metrics
│       └── plot_results.py # Visualization
├── results/                # JSON results per run
├── figures/                # Generated plots
├── notebooks/
│   └── comparison.ipynb    # Full comparison notebook
├── tests/
│   └── test_all.py
├── run_experiments.py      # Main experiment script
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/knapsack-quantum.git
cd knapsack-quantum
pip install -r requirements.txt
```

## Quick Start

```bash
# Generate instances
python data/generate_instances.py

# Run all solvers on all instances
python run_experiments.py

# Plot comparison
python src/utils/plot_results.py
```

## Requirements

- Python 3.10+
- qiskit >= 1.0
- qiskit-aer >= 0.14
- numpy, scipy, matplotlib, pandas, tqdm

## Results Summary

Results are saved in `results/summary.csv` after running experiments.

## Citation

```bibtex
@article{han2002quantum,
  title   = {Quantum-inspired evolutionary algorithm for a class of combinatorial optimization},
  author  = {Han, Kuk-Hyun and Kim, Jong-Hwan},
  journal = {IEEE Transactions on Evolutionary Computation},
  volume  = {6},
  number  = {6},
  pages   = {580--593},
  year    = {2002}
}
```
