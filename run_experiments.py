"""
run_experiments.py
------------------
Main experiment script: runs all solvers on all instances and saves results.

Usage:
    python run_experiments.py                          # full benchmark
    python run_experiments.py --light                  # PC limité (~5 min total)
    python run_experiments.py --methods DP QIGA        # solvers sélectionnés
    python run_experiments.py --sizes 10 15 --runs 3   # surcharger les defaults

Flags clés:
    --light        : n ∈ {5,8,10}, 5 runs, 200 générations QIGA, 150 iter VQE/QAOA
                     (générer d'abord: python data/generate_instances.py --light)
    --runs N       : nombre de runs indépendants par paire (method, instance)
    --qiga-gen N   : générations QIGA (surcharge le default de mode)
    --qaoa-p N     : profondeur du circuit QAOA (default 2)
    --vqe-reps N   : couches d'ansatz RealAmplitudes (default 2)

Output:
    results/summary.csv                    – tableau de métriques agrégées
    results/<method>_<inst>_run<k>.json    – résultat d'un run individuel
"""

import os
import sys
import json
import argparse
import csv
import time
import numpy as np

try:
    from tqdm import tqdm
except ImportError:                         # graceful fallback — no progress bar
    def tqdm(iterable, **kwargs):           # type: ignore[override]
        return iterable

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from algorithms.dp_solver   import DPSolver
from algorithms.qiga        import QIGA
from algorithms.vqe_solver  import VQESolver
from algorithms.qaoa_solver import QAOASolver
from utils.metrics          import aggregate_runs

INSTANCES_DIR = os.path.join(ROOT, 'data', 'instances')
RESULTS_DIR   = os.path.join(ROOT, 'results')

# Sizes to run quantum solvers on (VQE/QAOA are slow: n=20 takes ~2h on CPU simulator)
QUANTUM_MAX_N_FULL  = 15   # used in normal mode
QUANTUM_MAX_N_LIGHT = 10   # used in --light mode (all 3 sizes qualify)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run KP solver comparison")
    parser.add_argument(
        '--light', action='store_true',
        help='Lightweight mode: fewer runs, fewer generations, lower QUANTUM_MAX_N. '
             'Ideal for PCs with limited resources. Use with: '
             'python data/generate_instances.py --light  first.'
    )
    parser.add_argument(
        '--methods', nargs='+',
        default=['DP', 'QIGA', 'VQE', 'QAOA'],
        choices=['DP', 'QIGA', 'VQE', 'QAOA'],
        help='Solvers to run'
    )
    parser.add_argument(
        '--sizes', nargs='+', type=int,
        default=None,
        help='Instance sizes to run (default: all available)'
    )
    parser.add_argument(
        '--runs', type=int, default=None,
        help='Independent runs per (method, instance) pair '
             '(default: 5 in --light mode, 10 otherwise)'
    )
    parser.add_argument(
        '--qiga-pop', type=int, default=50,  help='QIGA population size'
    )
    parser.add_argument(
        '--qiga-gen', type=int, default=None,
        help='QIGA max generations (default: 200 in --light, 500 otherwise)'
    )
    parser.add_argument(
        '--qaoa-p', type=int, default=2,     help='QAOA circuit depth p'
    )
    parser.add_argument(
        '--vqe-reps', type=int, default=2,   help='VQE ansatz reps'
    )
    return parser.parse_args()


# ── Instance loader ───────────────────────────────────────────────────────────

def load_instance(fpath: str) -> dict:
    with open(fpath) as f:
        return json.load(f)


def list_instances(sizes=None):
    files = sorted(f for f in os.listdir(INSTANCES_DIR) if f.endswith('.json'))
    instances = []
    for fname in files:
        fpath = os.path.join(INSTANCES_DIR, fname)
        inst  = load_instance(fpath)
        n     = inst['n']
        if sizes is not None and n not in sizes:
            continue
        instances.append((fname, fpath, inst))
    return instances


# ── Single solver run ─────────────────────────────────────────────────────────

def run_solver(method: str, inst: dict, args, seed: int = None) -> dict:
    """Run one solver on one instance. Returns result dict."""
    p = inst['profits']
    w = inst['weights']
    W = inst['capacity']

    if method == 'DP':
        solver = DPSolver(p, w, W)
        return solver.solve()

    elif method == 'QIGA':
        solver = QIGA(
            p, w, W,
            pop_size = args.qiga_pop,
            max_gen  = args.qiga_gen,
            p_cross  = 0.75,
            p_mut    = 0.03,
            seed     = seed
        )
        return solver.run()

    elif method == 'VQE':
        solver = VQESolver(
            p, w, W,
            reps     = args.vqe_reps,
            max_iter = getattr(args, '_vqe_max_iter', 300),
            seed     = seed
        )
        return solver.run()

    elif method == 'QAOA':
        solver = QAOASolver(
            p, w, W,
            p        = args.qaoa_p,
            max_iter = getattr(args, '_qaoa_max_iter', 300),
            seed     = seed
        )
        return solver.run()

    else:
        raise ValueError(f"Unknown method: {method}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _clear_results():
    """Delete all files in RESULTS_DIR before a fresh run."""
    if not os.path.isdir(RESULTS_DIR):
        return
    deleted = 0
    for fname in os.listdir(RESULTS_DIR):
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)
            deleted += 1
    if deleted:
        print(f"  [reset] {deleted} fichier(s) supprime(s) de results/")


def main():
    args = parse_args()

    # ── Resolve light-mode defaults ───────────────────────────────────────────
    if args.light:
        quantum_max_n    = QUANTUM_MAX_N_LIGHT
        n_runs_default   = 5
        qiga_gen_default = 200
        vqe_max_iter     = 150
        mode_label       = "LIGHT (PC limité)"
    else:
        quantum_max_n    = QUANTUM_MAX_N_FULL
        n_runs_default   = 10
        qiga_gen_default = 500
        vqe_max_iter     = 300
        mode_label       = "FULL"

    # Allow CLI overrides on top of mode defaults
    n_runs        = args.runs     if args.runs     is not None else n_runs_default
    args.qiga_gen = args.qiga_gen if args.qiga_gen is not None else qiga_gen_default

    # In --light mode restrict sizes automatically (unless user forced them via --sizes)
    _LIGHT_SIZES = [5, 8, 10]          # must match SIZES_LIGHT in generate_instances.py
    if args.light and args.sizes is None:
        args.sizes = _LIGHT_SIZES

    # Patch VQE/QAOA max_iter via a tiny shim (solvers read from args in run_solver)
    args._vqe_max_iter  = vqe_max_iter
    args._qaoa_max_iter = vqe_max_iter

    os.makedirs(RESULTS_DIR, exist_ok=True)
    _clear_results()

    instances = list_instances(sizes=args.sizes)
    if not instances:
        hint = ("python data/generate_instances.py --light"
                if args.light else "python data/generate_instances.py")
        print(f"No instances found. Run: {hint}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Mode     : {mode_label}")
    print(f"  Methods  : {args.methods}")
    print(f"  Instances: {len(instances)}")
    print(f"  Runs/pair: {n_runs}  |  QIGA gen: {args.qiga_gen}  |  VQE/QAOA iter: {vqe_max_iter}")
    print(f"  Quantum max n: {quantum_max_n}")
    print(f"{'='*60}\n")

    summary_rows = []

    for fname, fpath, inst in instances:
        n       = inst['n']
        optimal = inst.get('optimal')
        inst_id = fname.replace('.json', '')

        print(f"\n-- Instance: {fname}  (n={n}, W={inst['capacity']}, opt={optimal}) --")

        for method in args.methods:

            # Skip slow quantum methods on large instances
            if method in ('VQE', 'QAOA') and n > quantum_max_n:
                print(f"  {method:12s} : skipped (n={n} > {quantum_max_n})")
                continue

            # DP is deterministic: run once
            method_runs = 1 if method == 'DP' else n_runs
            runs_results = []

            pbar = tqdm(
                range(method_runs),
                desc    = f"  {method:12s}",
                leave   = False,
                ncols   = 60
            )
            for run_idx in pbar:
                seed   = run_idx * 1000 + hash(fname) % 1000
                result = run_solver(method, inst, args, seed=seed)
                result['instance'] = inst_id
                result['n']        = n
                result['optimal']  = optimal
                runs_results.append(result)

                # Save individual run
                rname = f"{method}_{inst_id}_run{run_idx:02d}.json"
                with open(os.path.join(RESULTS_DIR, rname), 'w') as f:
                    history = result.get('history', [])
                    saved = {k: v for k, v in result.items() if k != 'history'}
                    saved['history_len'] = len(history)
                    # Save history for small instances (convergence plots)
                    if n <= 30 and history:
                        saved['history'] = history
                    json.dump(saved, f, indent=2)

            # Aggregate over runs
            agg = aggregate_runs(runs_results, optimal) if optimal is not None else {}
            best = max(r['best_value'] for r in runs_results)

            print(
                f"  {method:12s}  best={best:8.2f}  "
                f"mean={agg.get('mean', best):8.2f}  "
                f"std={agg.get('std', 0.0):6.2f}  "
                f"AR={agg.get('ar_mean', 1.0):.4f}  "
                f"opt%={agg.get('pct_optimal', 100.0):.1f}  "
                f"t={agg.get('time_mean', runs_results[0]['time_sec']):.3f}s"
            )

            row = {
                'instance'   : inst_id,
                'n'          : n,
                'capacity'   : inst['capacity'],
                'optimal'    : optimal,
                'method'     : method,
                'best'       : agg.get('best',        best),
                'mean'       : agg.get('mean',        best),
                'std'        : agg.get('std',         0.0),
                'ar_mean'    : agg.get('ar_mean',     1.0),
                'gap_mean'   : agg.get('gap_mean',    0.0),
                'time_mean'  : agg.get('time_mean',   runs_results[0]['time_sec']),
                'pct_optimal': agg.get('pct_optimal', 100.0),
                'n_runs'     : method_runs
            }
            summary_rows.append(row)

    # ── Write summary CSV ─────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, 'summary.csv')
    fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n{'='*60}")
    print(f"  Summary saved to: {csv_path}")
    print(f"  Total rows: {len(summary_rows)}")
    print(f"{'='*60}")

    # ── Auto-generate HTML report ─────────────────────────────────────────────
    _generate_report()


def _generate_report():
    """Call analyze_results.py to produce the HTML report automatically."""
    import importlib.util, traceback

    report_script = os.path.join(ROOT, 'analyze_results.py')
    if not os.path.exists(report_script):
        print("\n[rapport] analyze_results.py introuvable, rapport ignore.")
        return

    print(f"\n{'='*60}")
    print("  Generation du rapport HTML ...")
    print(f"{'='*60}")

    try:
        import sys
        saved_argv = sys.argv[:]
        sys.argv = [report_script]          # isolate from run_experiments args
        spec   = importlib.util.spec_from_file_location("analyze_results", report_script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
    except Exception:
        print("[rapport] Erreur lors de la generation du rapport :")
        traceback.print_exc()
    finally:
        sys.argv = saved_argv


if __name__ == "__main__":
    main()

