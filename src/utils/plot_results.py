"""
plot_results.py
---------------
Visualisation of experimental results.
Run after run_experiments.py:
    python src/utils/plot_results.py
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')

METHOD_COLORS = {
    'DP'      : '#2ecc71',
    'QIGA'    : '#3498db',
    'VQE'     : '#e67e22',
    'QAOA_p1' : '#9b59b6',
    'QAOA_p2' : '#e74c3c',
    'QAOA_p3' : '#c0392b',
}

METHOD_MARKERS = {
    'DP'      : 's',
    'QIGA'    : 'o',
    'VQE'     : '^',
    'QAOA_p1' : 'D',
    'QAOA_p2' : 'v',
    'QAOA_p3' : 'P',
}

# Shared style applied to every figure
_STYLE = {
    'font.size'        : 11,
    'axes.titlesize'   : 13,
    'axes.labelsize'   : 11,
    'legend.fontsize'  : 9,
    'xtick.labelsize'  : 9,
    'ytick.labelsize'  : 9,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
}


def _color(method: str) -> str:
    return METHOD_COLORS.get(method, '#7f8c8d')


def _marker(method: str) -> str:
    return METHOD_MARKERS.get(method, 'o')


def load_summary(csv_path: str = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = os.path.join(RESULTS_DIR, 'summary.csv')
    return pd.read_csv(csv_path)


def _save(fig: plt.Figure, name: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, f'{name}.pdf'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, f'{name}.png'), dpi=150, bbox_inches='tight')


# ── Figure 1: Approximation ratio vs instance size ────────────────────────────

def plot_approximation_ratio(df: pd.DataFrame, save: bool = True):
    """Grouped bar chart of mean AR ± std, one group per instance size."""
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(11, 5))

        methods = [m for m in df['method'].unique() if m != 'DP']
        sizes   = sorted(df['n'].unique())
        x       = np.arange(len(sizes))
        width   = 0.8 / max(len(methods), 1)

        for k, method in enumerate(methods):
            grp  = df[df['method'] == method].groupby('n')['ar_mean']
            means = np.array([grp.mean().get(n, np.nan) for n in sizes])
            stds  = np.array([grp.std().get(n, 0.0)     for n in sizes])
            stds  = np.nan_to_num(stds)

            offset = x + k * width - 0.4 + width / 2
            bars = ax.bar(
                offset, means,
                width   = width * 0.88,
                label   = method,
                color   = _color(method),
                alpha   = 0.85,
                yerr    = stds,
                capsize = 3,
                error_kw={'elinewidth': 1.2, 'ecolor': '#555'},
            )

            # Value labels on bars (skip NaN)
            for rect, val in zip(bars, means):
                if not np.isnan(val):
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + max(stds) * 0.05 + 0.001,
                        f'{val:.3f}',
                        ha='center', va='bottom', fontsize=7, rotation=90,
                    )

        ax.axhline(1.0, color='black', linestyle='--', linewidth=1.2, label='Optimal (DP)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'n={n}' for n in sizes])

        # Dynamic y-limits: leave room for value labels
        valid = df[df['method'] != 'DP']['ar_mean'].dropna()
        lo = max(0.0, valid.min() - 0.06) if not valid.empty else 0.0
        ax.set_ylim(lo, 1.08)

        ax.set_ylabel('Approximation Ratio (higher = better)')
        ax.set_title('Mean Approximation Ratio by Instance Size and Method')
        ax.legend(loc='lower left')
        ax.grid(axis='y', alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

        plt.tight_layout()
        if save:
            _save(fig, 'fig1_approx_ratio')
        return fig


# ── Figure 2: Computation time vs n (log scale) ───────────────────────────────

def plot_computation_time(df: pd.DataFrame, save: bool = True):
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        for method in df['method'].unique():
            sub    = df[df['method'] == method].groupby('n')['time_mean']
            sizes  = list(sub.mean().index)
            means  = list(sub.mean().values)
            stds   = list(sub.std().fillna(0).values)

            ax.semilogy(
                sizes, means,
                marker    = _marker(method),
                color     = _color(method),
                label     = method,
                linewidth = 1.8,
                markersize = 7,
            )
            # Confidence band (±1 std in log space, approximated)
            lo = [max(1e-9, m - s) for m, s in zip(means, stds)]
            hi = [m + s for m, s in zip(means, stds)]
            ax.fill_between(sizes, lo, hi, color=_color(method), alpha=0.15)

        ax.set_xlabel('Number of items n')
        ax.set_ylabel('Mean computation time (s) – log scale')
        ax.set_title('Computation Time vs Instance Size')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)

        plt.tight_layout()
        if save:
            _save(fig, 'fig2_time')
        return fig


# ── Figure 3: % runs reaching optimal solution ────────────────────────────────

def plot_pct_optimal(df: pd.DataFrame, save: bool = True):
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(11, 5))

        methods = [m for m in df['method'].unique() if m != 'DP']
        sizes   = sorted(df['n'].unique())
        x       = np.arange(len(sizes))
        width   = 0.8 / max(len(methods), 1)

        for k, method in enumerate(methods):
            grp  = df[df['method'] == method].groupby('n')['pct_optimal']
            means = np.array([grp.mean().get(n, 0.0) for n in sizes])
            stds  = np.array([grp.std().get(n, 0.0)  for n in sizes])
            stds  = np.nan_to_num(stds)

            offset = x + k * width - 0.4 + width / 2
            bars = ax.bar(
                offset, means,
                width   = width * 0.88,
                label   = method,
                color   = _color(method),
                alpha   = 0.85,
                yerr    = stds,
                capsize = 3,
                error_kw={'elinewidth': 1.2, 'ecolor': '#555'},
            )

            for rect, val in zip(bars, means):
                if val > 1:
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + 1.5,
                        f'{val:.0f}%',
                        ha='center', va='bottom', fontsize=7,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([f'n={n}' for n in sizes])
        ax.set_ylim(0, 115)
        ax.set_ylabel('% runs reaching optimal solution')
        ax.set_title('Success Rate per Method and Instance Size')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        if save:
            _save(fig, 'fig3_pct_optimal')
        return fig


# ── Figure 4: QIGA convergence history ────────────────────────────────────────

def plot_qiga_convergence(save: bool = True):
    """
    Plot best fitness vs generation for QIGA.
    Tries to overlay multiple runs (with a mean ± std band) if available.
    Falls back to a single run if only one has history saved.
    """
    histories = []
    optimal   = None

    for fname in sorted(os.listdir(RESULTS_DIR)):
        if 'QIGA' not in fname or not fname.endswith('.json'):
            continue
        fpath = os.path.join(RESULTS_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        h = data.get('history', [])
        if not h:
            continue
        histories.append(h)
        if optimal is None:
            optimal = data.get('optimal')

    if not histories:
        print("No QIGA history found in results/. Re-run experiments with n<=30.")
        return None

    # Pad histories to the same length (in case different max_gen)
    max_len = max(len(h) for h in histories)
    arr = np.full((len(histories), max_len), np.nan)
    for i, h in enumerate(histories):
        arr[i, :len(h)] = h

    mean_h = np.nanmean(arr, axis=0)
    std_h  = np.nanstd(arr,  axis=0)
    gens   = np.arange(max_len)

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(9, 4))

        ax.plot(gens, mean_h, color='#3498db', linewidth=1.8,
                label=f'QIGA mean ({len(histories)} runs)')
        ax.fill_between(gens,
                         np.maximum(0, mean_h - std_h),
                         mean_h + std_h,
                         color='#3498db', alpha=0.18, label='±1 std')

        if len(histories) > 1:
            for h in histories:
                ax.plot(range(len(h)), h, color='#3498db', alpha=0.12, linewidth=0.8)

        if optimal is not None:
            ax.axhline(optimal, color='#e74c3c', linestyle='--', linewidth=1.4,
                       label=f'Optimal (DP) = {optimal}')

        ax.set_xlabel('Generation')
        ax.set_ylabel('Best fitness (profit)')
        ax.set_title('QIGA Convergence History')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        if save:
            _save(fig, 'fig4_qiga_convergence')
        return fig


# ── Figure 5: VQE/QAOA energy convergence ─────────────────────────────────────

def plot_energy_convergence(save: bool = True):
    """
    Plot energy vs optimiser iteration for VQE and QAOA.
    Shows one example trace per method.
    """
    method_histories: dict[str, list] = {}

    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.endswith('.json'):
            continue
        if not any(m in fname for m in ['VQE', 'QAOA']):
            continue
        fpath = os.path.join(RESULTS_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        h = data.get('history', [])
        if not h:
            continue
        method = data.get('method', fname)
        if method not in method_histories:
            method_histories[method] = h

    if not method_histories:
        print("No VQE/QAOA history found in results/. Re-run experiments with n<=30.")
        return None

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(9, 4))

        for method, history in sorted(method_histories.items()):
            ax.plot(
                history,
                label     = method,
                color     = _color(method),
                linewidth = 1.6,
                alpha     = 0.9,
            )

        ax.set_xlabel('Optimiser iteration')
        ax.set_ylabel('Expectation value ⟨H_C⟩')
        ax.set_title('VQE / QAOA Energy Convergence')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        if save:
            _save(fig, 'fig5_energy_convergence')
        return fig


# ── Figure 6: Summary heatmap (AR by method × size) ─────────────────────────

def plot_ar_heatmap(df: pd.DataFrame, save: bool = True):
    """Heatmap of mean approximation ratio: rows = methods, cols = sizes."""
    methods = [m for m in df['method'].unique() if m != 'DP']
    sizes   = sorted(df['n'].unique())

    matrix = np.full((len(methods), len(sizes)), np.nan)
    for i, method in enumerate(methods):
        for j, n in enumerate(sizes):
            vals = df[(df['method'] == method) & (df['n'] == n)]['ar_mean']
            if not vals.empty:
                matrix[i, j] = vals.mean()

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(max(6, len(sizes) * 0.9), max(3, len(methods) * 0.7)))

        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.8, vmax=1.0, aspect='auto')
        plt.colorbar(im, ax=ax, label='Approximation Ratio')

        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([f'n={n}' for n in sizes])
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        ax.set_title('Approximation Ratio Heatmap (mean over instances)')

        for i in range(len(methods)):
            for j in range(len(sizes)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                            fontsize=8, color='black' if val > 0.9 else 'white')

        plt.tight_layout()
        if save:
            _save(fig, 'fig6_ar_heatmap')
        return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_summary()
    print(df.head())
    print(f"\nMethods found: {sorted(df['method'].unique())}")
    print(f"Sizes found  : {sorted(df['n'].unique())}\n")

    plot_approximation_ratio(df)
    plot_computation_time(df)
    plot_pct_optimal(df)
    plot_qiga_convergence()
    plot_energy_convergence()
    plot_ar_heatmap(df)

    print(f"Figures saved to '{FIGURES_DIR}/'")
    plt.show()
