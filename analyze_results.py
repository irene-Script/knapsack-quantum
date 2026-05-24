"""
analyze_results.py
------------------
Charge tous les fichiers JSON de results/, calcule les métriques agrégées,
produit des tableaux + graphiques, et génère un rapport HTML final.

Usage:
    python analyze_results.py
    python analyze_results.py --results-dir results/ --out rapport_final.html
"""

import os
import re
import json
import math
import argparse
import base64
import io
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

# ── Chemins par défaut ─────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, 'results')
OUT_FILE    = os.path.join(ROOT, 'rapport_final.html')

# ── Palette ────────────────────────────────────────────────────────────────────
COLORS = {
    'DP'      : '#27ae60',
    'QIGA'    : '#2980b9',
    'VQE'     : '#e67e22',
    'QAOA_p2' : '#8e44ad',
}
MARKERS = {'DP': 's', 'QIGA': 'o', 'VQE': '^', 'QAOA_p2': 'D'}
METHOD_ORDER = ['DP', 'QIGA', 'VQE', 'QAOA_p2']

RC = {
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 130,
}

# ─────────────────────────────────────────────────────────────────────────────
#  1. CHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def load_all_results(results_dir: str) -> pd.DataFrame:
    """
    Parcourt tous les .json de results_dir et construit un DataFrame plat.
    Chaque ligne = un run individuel.
    """
    records = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(results_dir, fname)
        with open(fpath, encoding='utf-8') as f:
            try:
                d = json.load(f)
            except json.JSONDecodeError:
                continue

        # Extraire le run index depuis le nom de fichier
        m = re.search(r'_run(\d+)\.json$', fname)
        run_idx = int(m.group(1)) if m else 0

        records.append({
            'file'        : fname,
            'run_idx'     : run_idx,
            'method'      : d.get('method', 'Unknown'),
            'instance'    : d.get('instance', ''),
            'n'           : int(d.get('n', 0)),
            'optimal'     : d.get('optimal'),
            'best_value'  : float(d.get('best_value', 0)),
            'time_sec'    : float(d.get('time_sec', 0)),
            'history_len' : int(d.get('history_len', 0)),
            'history'     : d.get('history', []),
            'opt_energy'  : d.get('opt_energy'),
        })

    df = pd.DataFrame(records)

    # Calcul AR et gap
    df['ar'] = np.where(
        df['optimal'].notna() & (df['optimal'] > 0),
        df['best_value'] / df['optimal'].astype(float),
        np.nan
    )
    df['gap_pct'] = np.where(
        df['optimal'].notna() & (df['optimal'] > 0),
        (df['optimal'].astype(float) - df['best_value']) / df['optimal'].astype(float) * 100,
        np.nan
    )
    df['is_optimal'] = df['ar'].apply(lambda x: bool(x is not None and not math.isnan(x) and x >= 1 - 1e-4))

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  2. MÉTRIQUES AGRÉGÉES
# ─────────────────────────────────────────────────────────────────────────────

def compute_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège par (method, n, instance) : mean / std / best / pct_optimal / time.
    """
    rows = []
    for (method, n, instance), g in df.groupby(['method', 'n', 'instance']):
        vals    = g['best_value'].values
        times   = g['time_sec'].values
        optimal = g['optimal'].iloc[0]
        rows.append({
            'method'     : method,
            'n'          : n,
            'instance'   : instance,
            'optimal'    : optimal,
            'best'       : float(np.max(vals)),
            'mean'       : float(np.mean(vals)),
            'std'        : float(np.std(vals)),
            'ar_mean'    : float(np.mean(g['ar'].dropna())),
            'ar_best'    : float(np.max(g['ar'].dropna())),
            'gap_mean'   : float(np.mean(g['gap_pct'].dropna())),
            'pct_optimal': float(g['is_optimal'].mean() * 100),
            'n_runs'     : len(g),
            'time_mean'  : float(np.mean(times)),
            'time_std'   : float(np.std(times)),
        })
    return pd.DataFrame(rows)


def global_summary(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Table globale : une ligne par (method, n).
    """
    rows = []
    for (method, n), g in agg.groupby(['method', 'n']):
        rows.append({
            'Méthode'       : method,
            'n'             : n,
            'AR moyen'      : round(g['ar_mean'].mean(), 4),
            'AR std'        : round(g['ar_mean'].std(), 4),
            'AR min'        : round(g['ar_mean'].min(), 4),
            'AR max'        : round(g['ar_mean'].max(), 4),
            'Gap moyen (%)'  : round(g['gap_mean'].mean(), 2),
            '% optimal'     : round(g['pct_optimal'].mean(), 1),
            'Temps moyen (s)': round(g['time_mean'].mean(), 4),
            'Temps std (s)'  : round(g['time_std'].mean(), 4),
            'Instances'     : len(g),
        })
    return pd.DataFrame(rows).sort_values(['n', 'Méthode'])


def worst_instances(agg: pd.DataFrame, n_top: int = 5) -> pd.DataFrame:
    """Les instances les plus difficiles (AR le plus bas) par méthode."""
    sub = agg[agg['method'] != 'DP'].copy()
    sub = sub.sort_values('ar_mean').head(n_top * len(sub['method'].unique()))
    return sub[['method','instance','n','ar_mean','gap_mean','pct_optimal']].rename(columns={
        'method': 'Méthode', 'instance': 'Instance', 'n': 'n',
        'ar_mean': 'AR moyen', 'gap_mean': 'Gap (%)', 'pct_optimal': '% optimal'
    })


# ─────────────────────────────────────────────────────────────────────────────
#  3. FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    """Encode une figure matplotlib en base64 PNG pour l'inclure dans HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _color(m): return COLORS.get(m, '#7f8c8d')
def _marker(m): return MARKERS.get(m, 'o')


def fig_ar_by_size(agg: pd.DataFrame) -> str:
    """Fig 1 — Bar chart AR moyen ± std par méthode et taille."""
    with plt.rc_context(RC):
        methods  = [m for m in METHOD_ORDER if m in agg['method'].unique() and m != 'DP']
        sizes    = sorted(agg['n'].unique())
        x        = np.arange(len(sizes))
        width    = 0.8 / max(len(methods), 1)

        fig, ax = plt.subplots(figsize=(11, 5))

        for k, method in enumerate(methods):
            g     = agg[agg['method'] == method].groupby('n')['ar_mean']
            means = np.array([g.mean().get(n, np.nan) for n in sizes])
            stds  = np.nan_to_num(np.array([g.std().get(n, 0.0) for n in sizes]))

            offsets = x + k * width - 0.4 + width / 2
            bars = ax.bar(offsets, means, width=width*0.88, label=method,
                          color=_color(method), alpha=0.85,
                          yerr=stds, capsize=3,
                          error_kw={'elinewidth': 1.2, 'ecolor': '#444'})

            for rect, val, std in zip(bars, means, stds):
                if not np.isnan(val):
                    ax.text(rect.get_x() + rect.get_width()/2,
                            rect.get_height() + std + 0.002,
                            f'{val:.4f}', ha='center', va='bottom', fontsize=7, rotation=90)

        ax.axhline(1.0, color='black', ls='--', lw=1.2, label='Optimal (DP)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'n={n}' for n in sizes])

        valid = agg[agg['method'] != 'DP']['ar_mean'].dropna()
        lo = max(0.0, valid.min() - 0.07) if not valid.empty else 0.0
        ax.set_ylim(lo, 1.10)
        ax.set_ylabel('Approximation Ratio (plus haut = meilleur)')
        ax.set_title('Fig 1 — Approximation Ratio moyen par taille et méthode')
        ax.legend(loc='lower left')
        ax.grid(axis='y', alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
        plt.tight_layout()
        return _fig_to_b64(fig)


def fig_time_logscale(agg: pd.DataFrame) -> str:
    """Fig 2 — Temps de calcul vs n (log scale)."""
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(8, 5))

        for method in METHOD_ORDER:
            if method not in agg['method'].unique():
                continue
            g     = agg[agg['method'] == method].groupby('n')['time_mean']
            sizes = list(g.mean().index)
            means = list(g.mean().values)
            stds  = list(agg[agg['method'] == method].groupby('n')['time_std'].mean().fillna(0).values)

            ax.semilogy(sizes, means, marker=_marker(method), color=_color(method),
                        label=method, lw=1.8, ms=8)
            lo = [max(1e-9, m - s) for m, s in zip(means, stds)]
            hi = [m + s for m, s in zip(means, stds)]
            ax.fill_between(sizes, lo, hi, color=_color(method), alpha=0.15)

        ax.set_xlabel('Nombre d\'items n')
        ax.set_ylabel('Temps moyen (s) — échelle log')
        ax.set_title('Fig 2 — Temps de calcul vs taille d\'instance')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        return _fig_to_b64(fig)


def fig_pct_optimal(agg: pd.DataFrame) -> str:
    """Fig 3 — % runs atteignant l'optimal."""
    with plt.rc_context(RC):
        methods = [m for m in METHOD_ORDER if m in agg['method'].unique() and m != 'DP']
        sizes   = sorted(agg['n'].unique())
        x       = np.arange(len(sizes))
        width   = 0.8 / max(len(methods), 1)

        fig, ax = plt.subplots(figsize=(11, 5))

        for k, method in enumerate(methods):
            g     = agg[agg['method'] == method].groupby('n')['pct_optimal']
            means = np.array([g.mean().get(n, 0.0) for n in sizes])
            stds  = np.nan_to_num(np.array([g.std().get(n, 0.0) for n in sizes]))

            offsets = x + k * width - 0.4 + width / 2
            bars = ax.bar(offsets, means, width=width*0.88, label=method,
                          color=_color(method), alpha=0.85,
                          yerr=stds, capsize=3,
                          error_kw={'elinewidth': 1.2, 'ecolor': '#444'})

            for rect, val in zip(bars, means):
                if val > 2:
                    ax.text(rect.get_x() + rect.get_width()/2,
                            rect.get_height() + 1.5,
                            f'{val:.0f}%', ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([f'n={n}' for n in sizes])
        ax.set_ylim(0, 118)
        ax.set_ylabel('% runs atteignant la solution optimale')
        ax.set_title('Fig 3 — Taux de succès par méthode et taille')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        return _fig_to_b64(fig)


def fig_ar_boxplot(df: pd.DataFrame) -> str:
    """Fig 4 — Boxplot de la distribution des AR par méthode."""
    with plt.rc_context(RC):
        methods = [m for m in METHOD_ORDER if m in df['method'].unique() and m != 'DP']
        sizes   = sorted(df['n'].unique())

        fig, axes = plt.subplots(1, len(sizes), figsize=(4 * len(sizes), 5),
                                 sharey=True, constrained_layout=True)
        if len(sizes) == 1:
            axes = [axes]

        for ax, n in zip(axes, sizes):
            data_by_method = [
                df[(df['method'] == m) & (df['n'] == n)]['ar'].dropna().values
                for m in methods
            ]
            bp = ax.boxplot(
                data_by_method,
                patch_artist=True,
                medianprops={'color': 'black', 'lw': 2},
                whiskerprops={'lw': 1.2},
                flierprops={'marker': '.', 'ms': 4, 'alpha': 0.5},
            )
            for patch, method in zip(bp['boxes'], methods):
                patch.set_facecolor(_color(method))
                patch.set_alpha(0.75)

            ax.axhline(1.0, color='black', ls='--', lw=1, alpha=0.7)
            ax.set_title(f'n = {n}')
            ax.set_xticks(range(1, len(methods) + 1))
            ax.set_xticklabels(methods, rotation=20, ha='right')
            ax.grid(axis='y', alpha=0.3)

        axes[0].set_ylabel('Approximation Ratio')
        fig.suptitle('Fig 4 — Distribution des AR (boxplots par taille)', fontsize=13)
        return _fig_to_b64(fig)


def fig_qiga_convergence(df: pd.DataFrame) -> str:
    """Fig 5 — Courbes de convergence QIGA (moyenne ± std sur toutes les runs)."""
    with plt.rc_context(RC):
        qiga_df = df[(df['method'] == 'QIGA') & (df['history'].apply(len) > 0)].copy()

        if qiga_df.empty:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.text(0.5, 0.5, 'Aucun historique QIGA disponible',
                    ha='center', va='center', transform=ax.transAxes)
            return _fig_to_b64(fig)

        sizes = sorted(qiga_df['n'].unique())
        n_cols = len(sizes)
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4.5),
                                 constrained_layout=True)
        if n_cols == 1:
            axes = [axes]

        for ax, n in zip(axes, sizes):
            sub      = qiga_df[qiga_df['n'] == n]
            histories = [np.array(h) for h in sub['history']]
            max_len   = max(len(h) for h in histories)

            mat = np.full((len(histories), max_len), np.nan)
            for i, h in enumerate(histories):
                mat[i, :len(h)] = h

            mean_h = np.nanmean(mat, axis=0)
            std_h  = np.nanstd(mat, axis=0)
            gens   = np.arange(max_len)

            # Individual traces (faint)
            for h in histories:
                ax.plot(range(len(h)), h, color='#2980b9', alpha=0.08, lw=0.7)

            ax.plot(gens, mean_h, color='#2980b9', lw=2,
                    label=f'Moyenne ({len(histories)} runs)')
            ax.fill_between(gens,
                            np.maximum(0, mean_h - std_h),
                            mean_h + std_h,
                            color='#2980b9', alpha=0.2, label='±1 std')

            # Optimal
            opt_vals = sub['optimal'].dropna()
            if not opt_vals.empty:
                opt = opt_vals.median()
                ax.axhline(opt, color='#e74c3c', ls='--', lw=1.4,
                           label=f'Optimal médian = {opt:.0f}')

            ax.set_title(f'n = {n}')
            ax.set_xlabel('Génération')
            ax.set_ylabel('Meilleur profit')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        fig.suptitle('Fig 5 — Convergence QIGA par taille', fontsize=13)
        return _fig_to_b64(fig)


def fig_energy_convergence(df: pd.DataFrame) -> str:
    """Fig 6 — Convergence énergie VQE / QAOA (normalisée 0→1)."""
    with plt.rc_context(RC):
        target_methods = ['VQE', 'QAOA_p2']
        sub = df[df['method'].isin(target_methods) & (df['history'].apply(len) > 0)]

        if sub.empty:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.text(0.5, 0.5, 'Aucun historique VQE/QAOA disponible',
                    ha='center', va='center', transform=ax.transAxes)
            return _fig_to_b64(fig)

        sizes   = sorted(sub['n'].unique())
        n_cols  = len(sizes)
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4.5),
                                 constrained_layout=True)
        if n_cols == 1:
            axes = [axes]

        for ax, n in zip(axes, sizes):
            for method in target_methods:
                msub = sub[(sub['method'] == method) & (sub['n'] == n)]
                if msub.empty:
                    continue

                # Prend le premier run disponible comme exemple
                h = np.array(msub.iloc[0]['history'], dtype=float)
                # Normalise pour pouvoir comparer VQE et QAOA sur la même échelle
                h_min, h_max = h.min(), h.max()
                h_norm = (h - h_min) / (h_max - h_min + 1e-12)

                ax.plot(h_norm, color=_color(method), lw=1.8, label=method, alpha=0.9)

            ax.set_title(f'n = {n}')
            ax.set_xlabel('Itération optimiseur')
            ax.set_ylabel('Energie normalisee (0=min, 1=max)')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        fig.suptitle('Fig 6 — Convergence énergie VQE / QAOA (normalisée)', fontsize=13)
        return _fig_to_b64(fig)


def fig_ar_heatmap(agg: pd.DataFrame) -> str:
    """Fig 7 — Heatmap AR : lignes = méthodes, colonnes = instances."""
    with plt.rc_context(RC):
        methods  = [m for m in METHOD_ORDER if m in agg['method'].unique() and m != 'DP']
        # Trier les instances par (n, id)
        instances = sorted(agg['instance'].unique(),
                           key=lambda s: (int(re.search(r'n(\d+)', s).group(1)),
                                          int(re.search(r'id(\d+)', s).group(1))))

        matrix = np.full((len(methods), len(instances)), np.nan)
        for i, method in enumerate(methods):
            for j, inst in enumerate(instances):
                vals = agg[(agg['method'] == method) & (agg['instance'] == inst)]['ar_mean']
                if not vals.empty:
                    matrix[i, j] = vals.mean()

        cmap = LinearSegmentedColormap.from_list('rg', ['#e74c3c', '#f39c12', '#27ae60'])
        fig_w = max(12, len(instances) * 0.35)
        fig_h = max(3,  len(methods)   * 0.75)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(matrix, cmap=cmap, vmin=0.80, vmax=1.0, aspect='auto')
        plt.colorbar(im, ax=ax, label='AR moyen', shrink=0.8)

        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        ax.set_xticks(range(len(instances)))
        ax.set_xticklabels(instances, rotation=75, ha='right', fontsize=7)
        ax.set_title('Fig 7 — Heatmap Approximation Ratio (méthode × instance)')

        for i in range(len(methods)):
            for j in range(len(instances)):
                val = matrix[i, j]
                if not np.isnan(val):
                    txt_color = 'white' if val < 0.92 else 'black'
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                            fontsize=6, color=txt_color)

        plt.tight_layout()
        return _fig_to_b64(fig)


def fig_gap_distribution(df: pd.DataFrame) -> str:
    """Fig 8 — Histogramme du gap d'optimalité par méthode."""
    with plt.rc_context(RC):
        methods = [m for m in METHOD_ORDER if m in df['method'].unique() and m != 'DP']
        fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4),
                                 constrained_layout=True)
        if len(methods) == 1:
            axes = [axes]

        for ax, method in zip(axes, methods):
            gaps = df[df['method'] == method]['gap_pct'].dropna()
            ax.hist(gaps, bins=20, color=_color(method), alpha=0.8, edgecolor='white')
            ax.axvline(gaps.mean(), color='black', ls='--', lw=1.5,
                       label=f'Moyenne = {gaps.mean():.2f}%')
            ax.axvline(0, color='#27ae60', ls='-', lw=1.2, label='Optimal (0%)')
            ax.set_title(method)
            ax.set_xlabel('Gap d\'optimalité (%)')
            ax.set_ylabel('Nombre de runs')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        fig.suptitle('Fig 8 — Distribution du gap d\'optimalité par méthode', fontsize=13)
        return _fig_to_b64(fig)


def fig_time_vs_ar(agg: pd.DataFrame) -> str:
    """Fig 9 — Scatter temps vs AR par run (trade-off qualité/vitesse)."""
    with plt.rc_context(RC):
        methods = [m for m in METHOD_ORDER if m in agg['method'].unique() and m != 'DP']
        fig, ax = plt.subplots(figsize=(8, 5))

        for method in methods:
            sub = agg[agg['method'] == method]
            ax.scatter(sub['time_mean'], sub['ar_mean'],
                       color=_color(method), label=method,
                       alpha=0.65, s=50, marker=_marker(method))

        ax.set_xscale('log')
        ax.set_xlabel('Temps moyen par run (s) — échelle log')
        ax.set_ylabel('AR moyen')
        ax.set_title('Fig 9 — Trade-off Temps vs Qualité de solution')
        ax.axhline(1.0, color='black', ls='--', lw=1, alpha=0.5)
        ax.legend()
        ax.grid(alpha=0.3, which='both')
        plt.tight_layout()
        return _fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  4. RAPPORT HTML
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #f5f6fa;
       color: #2c3e50; padding: 24px; }
h1 { font-size: 2em; border-bottom: 3px solid #2980b9; padding-bottom: 10px;
     margin-bottom: 20px; }
h2 { font-size: 1.35em; margin: 30px 0 10px; color: #2980b9; border-left: 4px solid #2980b9;
     padding-left: 10px; }
h3 { font-size: 1.1em; margin: 20px 0 8px; color: #555; }
p  { margin: 6px 0 12px; line-height: 1.6; }
.card { background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,.07);
        padding: 20px; margin-bottom: 22px; overflow-x: auto; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }
.kpi { background: white; border-radius: 10px; padding: 16px 20px;
       box-shadow: 0 2px 10px rgba(0,0,0,.07); text-align: center; }
.kpi .val { font-size: 2em; font-weight: 700; color: #2980b9; }
.kpi .lbl { font-size: 0.82em; color: #888; margin-top: 4px; }
table { border-collapse: collapse; width: 100%; font-size: 0.88em; }
thead tr { background: #2980b9; color: white; }
th, td { padding: 7px 10px; text-align: right; border-bottom: 1px solid #eee; }
td:first-child, th:first-child { text-align: left; }
tr:nth-child(even) { background: #f9f9f9; }
tr:hover { background: #eaf4fb; }
img { max-width: 100%; border-radius: 8px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 12px;
         font-size: 0.78em; font-weight: 600; }
.green { background: #d5f5e3; color: #1e8449; }
.orange { background: #fdebd0; color: #b9770e; }
.red { background: #fadbd8; color: #c0392b; }
.footer { text-align: center; color: #aaa; font-size: 0.8em; margin-top: 40px; }
"""

def _badge(val, thresholds=(0.99, 0.95)):
    if val >= thresholds[0]:
        return f'<span class="badge green">{val:.4f}</span>'
    elif val >= thresholds[1]:
        return f'<span class="badge orange">{val:.4f}</span>'
    return f'<span class="badge red">{val:.4f}</span>'


def _df_to_html(df: pd.DataFrame, badge_col: str = None, fmt: dict = None) -> str:
    if df.empty:
        return '<p><em>Aucune donnée.</em></p>'

    fmt = fmt or {}

    def _fmt_cell(col, val):
        if col == badge_col and isinstance(val, float):
            return _badge(val)
        if col in fmt:
            return fmt[col].format(val)
        if isinstance(val, float):
            return f'{val:.4f}'
        return str(val)

    header = '<thead><tr>' + ''.join(f'<th>{c}</th>' for c in df.columns) + '</tr></thead>'
    rows = []
    for _, row in df.iterrows():
        cells = ''.join(f'<td>{_fmt_cell(col, row[col])}</td>' for col in df.columns)
        rows.append(f'<tr>{cells}</tr>')
    body = '<tbody>' + ''.join(rows) + '</tbody>'
    return f'<table>{header}{body}</table>'


def build_html_report(
    df: pd.DataFrame,
    agg: pd.DataFrame,
    summary_tbl: pd.DataFrame,
    worst_tbl: pd.DataFrame,
    figures: dict,
    out_path: str,
):
    n_runs_total = len(df)
    n_instances  = agg['instance'].nunique()
    methods      = sorted(df['method'].unique())
    n_methods    = len(methods)
    avg_ar_qiga  = agg[agg['method'] == 'QIGA']['ar_mean'].mean()
    avg_ar_vqe   = agg[agg['method'] == 'VQE']['ar_mean'].mean()
    avg_ar_qaoa  = agg[agg['method'] == 'QAOA_p2']['ar_mean'].mean()

    # ── KPI cards ──────────────────────────────────────────────────────────────
    kpis_html = f"""
    <div class="grid3">
      <div class="kpi"><div class="val">{n_runs_total}</div><div class="lbl">Runs analysés</div></div>
      <div class="kpi"><div class="val">{n_instances}</div><div class="lbl">Instances testées</div></div>
      <div class="kpi"><div class="val">{n_methods}</div><div class="lbl">Méthodes comparées</div></div>
      <div class="kpi"><div class="val">{avg_ar_qiga:.4f}</div><div class="lbl">AR moyen — QIGA</div></div>
      <div class="kpi"><div class="val">{avg_ar_vqe:.4f}</div><div class="lbl">AR moyen — VQE</div></div>
      <div class="kpi"><div class="val">{avg_ar_qaoa:.4f}</div><div class="lbl">AR moyen — QAOA_p2</div></div>
    </div>"""

    # ── Tableau récapitulatif ──────────────────────────────────────────────────
    summary_html = _df_to_html(summary_tbl, badge_col='AR moyen')

    # ── Pires instances ────────────────────────────────────────────────────────
    worst_html = _df_to_html(worst_tbl.head(15))

    # ── Interprétation automatique ─────────────────────────────────────────────
    n_values = sorted(agg['n'].unique())
    interp_lines = []
    for n in n_values:
        sub = agg[agg['n'] == n]
        for method in ['QIGA', 'VQE', 'QAOA_p2']:
            msub = sub[sub['method'] == method]
            if msub.empty:
                continue
            ar   = msub['ar_mean'].mean()
            pct  = msub['pct_optimal'].mean()
            gap  = msub['gap_mean'].mean()
            qual = 'excellente' if ar >= 0.99 else ('bonne' if ar >= 0.95 else 'faible')
            interp_lines.append(
                f"<li><b>{method} (n={n})</b> : AR = {ar:.4f} → qualité <em>{qual}</em>, "
                f"gap moyen = {gap:.2f}%, {pct:.1f}% des runs atteignent l'optimal.</li>"
            )

    # Comparaison DP vs autres
    dp_sub  = agg[agg['method'] == 'DP']
    dp_note = f"DP résout {len(dp_sub)} instances de manière exacte (AR = 1.000) en {dp_sub['time_mean'].mean():.5f}s en moyenne."

    # Meilleure méthode quantique
    quantum_methods = ['QIGA', 'VQE', 'QAOA_p2']
    q_ars = {m: agg[agg['method'] == m]['ar_mean'].mean() for m in quantum_methods if m in agg['method'].unique()}
    best_q = max(q_ars, key=q_ars.get) if q_ars else 'N/A'

    # ── Recommandations ────────────────────────────────────────────────────────
    rec_lines = [
        f"<li><b>Pour la qualité</b> : <b>{best_q}</b> obtient le meilleur AR moyen ({q_ars.get(best_q, 0):.4f}) parmi les méthodes quantiques testées.</li>",
        "<li><b>Pour la vitesse</b> : QAOA_p2 est généralement plus rapide que VQE sur les petites instances.</li>",
        "<li><b>Scalabilité</b> : QIGA gère des instances plus grandes (n=20) où VQE/QAOA sont limités par la simulation quantique.</li>",
        "<li><b>Robustesse</b> : Augmenter <code>pop_size</code> (QIGA) ou <code>reps</code> (VQE) améliore le taux de succès au coût d'un temps de calcul accru.</li>",
    ]

    # ── Assemblage HTML ────────────────────────────────────────────────────────
    def img(b64): return f'<img src="data:image/png;base64,{b64}" />'

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>Rapport Final — Knapsack Quantum</title>
  <style>{_CSS}</style>
</head>
<body>

<h1>Rapport Final — Solveurs Quantiques pour le Problème du Sac à Dos (0-1 KP)</h1>
<p>Généré automatiquement à partir de <b>{n_runs_total}</b> runs sur <b>{n_instances}</b> instances — méthodes : {', '.join(f'<code>{m}</code>' for m in methods)}.</p>

<!-- KPIs -->
<h2>1. Vue d'ensemble</h2>
{kpis_html}

<!-- Interprétation -->
<div class="card">
<h3>Interprétation par méthode</h3>
<ul style="padding-left:20px;line-height:2;">
{''.join(interp_lines)}
</ul>
<p style="margin-top:10px;">Référence : <em>{dp_note}</em></p>
</div>

<!-- Table récap -->
<h2>2. Tableau récapitulatif global</h2>
<div class="card">
<p>AR moyen par (méthode, taille) avec badge couleur :
  <span class="badge green">≥ 0.99</span>
  <span class="badge orange">≥ 0.95</span>
  <span class="badge red">&lt; 0.95</span>
</p>
{summary_html}
</div>

<!-- Figures 1-3 -->
<h2>3. Approximation Ratio et Taux de Succès</h2>
<div class="card">{img(figures['ar_by_size'])}</div>
<div class="grid2">
  <div class="card">{img(figures['pct_optimal'])}</div>
  <div class="card">{img(figures['ar_boxplot'])}</div>
</div>

<!-- Figure heatmap -->
<h2>4. Heatmap détaillée (méthode × instance)</h2>
<div class="card">{img(figures['ar_heatmap'])}</div>

<!-- Figures temps -->
<h2>5. Temps de calcul et Trade-off Qualité/Vitesse</h2>
<div class="grid2">
  <div class="card">{img(figures['time_logscale'])}</div>
  <div class="card">{img(figures['time_vs_ar'])}</div>
</div>

<!-- Convergence -->
<h2>6. Convergence des algorithmes</h2>
<div class="card">{img(figures['qiga_convergence'])}</div>
<div class="card">{img(figures['energy_convergence'])}</div>

<!-- Distribution gap -->
<h2>7. Distribution du gap d'optimalité</h2>
<div class="card">{img(figures['gap_distribution'])}</div>

<!-- Pires instances -->
<h2>8. Instances les plus difficiles</h2>
<div class="card">
<p>Les instances où le gap moyen est le plus élevé (toutes méthodes, toutes tailles).</p>
{worst_html}
</div>

<!-- Recommandations -->
<h2>9. Recommandations</h2>
<div class="card">
<ul style="padding-left:20px;line-height:2;">
{''.join(rec_lines)}
</ul>
</div>

<div class="footer">Généré par analyze_results.py — Knapsack Quantum Project</div>
</body>
</html>"""

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Rapport sauvegardé : {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
#  5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Analyse complète des résultats KP')
    p.add_argument('--results-dir', default=RESULTS_DIR)
    p.add_argument('--out', default=OUT_FILE)
    return p.parse_args()


def main():
    args = parse_args()

    print(f'Chargement de {args.results_dir} ...')
    df = load_all_results(args.results_dir)
    print(f'  >> {len(df)} runs charges  |  methodes : {sorted(df["method"].unique())}  |  tailles : {sorted(df["n"].unique())}')

    print('Calcul des agregats ...')
    agg = compute_aggregates(df)

    summary_tbl = global_summary(agg)
    worst_tbl   = worst_instances(agg, n_top=5)

    print('Generation des figures ...')
    figures = {
        'ar_by_size'       : fig_ar_by_size(agg),
        'time_logscale'    : fig_time_logscale(agg),
        'pct_optimal'      : fig_pct_optimal(agg),
        'ar_boxplot'       : fig_ar_boxplot(df),
        'qiga_convergence' : fig_qiga_convergence(df),
        'energy_convergence': fig_energy_convergence(df),
        'ar_heatmap'       : fig_ar_heatmap(agg),
        'gap_distribution' : fig_gap_distribution(df),
        'time_vs_ar'       : fig_time_vs_ar(agg),
    }
    print(f'  >> {len(figures)} figures generees')

    print('Ecriture du rapport HTML ...')
    build_html_report(df, agg, summary_tbl, worst_tbl, figures, args.out)

    # Affiche aussi le résumé dans le terminal
    print('\n' + '='*70)
    print('RÉSUMÉ TERMINAL')
    print('='*70)
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 120)
    print(summary_tbl.to_string(index=False))
    print('='*70)


if __name__ == '__main__':
    main()
