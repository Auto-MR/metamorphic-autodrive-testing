"""
Sweep Visualization — matplotlib figures for parametric sweep results.
Produces the key research charts showing how pass rate degrades with intensity.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional
from evaluation.parametric_sweep import SweepResult, SweepPoint, SWEEP_CATALOGUE


COLORS = {
    "dave2_cnn":      "#3498db",
    "gpr_steering":   "#2ecc71",
    "svr_steering":   "#e67e22",
    "random_forest":  "#9b59b6",
    "linear_baseline": "#e74c3c",
    # single model fallback
    "steering_regression": "#3498db",
    "gpr_steering_stub":   "#2ecc71",
}

MODEL_LABELS = {
    "dave2_cnn":      "DAVE-2 (CNN)",
    "gpr_steering":   "GPR",
    "svr_steering":   "SVR",
    "random_forest":  "Random Forest",
    "linear_baseline": "Linear Reg.",
    "steering_regression": "DAVE-2",
    "gpr_steering_stub":   "GPR (stub)",
}


def _model_color(name: str) -> str:
    for key, color in COLORS.items():
        if key in name:
            return color
    palette = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c", "#1abc9c", "#f39c12"]
    return palette[hash(name) % len(palette)]


def _model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


# ── 1. Single-model degradation curve ─────────────────────────────────────────

def plot_single_sweep(result: SweepResult, save_path: str = None) -> plt.Figure:
    """
    2-panel figure for one model + one transform:
      Top:    Pass Rate vs Intensity  (with critical threshold marked)
      Bottom: Mean Delta vs Intensity (showing how violation magnitude grows)
    """
    cfg   = SWEEP_CATALOGUE.get(result.transform_name, {})
    label = cfg.get("label", result.transform_name)
    unit  = cfg.get("unit", "")

    x  = result.intensities
    pr = result.pass_rates
    md = result.mean_deltas

    fig = plt.figure(figsize=(10, 7), facecolor="#1a1a2e")
    fig.suptitle(
        f"Parametric Sweep — {_model_label(result.model_name)}\n"
        f"MR: {result.mr_name}   |   Transform: {result.transform_name}",
        color="white", fontsize=13, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45, top=0.88, bottom=0.10)

    # ── Pass Rate panel ───────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#16213e")
    ax1.plot(x, pr, "o-", color="#3498db", lw=2.5, markersize=7, label="Pass Rate")
    ax1.fill_between(x, pr, alpha=0.15, color="#3498db")
    ax1.axhline(0.80, color="#f39c12", linestyle="--", lw=1.2, label="80% threshold (critical)")
    ax1.axhline(0.50, color="#e74c3c", linestyle=":",  lw=1.2, label="50% threshold (collapse)")

    if result.critical_threshold is not None:
        ax1.axvline(result.critical_threshold, color="#f39c12", linestyle="--", lw=1.2, alpha=0.7)
        ax1.text(result.critical_threshold, 0.02,
                 f"  Degrades at\n  {result.critical_threshold}{unit}",
                 color="#f39c12", fontsize=8, va="bottom")

    if result.collapse_threshold is not None:
        ax1.axvline(result.collapse_threshold, color="#e74c3c", linestyle=":", lw=1.2, alpha=0.7)
        ax1.text(result.collapse_threshold, 0.02,
                 f"  Collapses at\n  {result.collapse_threshold}{unit}",
                 color="#e74c3c", fontsize=8, va="bottom")

    ax1.set_xlabel(f"{label}{' (' + unit + ')' if unit else ''}", color="white")
    ax1.set_ylabel("MR Pass Rate", color="white")
    ax1.set_ylim(-0.05, 1.10)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], color="white")
    ax1.tick_params(colors="white")
    ax1.legend(facecolor="#16213e", labelcolor="white", fontsize=8)
    ax1.set_title("Robustness Degradation Curve", color="white", fontsize=10)
    for sp in ax1.spines.values():
        sp.set_edgecolor("#444")

    # ── Delta Growth panel ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#16213e")
    ax2.plot(x, md, "s-", color="#e67e22", lw=2.5, markersize=7, label="Mean |Δθ|")

    # Overlay tolerance line
    if hasattr(result, "_tolerance"):
        ax2.axhline(result._tolerance, color="#2ecc71", linestyle="--", lw=1.2, label=f"Tolerance ε={result._tolerance}")

    ax2.set_xlabel(f"{label}{' (' + unit + ')' if unit else ''}", color="white")
    ax2.set_ylabel("Mean |Δ Steering|", color="white")
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#16213e", labelcolor="white", fontsize=8)
    ax2.set_title("MR Delta Growth Curve", color="white", fontsize=10)
    for sp in ax2.spines.values():
        sp.set_edgecolor("#444")

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())

    return fig


# ── 2. Cross-model degradation comparison ─────────────────────────────────────

def plot_cross_model_sweep(
    results_map: Dict[str, SweepResult],
    save_path: str = None,
) -> plt.Figure:
    """
    2-panel overlay: all models on the same chart.
      Top:    Pass Rate vs Intensity, one line per model
      Bottom: Mean Delta vs Intensity, one line per model

    This is the KEY research figure showing cross-model generalization behavior.
    """
    if not results_map:
        return plt.figure()

    sample = next(iter(results_map.values()))
    cfg    = SWEEP_CATALOGUE.get(sample.transform_name, {})
    label  = cfg.get("label", sample.transform_name)
    unit   = cfg.get("unit", "")

    fig = plt.figure(figsize=(12, 8), facecolor="#1a1a2e")
    fig.suptitle(
        f"Cross-Model Parametric Sweep — {sample.transform_name.replace('_', ' ').title()}\n"
        f"MR: {sample.mr_name}",
        color="white", fontsize=13, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.50, top=0.88, bottom=0.10)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    for ax in (ax1, ax2):
        ax.set_facecolor("#16213e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.tick_params(colors="white")

    for model_name, result in results_map.items():
        color = _model_color(model_name)
        lbl   = _model_label(model_name)
        x     = result.intensities
        pr    = result.pass_rates
        md    = result.mean_deltas

        ax1.plot(x, pr, "o-", color=color, lw=2.2, markersize=6, label=lbl)
        ax2.plot(x, md, "s-", color=color, lw=2.2, markersize=6, label=lbl, alpha=0.9)

    ax1.axhline(0.80, color="#f39c12", linestyle="--", lw=1.2, label="80% threshold", alpha=0.8)
    ax1.axhline(0.50, color="#e74c3c", linestyle=":",  lw=1.0, label="50% threshold", alpha=0.8)
    ax1.set_ylabel("MR Pass Rate", color="white")
    ax1.set_ylim(-0.05, 1.10)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], color="white")
    ax1.legend(facecolor="#16213e", labelcolor="white", fontsize=8, ncol=2)
    ax1.set_title("Pass Rate Degradation — All Models", color="white", fontsize=10)
    ax1.set_xlabel(f"{label}{' (' + unit + ')' if unit else ''}", color="white")

    ax2.set_ylabel("Mean |Δ Steering|", color="white")
    ax2.legend(facecolor="#16213e", labelcolor="white", fontsize=8, ncol=2)
    ax2.set_title("Delta Growth — All Models", color="white", fontsize=10)
    ax2.set_xlabel(f"{label}{' (' + unit + ')' if unit else ''}", color="white")

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())

    return fig


# ── 3. Critical threshold comparison bar chart ────────────────────────────────

def plot_critical_thresholds(
    results_map: Dict[str, SweepResult],
    save_path: str = None,
) -> plt.Figure:
    """
    Horizontal bar chart: at what intensity does each model start failing?
    Higher bar = more robust.
    """
    sample = next(iter(results_map.values()))
    cfg    = SWEEP_CATALOGUE.get(sample.transform_name, {})
    label  = cfg.get("label", sample.transform_name)
    unit   = cfg.get("unit", "")

    model_names   = list(results_map.keys())
    thresholds    = [results_map[m].critical_threshold for m in model_names]
    max_val       = max(results_map[m].intensities[-1] for m in model_names)

    # Replace None (never violated) with max value
    display_vals  = [t if t is not None else max_val for t in thresholds]
    labels        = [_model_label(m) for m in model_names]
    colors        = [_model_color(m) for m in model_names]

    fig, ax = plt.subplots(figsize=(9, 4), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")

    bars = ax.barh(labels, display_vals, color=colors, edgecolor="#444", height=0.55)
    ax.set_xlabel(f"Critical Intensity ({label}{' ' + unit if unit else ''})", color="white")
    ax.set_title(
        f"Robustness Threshold — when does each model first fail?\n"
        f"(Higher = more robust to {sample.transform_name})",
        color="white", fontsize=10, fontweight="bold"
    )
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")

    for bar, val, thresh in zip(bars, display_vals, thresholds):
        annotation = f"{val}{unit}" if thresh is not None else f"Never fails >{val}{unit}"
        ax.text(bar.get_width() + max_val * 0.01, bar.get_y() + bar.get_height() / 2,
                annotation, va="center", color="white", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())

    return fig


# ── 4. Multi-transform heatmap ─────────────────────────────────────────────────

def plot_sweep_summary_heatmap(
    multi_results: Dict[str, Dict[str, SweepResult]],
    save_path: str = None,
) -> plt.Figure:
    """
    Heatmap: rows = transforms, cols = models, cells = average pass rate across sweep.
    Shows which models are robust to which transforms overall.

    Args:
        multi_results: { transform_name: { model_name: SweepResult } }
    """
    transforms = list(multi_results.keys())
    models     = list(next(iter(multi_results.values())).keys())

    data = np.zeros((len(transforms), len(models)))
    for i, tf in enumerate(transforms):
        for j, model in enumerate(models):
            pr_list = multi_results[tf][model].pass_rates
            data[i, j] = np.mean(pr_list)

    fig, ax = plt.subplots(figsize=(10, max(5, len(transforms) * 0.7 + 2)), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")

    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([_model_label(m) for m in models], color="white", rotation=20, ha="right")
    ax.set_yticks(range(len(transforms)))
    ax.set_yticklabels(transforms, color="white", fontsize=9)
    ax.set_title("Average Pass Rate Across Full Sweep\n(Green = robust, Red = fragile)", color="white", fontweight="bold")

    for i in range(len(transforms)):
        for j in range(len(models)):
            ax.text(j, i, f"{data[i, j]:.0%}", ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())

    return fig
