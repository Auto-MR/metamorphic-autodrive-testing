"""
Visualization — matplotlib helpers for MR test results AND
side-by-side image/output comparison panels.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Dict, Optional
from core.metamorphic_engine.mr_base import MRResult


# ── 1. Pass-rate bar chart ────────────────────────────────────────────────────

def plot_pass_rates(results_map: Dict[str, List[MRResult]], save_path: str = None) -> plt.Figure:
    from evaluation.robustness_scorer import score_by_model
    scores = score_by_model(results_map)
    names  = list(scores.keys())
    vals   = [scores[n] for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2ecc71" if v >= 0.8 else "#e67e22" if v >= 0.5 else "#e74c3c" for v in vals]
    ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Pass Rate")
    ax.set_title("Robustness Pass Rates by Model")
    ax.axhline(0.8, color="grey", linestyle="--", linewidth=0.8, label="80% threshold")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


# ── 2. Delta distribution histogram ──────────────────────────────────────────

def plot_delta_distribution(results: List[MRResult], save_path: str = None) -> plt.Figure:
    deltas = [r.delta for r in results if r.delta is not None]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(deltas, bins=30, color="#3498db", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Delta (violation magnitude)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of MR Delta Values")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


# ── 3. IMAGE COMPARISON PANEL ─────────────────────────────────────────────────

def make_comparison_figure(
    original:    np.ndarray,          # (H, W, 3) float32 [0,1]
    transformed: np.ndarray,          # (H, W, 3) float32 [0,1]
    transform_name: str,
    output_orig:    np.ndarray,        # depth map (H,W) or None
    output_trans:   np.ndarray,        # depth map (H,W) or None
    model_name:     str = "cnn_depth",
    mr_results:     Optional[List[MRResult]] = None,
    save_path:      str = None,
) -> plt.Figure:
    """
    4-panel figure:
      [Original image]  [Transformed image]
      [Output original] [Output diff heatmap]

    Works for depth maps (2-D output).
    For steering (scalar), collapses rows 2 into a bar comparison.
    """
    is_depth = output_orig is not None and output_orig.ndim == 2

    fig = plt.figure(figsize=(12, 9), facecolor="#1a1a2e")
    fig.suptitle(
        f"Metamorphic Test — {model_name}  |  transform: {transform_name}",
        color="white", fontsize=14, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(
        3 if mr_results else 2, 2,
        figure=fig, hspace=0.45, wspace=0.25,
        left=0.06, right=0.96, top=0.92, bottom=0.06
    )

    # ── Row 1: original vs transformed image ──────────────────────────────────
    ax_orig  = fig.add_subplot(gs[0, 0])
    ax_trans = fig.add_subplot(gs[0, 1])

    ax_orig.imshow(np.clip(original, 0, 1))
    ax_orig.set_title("Original Input", color="white", fontsize=11)
    ax_orig.axis("off")

    ax_trans.imshow(np.clip(transformed, 0, 1))
    ax_trans.set_title(f"Transformed ({transform_name})", color="white", fontsize=11)
    ax_trans.axis("off")

    # ── Row 2: model output original vs diff heatmap ──────────────────────────
    ax_out  = fig.add_subplot(gs[1, 0])
    ax_diff = fig.add_subplot(gs[1, 1])

    if is_depth:
        im1 = ax_out.imshow(output_orig, cmap="plasma", vmin=0, vmax=1)
        ax_out.set_title("Depth Map (original)", color="white", fontsize=11)
        ax_out.axis("off")
        plt.colorbar(im1, ax=ax_out, fraction=0.046, pad=0.04)

        diff = np.abs(output_orig - output_trans)
        im2  = ax_diff.imshow(diff, cmap="hot", vmin=0, vmax=0.3)
        ax_diff.set_title("|Depth Diff| Heatmap", color="white", fontsize=11)
        ax_diff.axis("off")
        cbar = plt.colorbar(im2, ax=ax_diff, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        mean_diff = float(diff.mean())
        ax_diff.set_xlabel(
            f"mean |diff| = {mean_diff:.4f}",
            color="#f39c12" if mean_diff > 0.05 else "#2ecc71",
            fontsize=10
        )
    else:
        # Steering: show colour-coded output text
        for ax in (ax_out, ax_diff):
            ax.set_facecolor("#16213e")
            ax.axis("off")

        if output_orig is not None:
            ax_out.text(0.5, 0.5, f"Steering\n{float(output_orig):.4f}",
                        ha="center", va="center", fontsize=22, color="#3498db",
                        fontweight="bold", transform=ax_out.transAxes)
            ax_out.set_title("Steering (original)", color="white", fontsize=11)

            ax_diff.text(0.5, 0.5, f"Steering\n{float(output_trans):.4f}",
                         ha="center", va="center", fontsize=22, color="#e67e22",
                         fontweight="bold", transform=ax_diff.transAxes)
            ax_diff.set_title("Steering (transformed)", color="white", fontsize=11)

            diff_val = abs(float(output_orig) + float(output_trans))   # antisymmetry check
            label_color = "#2ecc71" if diff_val < 0.1 else "#e74c3c"
            fig.text(0.5, 0.35, f"|y + y_flip| = {diff_val:.4f}",
                     ha="center", color=label_color, fontsize=12, fontweight="bold")

    # ── Row 3 (optional): MR result summary table ─────────────────────────────
    if mr_results:
        ax_tbl = fig.add_subplot(gs[2, :])
        ax_tbl.axis("off")
        ax_tbl.set_facecolor("#16213e")

        col_labels = ["MR Name", "Pass/Fail", "Delta", "Message"]
        rows = [
            [r.mr_name,
             "PASS" if r.passed else "FAIL",
             f"{r.delta:.4f}" if r.delta is not None else "—",
             r.message or ""]
            for r in mr_results
        ]
        tbl = ax_tbl.table(
            cellText=rows,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.6)

        # Colour rows by pass/fail
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor("#444466")
            if row == 0:
                cell.set_facecolor("#2c2c54")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                passed = mr_results[row - 1].passed
                cell.set_facecolor("#1a3a1a" if passed else "#3a1a1a")
                cell.set_text_props(color="#ccffcc" if passed else "#ffcccc")

        ax_tbl.set_title("MR Verification Results", color="white", fontsize=11, pad=8)

    # Dark background on all axes
    for ax in fig.get_axes():
        ax.set_facecolor("#16213e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"[Visualization] Saved → {save_path}")

    return fig


# ── 4. STEERING GAUGE ────────────────────────────────────────────────────────

def make_steering_gauge(angle_orig: float, angle_trans: float, transform_name: str) -> plt.Figure:
    """
    Semicircular gauge comparing original vs transformed steering angle.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), subplot_kw={"projection": "polar"},
                             facecolor="#1a1a2e")

    for ax, angle, label, color in zip(
        axes,
        [angle_orig, angle_trans],
        ["Original", f"After {transform_name}"],
        ["#3498db", "#e67e22"]
    ):
        ax.set_facecolor("#16213e")
        ax.set_thetamin(-90)
        ax.set_thetamax(90)
        ax.set_ylim(0, 1)
        ax.set_rticks([])
        ax.set_thetagrids([-90, -45, 0, 45, 90], labels=["-90°", "-45°", "0°", "+45°", "+90°"],
                          color="grey", fontsize=8)

        # Draw needle
        theta = np.radians(angle * 90)   # angle in [-1,1] → radians
        ax.annotate("", xy=(theta, 0.85), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5))
        ax.set_title(f"{label}\n{angle:.3f}", color="white", fontsize=10, pad=12)

    fig.suptitle("Steering Angle Gauge", color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig
