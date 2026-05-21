"""
============================================================
 VISUALIZE ANTI-SPOOFING RESULTS  --  v1
============================================================
Reads results/Spoof_Report.xlsx (produced by test_anti_spoofing.py)
and exports clean academic-style PNG figures for the thesis.

Outputs (in results/report_images/):
  - 20_summary_table.png         Per-video table
  - 21_group_summary_table.png   Per-group averages
  - 22_metrics_table.png         APCER / BPCER / ACER  at Default + EER
  - 23_roc_curve.png             APCER vs BPCER threshold sweep
  - 24_score_distribution.png    Live-score histogram split by ground truth
  - 25_confusion_matrices.png    2 x 2 confusion matrices at Default + EER
  - 26_infographic.png           One-page summary
  - 27_sample_frames.png         3 rows (user / mask / spoof) x 5 cols
  - 28_confusion_samples.png     2 frames each per cell (TP / TN / FP / FN)

Style guide is reused from visualize_results.py (same palette, fonts and
card-drawing helpers) so the two reports share visual identity.

Run:
   python visualize_spoofing.py
============================================================
"""

from __future__ import annotations

import os
import sys
import glob
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colorbar as mcolorbar

# Reuse style + helpers from the detection visualizer so both reports look
# like they belong to the same thesis chapter.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from visualize_results import (        # noqa: E402
    apply_base_style,
    render_table,
    draw_card,
    draw_kpi_card,
    _tint,
    C_BLUE, C_PURPLE, C_PINK, C_GREEN, C_ORANGE, C_TEAL, C_YELLOW,
    C_BLUE_BG, C_PURPLE_BG, C_PINK_BG, C_GREEN_BG, C_ORANGE_BG,
    C_HEADER, C_TEXT, C_SUBTEXT, C_BG_ALT, C_BORDER,
    FS_TITLE, FS_SUBTITLE, FS_SECTION, FS_BODY,
    FS_AXIS_LABEL, FS_TICK, FS_NOTE,
    FS_KPI_VALUE, FS_KPI_LABEL,
    M_LEFT, M_RIGHT, M_TOP, M_BOTTOM,
)


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
RESULTS_DIR = "results"
REPORT_XLSX = os.path.join(RESULTS_DIR, "Spoof_Report.xlsx")
IMG_DIR = os.path.join(RESULTS_DIR, "report_images")
SAMPLE_ROOT = os.path.join(RESULTS_DIR, "spoof_samples")

# Semantic colour mapping for anti-spoofing reporting.
C_LIVE  = C_GREEN     # bona-fide
C_SPOOF = C_PINK      # attack
C_FP    = "#E74C3C"   # false accept (security incident) -- vivid red
C_FN    = C_ORANGE    # false reject

# Dark grey text used inside heatmaps (high contrast on pastel cells)
C_DARK_TEXT = "#2C3E50"   # dark slate -- easier to read than pure black

# Pastel-blue sequential colourmap for the confusion-matrix heatmap.
# Goes from near-white to the project blue (C_BLUE).
CMAP_BLUE = LinearSegmentedColormap.from_list(
    "pastel_blue", ["#FFFFFF", "#E8F1FB", "#C7DEF4", "#9CC4EA", C_BLUE, "#3B7AB8"]
)

# Group labels and palette (must match groups produced by test_anti_spoofing.py)
GROUP_COLORS = {
    "user":  C_BLUE,
    "mask":  C_ORANGE,
    "spoof": C_PINK,
}
GROUP_LABELS = {
    "user":  "User (live)",
    "mask":  "Masked (live)",
    "spoof": "Spoof (attack)",
}


# ---------------------------------------------------------------
# 20. Per-video summary table
# ---------------------------------------------------------------
def export_per_video_table(df_video: pd.DataFrame, out_path: str) -> None:
    if df_video.empty:
        print("[WARN] Per Video sheet is empty, skipping table.")
        return
    show = df_video[[
        "video", "group", "label", "total_frames",
        "detection_rate(%)", "mean_live_score",
        "frac_pred_live_default(%)", "mean_latency(ms)",
    ]].copy()
    show.columns = [
        "Video", "Group", "Label", "Frames",
        "Detect Rate (%)", "Mean Live Score",
        "% Pred Live", "Mean Latency (ms)",
    ]
    # Stringify numeric cells so render_table can compute column widths.
    show["Frames"]            = show["Frames"].apply(lambda v: f"{int(v):,}")
    show["Detect Rate (%)"]   = show["Detect Rate (%)"].apply(lambda v: f"{float(v):.2f}")
    show["Mean Live Score"]   = show["Mean Live Score"].apply(lambda v: f"{float(v):.3f}")
    show["% Pred Live"]       = show["% Pred Live"].apply(lambda v: f"{float(v):.2f}")
    show["Mean Latency (ms)"] = show["Mean Latency (ms)"].apply(lambda v: f"{float(v):.2f}")

    render_table(
        show, "Table 1 - Per Video Anti-Spoofing Results", out_path,
        footnote="Mean Live Score is the average softmax probability assigned to the "
                 "'live' class by MiniFASNet. '% Pred Live' uses the default "
                 "threshold from app.py.",
    )


# ---------------------------------------------------------------
# 21. Per-group summary table
# ---------------------------------------------------------------
def export_group_table(df_group: pd.DataFrame, out_path: str) -> None:
    if df_group.empty:
        print("[WARN] Per Group sheet is empty, skipping group table.")
        return
    show = df_group.copy()
    show.columns = [c.replace("_", " ").title() for c in show.columns]
    # Stringify every cell so render_table never sees raw numbers.
    for c in show.columns:
        if pd.api.types.is_float_dtype(show[c]):
            show[c] = show[c].apply(lambda v: f"{float(v):.2f}")
        elif pd.api.types.is_integer_dtype(show[c]):
            show[c] = show[c].apply(lambda v: f"{int(v):,}")
    render_table(
        show, "Table 2 - Group Averages", out_path,
        footnote="Groups: user_* (live, no mask), mask_* (live, surgical mask), "
                 "spoof_* (phone-replay attacks).",
    )


# ---------------------------------------------------------------
# 22. Metrics table (Default vs EER)
# ---------------------------------------------------------------
def export_metrics_table(df_summary: pd.DataFrame, out_path: str) -> None:
    if df_summary.empty:
        print("[WARN] Summary sheet is empty, skipping metrics table.")
        return
    show = df_summary.copy()
    # Format numeric columns as strings so render_table (which calls len() on cells)
    # never sees floats. Keep NaN as a dash.
    def _fmt_thr(v):
        return "-" if pd.isna(v) else f"{float(v):.2f}"

    def _fmt_pct(v):
        return "-" if pd.isna(v) else f"{float(v):.2f}"

    def _fmt_int(v):
        return "-" if pd.isna(v) else f"{int(v):,}"

    if "threshold" in show.columns:
        show["threshold"] = show["threshold"].apply(_fmt_thr)
    for col in ("APCER(%)", "BPCER(%)", "ACER(%)"):
        if col in show.columns:
            show[col] = show[col].apply(_fmt_pct)
    for col in ("TP", "FN", "FP", "TN"):
        if col in show.columns:
            show[col] = show[col].apply(_fmt_int)

    show.columns = [c.replace("(%)", " (%)").replace("_", " ").title() for c in show.columns]
    render_table(
        show, "Table 3 - APCER / BPCER / ACER per ISO/IEC 30107-3", out_path,
        footnote="APCER = attack presentations classified as live (lower = safer). "
                 "BPCER = bona-fide rejected (lower = more usable). "
                 "ACER = (APCER + BPCER) / 2. EER row uses the threshold where APCER ~= BPCER.",
    )


# ---------------------------------------------------------------
# 23. ROC curve (threshold sweep)
# ---------------------------------------------------------------
def export_roc_curve(df_roc: pd.DataFrame, df_summary: pd.DataFrame, out_path: str) -> None:
    if df_roc.empty:
        print("[WARN] ROC sheet is empty.")
        return
    apply_base_style()
    fig = plt.figure(figsize=(11, 6.2), dpi=150)
    fig.patch.set_facecolor("white")

    # Title band
    fig.text(0.5, 1 - M_TOP / 2, "Figure 1 - Threshold Sweep (APCER vs BPCER)",
             ha="center", va="center", fontsize=FS_TITLE, fontweight="bold", color=C_TEXT)
    fig.text(0.5, 1 - M_TOP / 2 - 0.04,
             "Anti-spoofing trade-off as the decision threshold moves from 0.05 to 0.95",
             ha="center", va="center", fontsize=FS_SUBTITLE, color=C_SUBTEXT)

    ax = fig.add_axes([M_LEFT + 0.04, M_BOTTOM + 0.10,
                       1 - M_LEFT - M_RIGHT - 0.08,
                       1 - M_TOP - M_BOTTOM - 0.20])

    thr = df_roc["threshold"]
    apcer = df_roc["APCER(%)"]
    bpcer = df_roc["BPCER(%)"]
    acer = df_roc["ACER(%)"]

    # Shaded recommended operating band (thr in [0.85, 0.95]) -- this is
    # where ACER tends to be lowest for our data and where APCER drops fast.
    ax.axvspan(0.85, 0.95, facecolor=_tint(C_GREEN, 0.18),
               edgecolor="none", zorder=0,
               label="Recommended band (0.85-0.95)")

    ax.plot(thr, apcer, color=C_FP,    lw=2.2, label="APCER (attack -> live)")
    ax.plot(thr, bpcer, color=C_FN,    lw=2.2, label="BPCER (live -> rejected)")
    ax.plot(thr, acer,  color=C_PURPLE, lw=2.0, linestyle="--", label="ACER (mean)")

    # Mark default + EER points if present in summary
    if not df_summary.empty:
        default_row = df_summary[df_summary["operating_point"].str.contains("default")]
        eer_row = df_summary[df_summary["operating_point"] == "EER"]
        if not eer_row.empty:
            t = float(eer_row["threshold"].iloc[0])
            a = float(eer_row["APCER(%)"].iloc[0])
            ax.axvline(t, color=C_PURPLE, lw=1.2, alpha=0.55, linestyle=":")
            ax.scatter([t], [a], color=C_PURPLE, s=80, zorder=5,
                       edgecolor="white", linewidth=1.5)
            ax.annotate(f"EER\n(thr={t:.2f}, {a:.1f}%)",
                        xy=(t, a), xytext=(t + 0.05, a + 6),
                        fontsize=FS_NOTE, color=C_PURPLE, fontweight="bold")
        if not default_row.empty and not pd.isna(default_row["threshold"].iloc[0]):
            t = float(default_row["threshold"].iloc[0])
            ax.axvline(t, color=C_TEXT, lw=1.0, alpha=0.4, linestyle="-.")

    ax.set_xlabel("Threshold on live-score", fontsize=FS_AXIS_LABEL, color=C_TEXT)
    ax.set_ylabel("Error rate (%)", fontsize=FS_AXIS_LABEL, color=C_TEXT)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0, max(105, float(np.nanmax([apcer.max(), bpcer.max()])) + 5))
    ax.grid(alpha=0.25, color=C_BORDER)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color(C_BORDER)
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(labelsize=FS_TICK, colors=C_SUBTEXT)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12),
                    ncol=4, frameon=False, fontsize=FS_NOTE)
    for txt in leg.get_texts():
        txt.set_color(C_TEXT)

    fig.text(0.5, M_BOTTOM / 2,
             "Lower curves are better. The green band marks thresholds where ACER is minimised "
             "(recommended operating range).",
             ha="center", va="center", fontsize=FS_NOTE, color=C_SUBTEXT, style="italic")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------
# 24. Score distribution histogram
# ---------------------------------------------------------------
def export_score_distribution(df_log: pd.DataFrame, df_summary: pd.DataFrame,
                              out_path: str) -> None:
    if df_log.empty:
        print("[WARN] Frame Log sheet is empty.")
        return
    apply_base_style()
    fig = plt.figure(figsize=(11, 5.8), dpi=150)
    fig.patch.set_facecolor("white")

    fig.text(0.5, 1 - M_TOP / 2, "Figure 2 - Distribution of MiniFASNet Live Scores",
             ha="center", va="center", fontsize=FS_TITLE, fontweight="bold", color=C_TEXT)
    fig.text(0.5, 1 - M_TOP / 2 - 0.045,
             "Per-frame live-class probability, separated by ground-truth presentation type",
             ha="center", va="center", fontsize=FS_SUBTITLE, color=C_SUBTEXT)

    ax = fig.add_axes([M_LEFT + 0.04, M_BOTTOM + 0.10,
                       1 - M_LEFT - M_RIGHT - 0.08,
                       1 - M_TOP - M_BOTTOM - 0.22])

    bins = np.linspace(0, 1, 31)
    for grp, color in [("live", C_LIVE), ("spoof", C_SPOOF)]:
        sub = df_log[df_log["label"] == grp]["live_score"]
        if not sub.empty:
            ax.hist(sub, bins=bins, color=color, alpha=0.55,
                    edgecolor="white", linewidth=0.6,
                    label=f"{grp.title()} (n={len(sub):,})")

    # Mark EER threshold
    if not df_summary.empty:
        eer_row = df_summary[df_summary["operating_point"] == "EER"]
        if not eer_row.empty:
            t = float(eer_row["threshold"].iloc[0])
            ax.axvline(t, color=C_PURPLE, lw=2.0, linestyle="--",
                       label=f"EER threshold = {t:.2f}")

    ax.set_xlabel("Live-score (MiniFASNet softmax)", fontsize=FS_AXIS_LABEL, color=C_TEXT)
    ax.set_ylabel("Frame count", fontsize=FS_AXIS_LABEL, color=C_TEXT)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25, color=C_BORDER, axis="y")
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color(C_BORDER)
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(labelsize=FS_TICK, colors=C_SUBTEXT)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10),
                    ncol=3, frameon=False, fontsize=FS_BODY)
    for txt in leg.get_texts():
        txt.set_color(C_TEXT)

    fig.text(0.5, M_BOTTOM / 2,
             "A clean separation between live (green) and spoof (pink) means MiniFASNet "
             "discriminates the two presentations well.",
             ha="center", va="center", fontsize=FS_NOTE, color=C_SUBTEXT, style="italic")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------
# 25. Confusion matrices (Default + EER) -- pastel heatmap style
# ---------------------------------------------------------------
def _draw_confusion_heatmap(ax, cax, df_cm: pd.DataFrame, title: str,
                            title_color: str, vmax: float) -> None:
    """Render a 2x2 confusion matrix as a pastel-blue sequential heatmap with
    a vertical colourbar. Cells with high counts get deeper blue; numbers and
    labels use dark slate so they read clearly on every shade.

    Layout matches `image.jpg`:
        rows = Actual (Negative=LIVE, Positive=SPOOF)
        cols = Predicted (Negative=LIVE, Positive=SPOOF)
        Top-left & bottom-right are 'correct' (deepest blue typically),
        Top-right & bottom-left are 'errors' (paler).
    """
    # df_cm has columns: actual, pred_live, pred_spoof. Row 0 = live, row 1 = spoof.
    mat = df_cm[["pred_live", "pred_spoof"]].to_numpy(dtype=float)
    actual = df_cm["actual"].tolist()
    cell_kinds = [["TP", "FN"], ["FP", "TN"]]

    # Turn off any background grid -- the heatmap cells are the only visual.
    ax.grid(False)
    ax.set_axisbelow(False)

    im = ax.imshow(mat, cmap=CMAP_BLUE, vmin=0, vmax=vmax, aspect="equal")

    # ONE number per cell, centered -- exactly like the seaborn reference.
    # No TP/FN chips, no percentages, no extra dividing lines.
    for r in range(2):
        for c in range(2):
            val = int(mat[r, c])
            intensity = mat[r, c] / vmax if vmax > 0 else 0
            txt_color = "white" if intensity > 0.55 else C_DARK_TEXT
            ax.text(c, r, f"{val:,}",
                    ha="center", va="center",
                    fontsize=FS_KPI_VALUE, fontweight="bold",
                    color=txt_color)

    # Single-line tick labels -- LIVE/SPOOF directly, no parenthetical clutter.
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["LIVE", "SPOOF"],
                       fontsize=FS_BODY, color=C_DARK_TEXT)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([actual[0].upper(), actual[1].upper()],
                       fontsize=FS_BODY, color=C_DARK_TEXT,
                       rotation=90, va="center")
    ax.set_xlabel("Predicted Label", fontsize=FS_BODY, color=C_DARK_TEXT, labelpad=10)
    ax.set_ylabel("Actual Label", fontsize=FS_BODY, color=C_DARK_TEXT, labelpad=10)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=FS_SECTION, fontweight="bold",
                 color=title_color, pad=14)

    # Colourbar on the right
    cbar = plt.colorbar(im, cax=cax)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=FS_TICK, colors=C_DARK_TEXT, length=0)
    cbar.ax.set_ylabel("Frame count", fontsize=FS_NOTE, color=C_DARK_TEXT,
                       rotation=270, labelpad=14)


def export_confusion_matrices(df_cm_default: pd.DataFrame,
                              df_cm_eer: pd.DataFrame,
                              df_summary: pd.DataFrame,
                              out_path: str) -> None:
    if df_cm_default.empty or df_cm_eer.empty:
        print("[WARN] Confusion matrix data empty.")
        return
    apply_base_style()
    fig = plt.figure(figsize=(14, 6.8), dpi=150)
    fig.patch.set_facecolor("white")

    fig.text(0.5, 1 - M_TOP / 2,
             "Figure 3 - Confusion Matrices at Two Operating Points",
             ha="center", va="center", fontsize=FS_TITLE,
             fontweight="bold", color=C_TEXT)
    fig.text(0.5, 1 - M_TOP / 2 - 0.04,
             "Counts of frame-level decisions: default threshold from app.py vs. EER threshold",
             ha="center", va="center", fontsize=FS_SUBTITLE, color=C_SUBTEXT)

    # Shared vmax across both matrices for an honest visual comparison
    mat_l = df_cm_default[["pred_live", "pred_spoof"]].to_numpy(dtype=float)
    mat_r = df_cm_eer[["pred_live", "pred_spoof"]].to_numpy(dtype=float)
    vmax = float(max(mat_l.max(), mat_r.max()))

    # Two heatmaps side by side, each with its own colourbar.
    # Lift the plot region a bit so the metric strip below has room.
    ax_l  = fig.add_axes([0.10, 0.24, 0.30, 0.56])
    cax_l = fig.add_axes([0.405, 0.24, 0.012, 0.56])
    ax_r  = fig.add_axes([0.56, 0.24, 0.30, 0.56])
    cax_r = fig.add_axes([0.865, 0.24, 0.012, 0.56])

    def _stats(row_label: str) -> str:
        if df_summary.empty:
            return ""
        r = df_summary[df_summary["operating_point"].str.contains(row_label)]
        if r.empty:
            return ""
        return (f"APCER {float(r['APCER(%)'].iloc[0]):.2f}%   "
                f"BPCER {float(r['BPCER(%)'].iloc[0]):.2f}%   "
                f"ACER {float(r['ACER(%)'].iloc[0]):.2f}%")

    _draw_confusion_heatmap(ax_l, cax_l, df_cm_default,
                            "Default threshold (app.py)", C_HEADER, vmax)
    _draw_confusion_heatmap(ax_r, cax_r, df_cm_eer,
                            "At EER threshold", C_PURPLE, vmax)

    # Metric strip under each matrix -- below the x-axis label, not over it.
    fig.text(0.25, 0.05, _stats("default"),
             ha="center", fontsize=FS_BODY, color=C_DARK_TEXT, fontweight="bold")
    fig.text(0.71, 0.05, _stats("EER"),
             ha="center", fontsize=FS_BODY, color=C_DARK_TEXT, fontweight="bold")

    fig.text(0.5, M_BOTTOM / 2,
             "FP (false accept) is the most dangerous error: a phone replay was treated as a real face. "
             "Deeper blue = higher frame count.",
             ha="center", va="center", fontsize=FS_NOTE,
             color=C_SUBTEXT, style="italic")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------
# 25b. SINGLE confusion matrix (for the final / production report
#      after the threshold has been frozen on a dev set)
# ---------------------------------------------------------------
def export_single_confusion_matrix(df_cm: pd.DataFrame,
                                   df_summary: pd.DataFrame,
                                   row_label: str,
                                   out_path: str,
                                   title_suffix: str = "") -> None:
    """Render ONE confusion matrix for the final report.

    Use this once a threshold has been chosen on a dev / tuning set and you
    are reporting the held-out test result.

    Parameters
    ----------
    df_cm        : DataFrame from 'Confusion @ Default' or 'Confusion @ EER'.
    df_summary   : Summary sheet (for the APCER/BPCER/ACER strip).
    row_label    : substring used to pull the matching row from Summary --
                   typically 'default' or 'EER'.
    title_suffix : optional text appended to the title, e.g.
                   '(test set, threshold = 0.94)'.
    """
    if df_cm.empty:
        print("[WARN] Single confusion matrix data empty.")
        return
    apply_base_style()
    fig = plt.figure(figsize=(9, 7.2), dpi=150)
    fig.patch.set_facecolor("white")

    title = "Figure 3 - Confusion Matrix"
    if title_suffix:
        title = f"{title}  {title_suffix}"
    fig.text(0.5, 1 - M_TOP / 2,
             title,
             ha="center", va="center", fontsize=FS_TITLE,
             fontweight="bold", color=C_TEXT)
    fig.text(0.5, 1 - M_TOP / 2 - 0.045,
             "Frame-level decisions at the chosen operating threshold",
             ha="center", va="center", fontsize=FS_SUBTITLE, color=C_SUBTEXT)

    mat = df_cm[["pred_live", "pred_spoof"]].to_numpy(dtype=float)
    vmax = float(mat.max()) if mat.size else 1.0

    # Centered heatmap + colourbar.
    ax  = fig.add_axes([0.20, 0.20, 0.50, 0.56])
    cax = fig.add_axes([0.715, 0.20, 0.018, 0.56])
    _draw_confusion_heatmap(ax, cax, df_cm, "", C_HEADER, vmax)
    ax.set_title("")  # remove the (blank) per-axes title slot

    # Stats strip below the matrix.
    if not df_summary.empty:
        r = df_summary[df_summary["operating_point"].str.contains(row_label)]
        if not r.empty:
            stats = (f"APCER {float(r['APCER(%)'].iloc[0]):.2f}%   "
                     f"BPCER {float(r['BPCER(%)'].iloc[0]):.2f}%   "
                     f"ACER {float(r['ACER(%)'].iloc[0]):.2f}%")
            fig.text(0.46, 0.10, stats,
                     ha="center", fontsize=FS_BODY + 1,
                     color=C_DARK_TEXT, fontweight="bold")

    fig.text(0.5, M_BOTTOM / 2,
             "Top-left and bottom-right are correct decisions; "
             "top-right = FN (live rejected), bottom-left = FP (attack accepted).",
             ha="center", va="center", fontsize=FS_NOTE,
             color=C_SUBTEXT, style="italic")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------
# 26. One-page infographic
# ---------------------------------------------------------------
def export_infographic(df_video: pd.DataFrame, df_group: pd.DataFrame,
                       df_roc: pd.DataFrame, df_summary: pd.DataFrame,
                       df_log: pd.DataFrame, out_path: str) -> None:
    apply_base_style()
    fig = plt.figure(figsize=(13, 16.5), dpi=150)
    fig.patch.set_facecolor("white")

    # Vertical layout (figure-fraction)
    HEADER_H = 0.060
    GAP = 0.020
    KPI_H = 0.070
    PV_H = 0.180   # per-video bar
    ROC_H = 0.180
    HIST_H = 0.180
    CONC_H = 0.180   # taller -- holds 4 bullets now

    # Position bands from top to bottom
    y_top = 1 - M_TOP
    header_bot = y_top - HEADER_H
    kpi_bot = header_bot - GAP - KPI_H
    pv_bot  = kpi_bot - GAP - PV_H
    roc_bot = pv_bot - GAP - ROC_H
    hist_bot = roc_bot - GAP - HIST_H
    conc_bot = hist_bot - GAP - CONC_H

    # Title
    fig.text(0.5, header_bot + HEADER_H * 0.55,
             "Anti-Spoofing Evaluation Report",
             ha="center", va="center", fontsize=FS_TITLE + 2,
             fontweight="bold", color=C_TEXT)
    fig.text(0.5, header_bot + HEADER_H * 0.15,
             "Smart Locker System  -  MiniFASNet (Silent-Face)  -  ISO/IEC 30107-3",
             ha="center", va="center", fontsize=FS_SUBTITLE, color=C_SUBTEXT)

    # ===== KPI row =====
    total_frames = int(df_log.shape[0]) if not df_log.empty else 0
    if not df_summary.empty:
        eer_row = df_summary[df_summary["operating_point"] == "EER"]
        def_row = df_summary[df_summary["operating_point"].str.contains("default")]
        eer_thr = float(eer_row["threshold"].iloc[0]) if not eer_row.empty else float("nan")
        eer_acer = float(eer_row["ACER(%)"].iloc[0]) if not eer_row.empty else float("nan")
        apcer_d = float(def_row["APCER(%)"].iloc[0]) if not def_row.empty else float("nan")
        bpcer_d = float(def_row["BPCER(%)"].iloc[0]) if not def_row.empty else float("nan")
        acer_d  = float(def_row["ACER(%)"].iloc[0])  if not def_row.empty else float("nan")
    else:
        eer_thr = eer_acer = apcer_d = bpcer_d = acer_d = float("nan")

    ax_kpi = fig.add_axes([M_LEFT, kpi_bot, 1 - M_LEFT - M_RIGHT, KPI_H])
    ax_kpi.set_xlim(0, 1); ax_kpi.set_ylim(0, 1); ax_kpi.axis("off")
    cards = [
        (f"{total_frames:,}",     "Total Frames",      C_BLUE),
        (f"{apcer_d:.2f}%",       "APCER (default)",   C_FP),
        (f"{bpcer_d:.2f}%",       "BPCER (default)",   C_FN),
        (f"{acer_d:.2f}%",        "ACER (default)",    C_PURPLE),
        (f"{eer_acer:.2f}%",      f"EER  (thr={eer_thr:.2f})", C_GREEN),
    ]
    n = len(cards)
    gap = 0.018
    cw = (1 - gap * (n - 1)) / n
    for i, (val, lab, col) in enumerate(cards):
        x = i * (cw + gap)
        draw_kpi_card(ax_kpi, x, 0.0, cw, 1.0, val, lab, stroke=col)

    # ===== Card: Per-video live-score bar =====
    ax_pv_bg = fig.add_axes([M_LEFT, pv_bot, 1 - M_LEFT - M_RIGHT, PV_H])
    ax_pv_bg.set_xlim(0, 1); ax_pv_bg.set_ylim(0, 1); ax_pv_bg.axis("off")
    draw_card(ax_pv_bg, 0.0, 0.0, 1.0, 1.0, stroke=C_BLUE, fill="white",
              radius=0.025, lw=1.4)
    ax_pv_bg.text(0.03, 0.90, "Mean Live-Score per Video",
                  fontsize=FS_SECTION, fontweight="bold", color=C_TEXT,
                  transform=ax_pv_bg.transAxes)
    # Legend chips on top-right
    legend_items = [(C_BLUE, "User (live)"), (C_ORANGE, "Masked (live)"), (C_PINK, "Spoof (attack)")]
    lx = 0.97
    for col, lab in reversed(legend_items):
        ax_pv_bg.add_patch(plt.Rectangle((lx - 0.013, 0.91), 0.013, 0.05,
                                         color=col, transform=ax_pv_bg.transAxes))
        ax_pv_bg.text(lx - 0.018, 0.93, lab, ha="right", va="center",
                      fontsize=FS_NOTE, color=C_TEXT,
                      transform=ax_pv_bg.transAxes)
        lx -= 0.13

    pad_l, pad_r = 0.075, 0.035
    chart_y_b, chart_y_t = 0.30, 0.72
    bx_x = M_LEFT + pad_l * (1 - M_LEFT - M_RIGHT)
    bx_w = (1 - pad_l - pad_r) * (1 - M_LEFT - M_RIGHT)
    bx_y = pv_bot + chart_y_b * PV_H
    bx_h = (chart_y_t - chart_y_b) * PV_H
    ax_pv = fig.add_axes([bx_x, bx_y, bx_w, bx_h])

    if not df_video.empty:
        df_s = df_video.sort_values(["group", "video"]).reset_index(drop=True)
        colors = [GROUP_COLORS.get(g, C_HEADER) for g in df_s["group"]]
        bars = ax_pv.bar(range(len(df_s)),
                         df_s["mean_live_score"].astype(float),
                         color=colors, edgecolor="white", linewidth=1.2)
        for i, (b, v) in enumerate(zip(bars, df_s["mean_live_score"].astype(float))):
            ax_pv.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03,
                       f"{v:.2f}", ha="center", fontsize=FS_NOTE,
                       color=C_TEXT, fontweight="bold")
        ax_pv.set_xticks(range(len(df_s)))
        ax_pv.set_xticklabels(df_s["video"], rotation=30, ha="right",
                              fontsize=FS_TICK, color=C_TEXT)
        ax_pv.set_ylim(0, 1.18)
        ax_pv.set_ylabel("Mean live-score", fontsize=FS_AXIS_LABEL, color=C_TEXT)
        if not df_summary.empty:
            eer_row = df_summary[df_summary["operating_point"] == "EER"]
            if not eer_row.empty:
                t = float(eer_row["threshold"].iloc[0])
                ax_pv.axhline(t, color=C_PURPLE, lw=1.2,
                              linestyle="--", alpha=0.7)
                ax_pv.text(len(df_s) - 0.5, t + 0.02, f"EER thr {t:.2f}",
                           color=C_PURPLE, fontsize=FS_NOTE, ha="right",
                           style="italic")
    ax_pv.tick_params(axis="y", labelsize=FS_TICK, colors=C_SUBTEXT)
    ax_pv.grid(axis="y", alpha=0.25, color=C_BORDER)
    ax_pv.set_axisbelow(True)
    ax_pv.set_facecolor("none")
    for sp in ["top", "right"]:
        ax_pv.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax_pv.spines[sp].set_color(C_BORDER)
        ax_pv.spines[sp].set_linewidth(0.8)

    # ===== Card: ROC curve =====
    ax_roc_bg = fig.add_axes([M_LEFT, roc_bot, 1 - M_LEFT - M_RIGHT, ROC_H])
    ax_roc_bg.set_xlim(0, 1); ax_roc_bg.set_ylim(0, 1); ax_roc_bg.axis("off")
    draw_card(ax_roc_bg, 0.0, 0.0, 1.0, 1.0, stroke=C_PURPLE, fill="white",
              radius=0.025, lw=1.4)
    ax_roc_bg.text(0.03, 0.90, "APCER vs BPCER Threshold Sweep",
                   fontsize=FS_SECTION, fontweight="bold", color=C_TEXT,
                   transform=ax_roc_bg.transAxes)

    bx_y2 = roc_bot + chart_y_b * ROC_H
    bx_h2 = (chart_y_t - chart_y_b) * ROC_H
    ax_roc = fig.add_axes([bx_x, bx_y2, bx_w, bx_h2])
    if not df_roc.empty:
        ax_roc.plot(df_roc["threshold"], df_roc["APCER(%)"], color=C_FP, lw=2,
                    label="APCER")
        ax_roc.plot(df_roc["threshold"], df_roc["BPCER(%)"], color=C_FN, lw=2,
                    label="BPCER")
        ax_roc.plot(df_roc["threshold"], df_roc["ACER(%)"], color=C_PURPLE,
                    lw=1.6, linestyle="--", label="ACER")
        if not df_summary.empty:
            eer_row = df_summary[df_summary["operating_point"] == "EER"]
            if not eer_row.empty:
                t = float(eer_row["threshold"].iloc[0])
                ax_roc.axvline(t, color=C_PURPLE, lw=1.2, alpha=0.5, linestyle=":")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_xlabel("Threshold", fontsize=FS_AXIS_LABEL, color=C_TEXT)
    ax_roc.set_ylabel("Error rate (%)", fontsize=FS_AXIS_LABEL, color=C_TEXT)
    ax_roc.tick_params(labelsize=FS_TICK, colors=C_SUBTEXT)
    ax_roc.grid(alpha=0.25, color=C_BORDER)
    ax_roc.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax_roc.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax_roc.spines[sp].set_color(C_BORDER)
        ax_roc.spines[sp].set_linewidth(0.8)
    leg = ax_roc.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18),
                        ncol=3, frameon=False, fontsize=FS_NOTE)
    for txt in leg.get_texts():
        txt.set_color(C_TEXT)

    # ===== Card: Score distribution =====
    ax_h_bg = fig.add_axes([M_LEFT, hist_bot, 1 - M_LEFT - M_RIGHT, HIST_H])
    ax_h_bg.set_xlim(0, 1); ax_h_bg.set_ylim(0, 1); ax_h_bg.axis("off")
    draw_card(ax_h_bg, 0.0, 0.0, 1.0, 1.0, stroke=C_GREEN, fill="white",
              radius=0.025, lw=1.4)
    ax_h_bg.text(0.03, 0.90, "Live-Score Distribution (Live vs Spoof)",
                 fontsize=FS_SECTION, fontweight="bold", color=C_TEXT,
                 transform=ax_h_bg.transAxes)

    bx_y3 = hist_bot + chart_y_b * HIST_H
    bx_h3 = (chart_y_t - chart_y_b) * HIST_H
    ax_h = fig.add_axes([bx_x, bx_y3, bx_w, bx_h3])
    if not df_log.empty:
        bins = np.linspace(0, 1, 31)
        for grp, color in [("live", C_LIVE), ("spoof", C_SPOOF)]:
            sub = df_log[df_log["label"] == grp]["live_score"]
            if not sub.empty:
                ax_h.hist(sub, bins=bins, color=color, alpha=0.55,
                          edgecolor="white", linewidth=0.6,
                          label=f"{grp} (n={len(sub):,})")
    ax_h.set_xlim(0, 1)
    ax_h.set_xlabel("Live-score", fontsize=FS_AXIS_LABEL, color=C_TEXT)
    ax_h.set_ylabel("Frames", fontsize=FS_AXIS_LABEL, color=C_TEXT)
    ax_h.tick_params(labelsize=FS_TICK, colors=C_SUBTEXT)
    ax_h.grid(alpha=0.25, color=C_BORDER, axis="y")
    ax_h.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax_h.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax_h.spines[sp].set_color(C_BORDER)
        ax_h.spines[sp].set_linewidth(0.8)
    leg = ax_h.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18),
                      ncol=2, frameon=False, fontsize=FS_NOTE)
    for txt in leg.get_texts():
        txt.set_color(C_TEXT)

    # ===== Findings + Recommendations card =====
    ax_c_bg = fig.add_axes([M_LEFT, conc_bot, 1 - M_LEFT - M_RIGHT, CONC_H])
    ax_c_bg.set_xlim(0, 1); ax_c_bg.set_ylim(0, 1); ax_c_bg.axis("off")
    draw_card(ax_c_bg, 0.0, 0.0, 1.0, 1.0, stroke=C_GREEN, fill=_tint(C_GREEN, 0.08),
              radius=0.025, lw=1.4)
    ax_c_bg.text(0.5, 0.92, "Findings & Recommendations", ha="center", va="center",
                 fontsize=FS_SECTION + 1, fontweight="bold",
                 color=C_GREEN, transform=ax_c_bg.transAxes)

    # ---- Identify the worst spoof video (the one driving APCER up) ----
    worst_spoof = None
    if not df_video.empty:
        sp = df_video[df_video["label"] == "spoof"].copy()
        if not sp.empty and "frac_pred_live_default(%)" in sp.columns:
            sp = sp.sort_values("frac_pred_live_default(%)", ascending=False)
            top = sp.iloc[0]
            worst_spoof = (str(top["video"]),
                           float(top["frac_pred_live_default(%)"]))

    spoof_mean = (df_video[df_video["label"] == "spoof"]["mean_live_score"].mean()
                  if not df_video.empty else float("nan"))
    live_mean  = (df_video[df_video["label"] == "live"]["mean_live_score"].mean()
                  if not df_video.empty else float("nan"))

    bullets: List[Tuple[str, str, str]] = []   # (icon-colour, prefix, body)

    # 1. Headline finding -- whether default threshold is acceptable
    if not pd.isna(acer_d):
        if acer_d < 3.5:
            bullets.append((C_GREEN, "PASS:",
                            f"Default threshold delivers ACER {acer_d:.2f}% "
                            f"(APCER {apcer_d:.2f}%, BPCER {bpcer_d:.2f}%) -- below 3.5% target."))
        else:
            extra = (f" Mainly driven by {worst_spoof[0]} ({worst_spoof[1]:.1f}% of frames "
                     f"misclassified as live)." if worst_spoof else "")
            bullets.append((C_FP, "RISK:",
                            f"Default threshold yields APCER {apcer_d:.2f}% -- well above 3.5% target."
                            f"{extra}"))

    # 2. Threshold recommendation (always show)
    if not pd.isna(eer_thr):
        bullets.append((C_PURPLE, "TUNE:",
                        f"Raise decision threshold to ~{eer_thr:.2f} (EER point, ACER {eer_acer:.2f}%) "
                        f"to cut APCER roughly in half with only a small BPCER increase."))

    # 3. Data + design recommendations
    data_bits = []
    n_spoof = int((df_video["label"] == "spoof").sum()) if not df_video.empty else 0
    n_live  = int((df_video["label"] == "live").sum())  if not df_video.empty else 0
    if n_spoof < 6:
        data_bits.append(f"Add more spoof videos (now {n_spoof}; target 12-15 covering different phones, "
                         "distances, and a print-attack variant).")
    if not (pd.isna(spoof_mean) or pd.isna(live_mean)) and abs(live_mean - spoof_mean) < 0.4:
        data_bits.append("Score gap between live and spoof is modest -- include harder negatives.")
    if data_bits:
        bullets.append((C_BLUE, "DATA:", " ".join(data_bits)))

    # 4. Production hardening -- multi-frame voting
    bullets.append((C_ORANGE, "DEPLOY:",
                    "Add multi-frame voting (>=80% of frames in a 1-second window must be LIVE) "
                    "before unlocking -- rejects intermittent attacks without raising BPCER."))

    # ---- Render bullets ----
    # Hard-wrap any over-long body so it does not collide with the next bullet
    # or run off the card. ~115 chars per line works for FS_BODY at this width.
    def _wrap(s: str, n: int = 115) -> str:
        words = s.split()
        lines, cur = [], ""
        for w in words:
            if len(cur) + 1 + len(w) > n:
                lines.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            lines.append(cur)
        return "\n".join(lines)

    n_b = min(4, len(bullets))
    if n_b == 0:
        n_b = 1
        bullets = [(C_GREEN, "OK:", "No findings.")]
    top_y, bot_y = 0.78, 0.10
    step = (top_y - bot_y) / max(n_b - 1, 1) if n_b > 1 else 0
    for i, (icon_col, prefix, body) in enumerate(bullets[:n_b]):
        y = top_y - i * step if n_b > 1 else (top_y + bot_y) / 2
        ax_c_bg.scatter(0.035, y, s=85, color=icon_col, transform=ax_c_bg.transAxes,
                        zorder=3, edgecolor="white", linewidth=1.2)
        ax_c_bg.text(0.062, y, prefix, ha="left", va="center",
                     fontsize=FS_BODY, fontweight="bold",
                     color=icon_col, transform=ax_c_bg.transAxes)
        ax_c_bg.text(0.115, y, _wrap(body), ha="left", va="center",
                     fontsize=FS_BODY, color=C_TEXT,
                     transform=ax_c_bg.transAxes)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------
# 27. Sample frames per group  (3 rows x 5 cols)
# ---------------------------------------------------------------
def _pick_group_samples(sample_root: str, video_names: List[str], n: int = 5
                        ) -> List[Tuple[str, str, str]]:
    """Return up to `n` (video_name, role, image_path) tuples, prioritising:
       - 3 best_*  (correct, highly confident)
       - 2 worst_* (most-damaging error)
    Distributes across videos so we don't pick all 5 from the same video.
    """
    bests, worsts = [], []
    for v in video_names:
        d = os.path.join(sample_root, v)
        for p in sorted(glob.glob(os.path.join(d, "best_*.jpg"))):
            bests.append((v, "BEST", p))
        for p in sorted(glob.glob(os.path.join(d, "worst_*.jpg"))):
            worsts.append((v, "WORST", p))

    # Round-robin across videos for diversity
    def _diversify(rows, take):
        per_video = {}
        for v, role, p in rows:
            per_video.setdefault(v, []).append((v, role, p))
        out = []
        i = 0
        videos = sorted(per_video.keys())
        while len(out) < take and any(per_video[v] for v in videos):
            v = videos[i % len(videos)]
            if per_video[v]:
                out.append(per_video[v].pop(0))
            i += 1
        return out

    chosen = _diversify(bests, 3) + _diversify(worsts, 2)
    return chosen[:n]


def export_sample_frames(sample_root: str, df_video: pd.DataFrame, out_path: str) -> None:
    if df_video.empty:
        print("[WARN] No per-video data, skipping sample frames figure.")
        return
    if not os.path.isdir(sample_root):
        print(f"[WARN] No sample_root dir at {sample_root}; skipping sample frames figure.")
        return

    apply_base_style()
    fig = plt.figure(figsize=(16, 11), dpi=150)
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.965, "Figure 4 - Sample Frames per Group",
             ha="center", va="center", fontsize=FS_TITLE,
             fontweight="bold", color=C_TEXT)
    fig.text(0.5, 0.928,
             "3 best + 2 worst frames per group  -  border colour encodes the confusion cell",
             ha="center", va="center", fontsize=FS_SUBTITLE, color=C_SUBTEXT)

    groups_present = [g for g in ("user", "mask", "spoof")
                      if g in df_video["group"].unique()]
    n_rows = len(groups_present)
    n_cols = 5
    if n_rows == 0:
        print("[WARN] no usable groups for sample frames")
        return

    LEFT_LABEL_W = 0.07
    TOP_MARGIN = 0.875
    BOT_MARGIN = 0.06
    grid_h = TOP_MARGIN - BOT_MARGIN
    row_h = grid_h / n_rows

    for ri, grp in enumerate(groups_present):
        videos = (df_video[df_video["group"] == grp]["video"]
                  .str.replace(".mp4", "", regex=False).tolist())
        samples = _pick_group_samples(sample_root, videos, n=n_cols)

        # Row label band
        row_top = TOP_MARGIN - ri * row_h
        row_bot = row_top - row_h
        ax_lab = fig.add_axes([M_LEFT, row_bot + 0.06 * row_h,
                               LEFT_LABEL_W - M_LEFT - 0.005,
                               row_h - 0.12 * row_h])
        ax_lab.axis("off")
        col = GROUP_COLORS.get(grp, C_HEADER)
        ax_lab.add_patch(FancyBboxPatch(
            (0.0, 0.0), 1.0, 1.0,
            boxstyle="round,pad=0.0,rounding_size=0.08",
            facecolor=_tint(col, 0.18), edgecolor=col,
            linewidth=1.3, transform=ax_lab.transAxes))
        ax_lab.text(0.5, 0.5, GROUP_LABELS.get(grp, grp.title()),
                    rotation=90, ha="center", va="center",
                    fontsize=FS_BODY, fontweight="bold", color=col,
                    transform=ax_lab.transAxes)

        # Image cells
        cell_w = (1 - LEFT_LABEL_W - M_RIGHT) / n_cols
        for ci in range(n_cols):
            ax_im = fig.add_axes([LEFT_LABEL_W + ci * cell_w + 0.005,
                                  row_bot + 0.06 * row_h,
                                  cell_w - 0.01,
                                  row_h - 0.12 * row_h])
            ax_im.axis("off")
            if ci < len(samples):
                v, role, path = samples[ci]
                if os.path.exists(path):
                    try:
                        img = mpimg.imread(path)
                        ax_im.imshow(img)
                    except Exception as exc:
                        ax_im.text(0.5, 0.5, f"(load err)\n{exc}",
                                   ha="center", va="center", fontsize=FS_NOTE,
                                   color=C_SUBTEXT, transform=ax_im.transAxes)
                # Caption strip (bottom-left)
                strip_color = C_GREEN if role == "BEST" else C_PINK
                ax_im.text(0.02, 0.04, f" {v} - {role} ",
                           ha="left", va="bottom",
                           fontsize=FS_NOTE, fontweight="bold",
                           color="white",
                           bbox=dict(boxstyle="round,pad=0.25",
                                     facecolor=strip_color, edgecolor="none"),
                           transform=ax_im.transAxes)
            else:
                ax_im.add_patch(plt.Rectangle((0.05, 0.1), 0.9, 0.8,
                                              facecolor=C_BG_ALT,
                                              edgecolor=C_BORDER,
                                              linewidth=1,
                                              transform=ax_im.transAxes))
                ax_im.text(0.5, 0.5, "no sample",
                           ha="center", va="center",
                           fontsize=FS_NOTE, color=C_SUBTEXT, style="italic",
                           transform=ax_im.transAxes)

    fig.text(0.5, 0.025,
             "Green border = TP (live accepted).  Blue border = TN (spoof rejected).  "
             "Red border = FP (spoof accepted - critical).  Orange border = FN (live rejected).",
             ha="center", va="center", fontsize=FS_NOTE,
             color=C_SUBTEXT, style="italic")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------
# 28. Confusion-cell samples figure  (4 cols x 2 rows = TP, TN, FP, FN)
# ---------------------------------------------------------------
def export_confusion_samples_fig(sample_root: str, out_path: str) -> None:
    cdir = os.path.join(sample_root, "_confusion")
    if not os.path.isdir(cdir):
        print(f"[WARN] No confusion samples dir at {cdir}; skipping figure 28.")
        return

    apply_base_style()
    fig = plt.figure(figsize=(15, 9.2), dpi=150)
    fig.patch.set_facecolor("white")

    # Title goes higher so the chip row doesn't collide with the subtitle.
    fig.text(0.5, 0.975, "Figure 5 - Per-Cell Confusion Samples",
             ha="center", va="center", fontsize=FS_TITLE,
             fontweight="bold", color=C_TEXT)
    fig.text(0.5, 0.945,
             "Two representative frames for each cell of the confusion matrix",
             ha="center", va="center", fontsize=FS_SUBTITLE, color=C_SUBTEXT)

    # Two-line chip labels so they fit inside each column and never overlap.
    cells = [
        ("TP", "True Positive",  "live accepted",                  C_LIVE),
        ("TN", "True Negative",  "spoof rejected",                 C_SPOOF),
        ("FP", "False Positive", "spoof accepted (security risk)", C_FP),
        ("FN", "False Negative", "live rejected",                  C_FN),
    ]
    n_cols = 4
    n_rows = 2

    # Chip strip is its own band; image grid sits below.
    CHIP_TOP = 0.905
    CHIP_H   = 0.060
    BOT      = 0.06
    GRID_TOP = CHIP_TOP - CHIP_H - 0.020

    for ci, (cell, line1, line2, col) in enumerate(cells):
        # Header chip with two-line text so the long FP label fits.
        col_x = M_LEFT + ci * (1 - M_LEFT - M_RIGHT) / n_cols + 0.008
        col_w = (1 - M_LEFT - M_RIGHT) / n_cols - 0.016
        ax_h = fig.add_axes([col_x, CHIP_TOP - CHIP_H, col_w, CHIP_H])
        ax_h.axis("off")
        ax_h.add_patch(FancyBboxPatch((0, 0), 1, 1,
                                      boxstyle="round,pad=0.0,rounding_size=0.10",
                                      facecolor=_tint(col, 0.20),
                                      edgecolor=col, linewidth=1.3,
                                      transform=ax_h.transAxes))
        ax_h.text(0.5, 0.70, line1, ha="center", va="center",
                  fontsize=FS_BODY, fontweight="bold", color=col,
                  transform=ax_h.transAxes)
        ax_h.text(0.5, 0.28, line2, ha="center", va="center",
                  fontsize=FS_NOTE, color=col, style="italic",
                  transform=ax_h.transAxes)

        imgs = sorted(glob.glob(os.path.join(cdir, f"{cell}_*.jpg")))[:n_rows]
        grid_h = GRID_TOP - BOT
        cell_h = grid_h / n_rows
        for ri in range(n_rows):
            ax_im = fig.add_axes([col_x,
                                  BOT + (n_rows - 1 - ri) * cell_h + 0.005,
                                  col_w,
                                  cell_h - 0.015])
            ax_im.axis("off")
            if ri < len(imgs) and os.path.exists(imgs[ri]):
                try:
                    img = mpimg.imread(imgs[ri])
                    ax_im.imshow(img)
                except Exception:
                    pass
            else:
                ax_im.add_patch(plt.Rectangle((0.05, 0.1), 0.9, 0.8,
                                              facecolor=C_BG_ALT,
                                              edgecolor=C_BORDER, linewidth=1,
                                              transform=ax_im.transAxes))
                msg = "no sample" if cell not in ("FP", "FN") else "no errors (ideal)"
                ax_im.text(0.5, 0.5, msg, ha="center", va="center",
                           fontsize=FS_NOTE, color=C_SUBTEXT, style="italic",
                           transform=ax_im.transAxes)

    fig.text(0.5, 0.025,
             "FP (false accept) and FN (false reject) are the failure modes -- "
             "FP is the most dangerous because it lets an attacker in.",
             ha="center", va="center", fontsize=FS_NOTE,
             color=C_SUBTEXT, style="italic")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------
# 26b. Mask failure analysis
#
# Two-panel figure that calls out the most interesting weakness of
# MiniFASNet on this dataset: bona-fide subjects wearing a surgical
# mask being mis-classified as a presentation attack.
#   left  : 1-row confusion matrix for the 'mask' subgroup
#   right : grouped bar chart of BPCER_user vs BPCER_mask vs APCER_spoof
# ---------------------------------------------------------------
def export_mask_failure(df_cm_mask: pd.DataFrame, df_group: pd.DataFrame,
                          df_summary: pd.DataFrame, out_path: str) -> None:
    if df_cm_mask is None or df_cm_mask.empty:
        print("[INFO] Confusion Mask Only sheet missing -- skipping fig 26b.")
        return
    apply_base_style()
    fig = plt.figure(figsize=(15.5, 6.2))
    fig.suptitle("Figure 6b - Mask Failure Analysis",
                  fontsize=FS_TITLE, fontweight="bold", color=C_HEADER,
                  x=0.5, y=0.965)

    # ---- Left panel: mask-only confusion (1 row x 2 cols) -----------------
    ax_cm = fig.add_axes([0.06, 0.18, 0.42, 0.68])
    row = df_cm_mask.iloc[0]
    pred_live = int(row.get("pred_live", 0))
    pred_spoof = int(row.get("pred_spoof", 0))
    total = max(1, pred_live + pred_spoof)
    bpcer_mask = pred_spoof / total

    cells = [("Pred LIVE", pred_live, C_GREEN, True),
             ("Pred SPOOF", pred_spoof, C_FP, False)]
    vmax = max(pred_live, pred_spoof, 1)
    for i, (col_label, val, edge_col, is_correct) in enumerate(cells):
        face = _tint(C_BLUE, val / vmax * 0.85)
        ax_cm.add_patch(plt.Rectangle((i, 0), 1, 1,
                                       facecolor=face,
                                       edgecolor=edge_col, linewidth=3.0))
        ax_cm.text(i + 0.5, 0.5, f"{val:,}", ha="center", va="center",
                   fontsize=FS_TITLE + 4, fontweight="bold",
                   color=C_DARK_TEXT)
        ax_cm.text(i + 0.5, -0.18, col_label, ha="center", va="center",
                   fontsize=FS_BODY, color=C_HEADER, fontweight="bold")
        verdict = "correct" if is_correct else "BPCER"
        ax_cm.text(i + 0.5, 1.18, verdict, ha="center", va="center",
                   fontsize=FS_NOTE, color=C_SUBTEXT, style="italic")
    ax_cm.text(-0.18, 0.5, "Actual:\nLive (mask)", ha="right", va="center",
               fontsize=FS_BODY, color=C_HEADER, fontweight="bold")
    ax_cm.set_xlim(-0.35, 2.05)
    ax_cm.set_ylim(-0.42, 1.42)
    ax_cm.set_aspect("equal")
    ax_cm.set_xticks([]); ax_cm.set_yticks([])
    for s in ax_cm.spines.values():
        s.set_visible(False)
    ax_cm.set_title(f"Mask-only confusion  (BPCER_mask = {bpcer_mask*100:.2f}%)",
                     fontsize=FS_SECTION, color=C_HEADER, pad=14)

    # ---- Right panel: per-group rate bars --------------------------------
    ax_bar = fig.add_axes([0.56, 0.18, 0.40, 0.68])
    bar_data: List[Tuple[str, float, str]] = []
    if df_group is not None and not df_group.empty:
        for grp_key, label_disp in (
            ("user",  "BPCER (user)"),
            ("mask",  "BPCER (mask)"),
            ("spoof", "APCER (spoof)"),
        ):
            sub = df_group[df_group["group"] == grp_key]
            if sub.empty:
                continue
            col = "BPCER(%)" if grp_key in ("user", "mask") else "APCER(%)"
            if col not in sub.columns:
                continue
            val = float(sub.iloc[0][col]) if not pd.isna(sub.iloc[0][col]) else 0.0
            color = {"user": C_BLUE, "mask": C_ORANGE, "spoof": C_PINK}[grp_key]
            bar_data.append((label_disp, val, color))

    if bar_data:
        labels = [b[0] for b in bar_data]
        vals = [b[1] for b in bar_data]
        colors = [b[2] for b in bar_data]
        xs = np.arange(len(labels))
        bars = ax_bar.bar(xs, vals, color=colors,
                           edgecolor=[c for c in colors],
                           linewidth=1.6, width=0.62, alpha=0.92)
        for b, v in zip(bars, vals):
            ax_bar.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02,
                        f"{v:.2f}%", ha="center", va="bottom",
                        fontsize=FS_BODY, fontweight="bold", color=C_DARK_TEXT)
        ax_bar.set_xticks(xs)
        ax_bar.set_xticklabels(labels, fontsize=FS_TICK, color=C_HEADER)
        ax_bar.set_ylabel("Error rate (%)", fontsize=FS_AXIS_LABEL, color=C_HEADER)
        ax_bar.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1.0)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.grid(axis="y", linestyle=":", alpha=0.4)
    else:
        ax_bar.text(0.5, 0.5, "per-group rates not available",
                    ha="center", va="center", color=C_SUBTEXT, style="italic",
                    transform=ax_bar.transAxes, fontsize=FS_BODY)
        ax_bar.set_xticks([]); ax_bar.set_yticks([])
    ax_bar.set_title("Per-group error rates at default threshold",
                       fontsize=FS_SECTION, color=C_HEADER, pad=14)

    fig.text(0.5, 0.04,
             "Higher BPCER(mask) means more masked bona-fide subjects "
             "are wrongly rejected as spoof -- a key UX issue when face "
             "masks are common.",
             ha="center", va="center", fontsize=FS_NOTE,
             color=C_SUBTEXT, style="italic")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------
# 28b. Mask misclassified sample frames
#
# Show the 6 mask frames where the model was MOST confident the
# subject was a spoof (lowest live_score).  These are the frames
# the report should discuss when explaining the mask failure mode.
# ---------------------------------------------------------------
def export_mask_misclassified_samples(df_log: pd.DataFrame, sample_root: str,
                                        out_path: str, n: int = 6) -> None:
    if df_log is None or df_log.empty or "group" not in df_log.columns:
        print("[INFO] Frame Log empty -- skipping fig 28b.")
        return
    mask_log = df_log[(df_log["group"] == "mask") & (df_log["is_real_default"] == 0)]
    if mask_log.empty:
        print("[INFO] No mis-classified mask frames -- skipping fig 28b.")
        return

    # Sort by lowest live_score (most confidently mis-classified)
    mask_log = mask_log.sort_values("live_score", ascending=True).head(n)

    apply_base_style()
    cols = 3
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 4.4, rows * 3.4 + 1.6))
    fig.suptitle("Figure 8b - Masked Bona-fide Frames Mis-classified as Spoof",
                  fontsize=FS_TITLE, fontweight="bold", color=C_HEADER,
                  x=0.5, y=0.97)
    fig.text(0.5, 0.93,
             "Lowest live_score frames from the 'mask' group -- these are the "
             "most representative BPCER failures to feature in the report.",
             ha="center", va="center", fontsize=FS_NOTE,
             color=C_SUBTEXT, style="italic")

    gs = fig.add_gridspec(rows, cols, left=0.04, right=0.96,
                            top=0.88, bottom=0.06,
                            wspace=0.10, hspace=0.22)

    for i, (_, row) in enumerate(mask_log.iterrows()):
        r, c = i // cols, i % cols
        ax = fig.add_subplot(gs[r, c])
        video = str(row["video"]).replace(".mp4", "")
        fidx = int(row["frame_idx"])
        score = float(row["live_score"])
        # Look for matching sample file in best/worst heaps written by the test
        candidates = []
        vdir = os.path.join(sample_root, video)
        if os.path.isdir(vdir):
            for fn in os.listdir(vdir):
                if fn.lower().endswith((".jpg", ".png")):
                    candidates.append(os.path.join(vdir, fn))
        img = None
        for cand in candidates:
            if f"_f{fidx}" in cand or f"_frame{fidx}" in cand:
                img = cand
                break
        if img is None and candidates:
            img = candidates[0]
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(C_FP); s.set_linewidth(2.0)
        if img and os.path.exists(img):
            try:
                ax.imshow(mpimg.imread(img))
            except Exception:
                ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                            facecolor=C_BG_ALT))
                ax.text(0.5, 0.5, "image not available", ha="center", va="center",
                         transform=ax.transAxes, color=C_SUBTEXT, style="italic")
        else:
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                        facecolor=C_BG_ALT))
            ax.text(0.5, 0.5, "no representative frame saved",
                     ha="center", va="center", transform=ax.transAxes,
                     color=C_SUBTEXT, style="italic", fontsize=FS_NOTE)
        ax.set_title(f"{video}  frame {fidx}\nlive_score = {score:.3f}",
                      fontsize=FS_NOTE, color=C_HEADER, pad=4)

    # Hide unused axes
    for j in range(len(mask_log), rows * cols):
        r, c = j // cols, j % cols
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
# ============================================================
# Figure 26c -- Spoof Type Comparison (phone vs print)
# ============================================================
def export_spoof_type_comparison(df_spoof_type: pd.DataFrame,
                                  df_log: pd.DataFrame,
                                  out_path: str) -> None:
    """Tach APCER cua phone vs print, va so sanh score distribution.

    Day la figure quan trong cho luan van: chung minh he thong khang
    duoc nhieu loai presentation attack, va chi ra loai nao kho hon.

    Layout:
      [Trai] Bar chart APCER + so frame moi loai
      [Phai] Score distribution histogram phone vs print (overlay)
    """
    apply_base_style()

    if df_spoof_type.empty:
        print(f"[SKIP] Per Spoof Type empty -> {out_path}")
        return

    fig = plt.figure(figsize=(15.5, 6.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25],
                          left=0.06, right=0.97, top=0.86, bottom=0.13,
                          wspace=0.28)

    # ---- LEFT: APCER per spoof type ----
    ax1 = fig.add_subplot(gs[0, 0])
    stypes = df_spoof_type["spoof_type"].tolist()
    apcers = df_spoof_type["APCER(%)"].to_numpy()
    n_frames = df_spoof_type["total_frames"].to_numpy()
    n_videos = df_spoof_type["videos"].to_numpy()

    colors_map = {"phone": C_PINK, "print": C_PURPLE, "other": C_ORANGE}
    bar_colors = [colors_map.get(s, C_BLUE) for s in stypes]

    x_pos = np.arange(len(stypes))
    bars = ax1.bar(x_pos, apcers, color=bar_colors, edgecolor="white",
                   linewidth=2.5, width=0.55, zorder=3)

    # Label tren bar
    for i, (b, a, n, v) in enumerate(zip(bars, apcers, n_frames, n_videos)):
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, h + max(apcers)*0.04,
                 f"{a:.2f}%", ha="center", va="bottom",
                 fontsize=FS_BODY + 2, fontweight="bold", color=C_TEXT)
        ax1.text(b.get_x() + b.get_width()/2, -max(apcers)*0.08,
                 f"{int(v)} videos, {int(n):,} frames",
                 ha="center", va="top", fontsize=FS_NOTE, color=C_SUBTEXT)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([s.capitalize() for s in stypes], fontsize=FS_BODY + 1)
    ax1.set_ylabel("APCER (%)", fontsize=FS_AXIS_LABEL)
    ax1.set_title("APCER by attack type", fontsize=FS_SECTION,
                  fontweight="bold", color=C_HEADER, pad=12)
    ax1.set_ylim(0, max(max(apcers) * 1.30, 1.0))
    ax1.grid(True, axis="y", alpha=0.3, zorder=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ---- RIGHT: Score distribution overlay ----
    ax2 = fig.add_subplot(gs[0, 1])

    if not df_log.empty and "spoof_type" in df_log.columns:
        spoof_log = df_log[df_log["group"] == "spoof"]
        bins = np.linspace(0, 1, 41)
        for stype in stypes:
            sub = spoof_log[spoof_log["spoof_type"] == stype]
            if sub.empty:
                continue
            ax2.hist(sub["live_score"], bins=bins,
                     color=colors_map.get(stype, C_BLUE),
                     alpha=0.55, edgecolor="white", linewidth=0.8,
                     label=f"{stype.capitalize()} (n={len(sub):,})", zorder=3)

    ax2.set_xlabel("Live score", fontsize=FS_AXIS_LABEL)
    ax2.set_ylabel("Frame count", fontsize=FS_AXIS_LABEL)
    ax2.set_title("Live-score distribution per attack type",
                  fontsize=FS_SECTION, fontweight="bold",
                  color=C_HEADER, pad=12)
    ax2.legend(loc="upper center", fontsize=FS_BODY, frameon=False,
               ncol=len(stypes))
    ax2.grid(True, alpha=0.3, zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_xlim(0, 1)

    # ---- Suptitle ----
    fig.suptitle("Figure 6c - Spoof Attack Type Comparison",
                 fontsize=FS_TITLE, fontweight="bold",
                 color=C_HEADER, y=0.96)
    fig.text(0.5, 0.04,
             "Lower APCER means the system is more robust against that attack type.",
             ha="center", fontsize=FS_NOTE, color=C_SUBTEXT, style="italic")

    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Spoof type comparison -> {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Render figures for the anti-spoofing report.")
    parser.add_argument(
        "--single", choices=["default", "eer"], default=None,
        help=("Also render ONE confusion matrix for the final / production report "
              "(use after threshold has been frozen on a dev set). "
              "'eer' = use the EER threshold; 'default' = use whatever "
              "threshold is currently in app.py."))
    parser.add_argument(
        "--single-only", action="store_true",
        help="Skip the dev figures and render only the single confusion matrix.")
    args = parser.parse_args()

    if not os.path.exists(REPORT_XLSX):
        print(f"[ERROR] {REPORT_XLSX} not found. Run test_anti_spoofing.py first.")
        sys.exit(1)
    os.makedirs(IMG_DIR, exist_ok=True)

    xw = pd.ExcelFile(REPORT_XLSX)
    df_video      = pd.read_excel(xw, "Per Video")            if "Per Video"           in xw.sheet_names else pd.DataFrame()
    df_group      = pd.read_excel(xw, "Per Group")            if "Per Group"           in xw.sheet_names else pd.DataFrame()
    df_cm_default = pd.read_excel(xw, "Confusion @ Default")  if "Confusion @ Default" in xw.sheet_names else pd.DataFrame()
    df_cm_eer     = pd.read_excel(xw, "Confusion @ EER")      if "Confusion @ EER"     in xw.sheet_names else pd.DataFrame()
    df_cm_mask    = pd.read_excel(xw, "Confusion Mask Only")  if "Confusion Mask Only" in xw.sheet_names else pd.DataFrame()
    df_spoof_type = pd.read_excel(xw, "Per Spoof Type")       if "Per Spoof Type"      in xw.sheet_names else pd.DataFrame()
    df_roc        = pd.read_excel(xw, "ROC Sweep")            if "ROC Sweep"           in xw.sheet_names else pd.DataFrame()
    df_summary    = pd.read_excel(xw, "Summary")              if "Summary"             in xw.sheet_names else pd.DataFrame()
    df_log        = pd.read_excel(xw, "Frame Log")            if "Frame Log"           in xw.sheet_names else pd.DataFrame()

    if not args.single_only:
        export_per_video_table   (df_video,      os.path.join(IMG_DIR, "20_summary_table.png"))
        export_group_table       (df_group,      os.path.join(IMG_DIR, "21_group_summary_table.png"))
        export_metrics_table     (df_summary,    os.path.join(IMG_DIR, "22_metrics_table.png"))
        export_roc_curve         (df_roc, df_summary,
                                  os.path.join(IMG_DIR, "23_roc_curve.png"))
        export_score_distribution(df_log, df_summary,
                                  os.path.join(IMG_DIR, "24_score_distribution.png"))
        export_confusion_matrices(df_cm_default, df_cm_eer, df_summary,
                                  os.path.join(IMG_DIR, "25_confusion_matrices.png"))
        export_infographic       (df_video, df_group, df_roc, df_summary, df_log,
                                  os.path.join(IMG_DIR, "26_infographic.png"))
        export_mask_failure      (df_cm_mask, df_group, df_summary,
                                  os.path.join(IMG_DIR, "26b_mask_failure.png"))
        export_spoof_type_comparison(df_spoof_type, df_log,
                                      os.path.join(IMG_DIR, "26c_spoof_type_comparison.png"))
        export_sample_frames     (SAMPLE_ROOT, df_video,
                                  os.path.join(IMG_DIR, "27_sample_frames.png"))
        export_confusion_samples_fig(SAMPLE_ROOT,
                                     os.path.join(IMG_DIR, "28_confusion_samples.png"))
        export_mask_misclassified_samples(df_log, SAMPLE_ROOT,
                                            os.path.join(IMG_DIR, "28b_mask_misclassified.png"))

    if args.single:
        row_key = "default" if args.single == "default" else "EER"
        df_cm_single = df_cm_default if args.single == "default" else df_cm_eer
        thr_txt = ""
        if not df_summary.empty:
            r = df_summary[df_summary["operating_point"].str.contains(row_key)]
            if not r.empty:
                t = r["threshold"].iloc[0]
                if pd.isna(t):
                    thr_txt = "(production threshold)"
                else:
                    thr_txt = f"(threshold = {float(t):.2f})"
        out_path = os.path.join(IMG_DIR, "29_confusion_matrix_single.png")
        export_single_confusion_matrix(df_cm_single, df_summary, row_key,
                                       out_path, title_suffix=thr_txt)
        print(f"[OK] Single confusion matrix -> {out_path}")

    print(f"\n[OK] Anti-spoofing figures exported to: {IMG_DIR}/")


if __name__ == "__main__":
    main()
