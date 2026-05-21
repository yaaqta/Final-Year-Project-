"""
============================================================
 VISUALIZE FACE-RECOGNITION RESULTS  --  v1
============================================================
Reads results/Recog_Report.xlsx (produced by test_face_recognition_v2.py)
and exports clean academic-style PNG figures for the thesis.

Outputs (in results/report_images/):
  - 29a_dataset_composition.png     Stacked-bar of frames per identity x group
  - 29b_split_overview.png          12 identities -> 8R+4S -> dev/test flow
  - 30_summary_table.png            Per-video table
  - 31_group_summary_table.png      Per-identity averages
  - 31b_per_group_breakdown.png     Per-group metrics (normal / mask / spoof / ALL)
  - 32_metrics_table.png            Accuracy / FAR / FRR / IDR at Default + Optimal
  - 33_threshold_sweep.png          FAR vs FRR vs Accuracy threshold sweep
  - 34_distance_distribution.png    Cosine-distance histogram (registered vs stranger)
  - 35a_confusion_binary.png        2x2 confusion matrix (Default + Optimal)
  - 35b_confusion_full.png          (N+1) x (N+1) confusion matrix (skipped when N>7)
  - 36_infographic.png              One-page summary
  - 37_sample_frames.png            Per-identity sample frames (3 best + 2 worst)
  - 38_confusion_samples.png        2 frames each per cell (TP / TN / FA / FR)

With `--single default|optimal` an extra figure is rendered for the final
production report:
  - 39_confusion_matrix_single.png  ONE confusion matrix at the chosen op-point

Style guide is reused from visualize_results.py / visualize_spoofing.py so
all three test reports share the same visual identity.

Run:
   python visualize_recognition.py
   python visualize_recognition.py --single optimal --single-only
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
from matplotlib.colors import LinearSegmentedColormap

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
REPORT_XLSX = os.path.join(RESULTS_DIR, "Recog_Report.xlsx")
IMG_DIR = os.path.join(RESULTS_DIR, "report_images")
SAMPLE_ROOT = os.path.join(RESULTS_DIR, "recog_samples")

# Semantic colour mapping for recognition reporting.
C_REGISTERED = C_BLUE      # actual = registered user
C_STRANGER   = C_PINK      # actual = stranger (Unknown)
C_FA = "#E74C3C"           # false accept (security incident)
C_FR = C_ORANGE            # false reject  (UX issue)
C_TP = C_GREEN
C_TN = C_TEAL

C_DARK_TEXT = "#2C3E50"

# Pastel sequential colourmap for confusion-matrix heatmap.
CMAP_BLUE = LinearSegmentedColormap.from_list(
    "pastel_blue", ["#FFFFFF", "#E8F1FB", "#C7DEF4", "#9CC4EA", C_BLUE, "#3B7AB8"]
)


# ---------------------------------------------------------------
# 30. Per-video summary table
# ---------------------------------------------------------------
def export_per_video_table(df_video: pd.DataFrame, out_path: str) -> None:
    if df_video.empty:
        print("[WARN] Per Video sheet is empty, skipping table.")
        return
    show = df_video[[
        "video", "true_label", "is_registered", "total_frames",
        "detection_rate(%)", "mean_min_dist",
        "frame_accuracy(%)", "mean_latency(ms)",
    ]].copy()
    show.columns = [
        "Video", "True Label", "Registered", "Frames",
        "Detect Rate (%)", "Mean Min-Dist",
        "Frame Accuracy (%)", "Mean Latency (ms)",
    ]
    show["Registered"]           = show["Registered"].apply(lambda v: "Yes" if int(v) else "No")
    show["Frames"]               = show["Frames"].apply(lambda v: f"{int(v):,}")
    show["Detect Rate (%)"]      = show["Detect Rate (%)"].apply(lambda v: f"{float(v):.2f}")
    show["Mean Min-Dist"]        = show["Mean Min-Dist"].apply(lambda v: f"{float(v):.3f}")
    show["Frame Accuracy (%)"]   = show["Frame Accuracy (%)"].apply(lambda v: f"{float(v):.2f}")
    show["Mean Latency (ms)"]    = show["Mean Latency (ms)"].apply(lambda v: f"{float(v):.2f}")

    render_table(
        show, "Table 1 - Per Video Face-Recognition Results", out_path,
        footnote="Mean Min-Dist is the average cosine distance to the closest "
                 "registered embedding (lower = closer match). "
                 "Frame Accuracy uses the default threshold from app.py.",
    )


# ---------------------------------------------------------------
# 31. Per-identity summary table
# ---------------------------------------------------------------
def export_identity_table(df_ident: pd.DataFrame, out_path: str) -> None:
    if df_ident.empty:
        print("[WARN] Per Identity sheet is empty, skipping table.")
        return
    show = df_ident.copy()
    show.columns = [c.replace("_", " ").title() for c in show.columns]
    for c in show.columns:
        if pd.api.types.is_float_dtype(show[c]):
            show[c] = show[c].apply(lambda v: f"{float(v):.2f}")
        elif pd.api.types.is_integer_dtype(show[c]):
            show[c] = show[c].apply(lambda v: f"{int(v):,}")
    render_table(
        show, "Table 2 - Per Identity Averages", out_path,
        footnote="Each row aggregates all videos labelled with the same identity. "
                 "'Unknown' rows correspond to stranger (out-of-DB) videos.",
    )


# ---------------------------------------------------------------
# 32. Metrics table (Default vs Optimal)
# ---------------------------------------------------------------
def export_metrics_table(df_summary: pd.DataFrame, out_path: str) -> None:
    if df_summary.empty:
        print("[WARN] Summary sheet is empty, skipping metrics table.")
        return
    show = df_summary.copy()
    def _fmt_thr(v): return "-" if pd.isna(v) else f"{float(v):.2f}"
    def _fmt_pct(v): return "-" if pd.isna(v) else f"{float(v):.2f}"
    def _fmt_int(v): return "-" if pd.isna(v) else f"{int(v):,}"

    if "threshold" in show.columns:
        show["threshold"] = show["threshold"].apply(_fmt_thr)
    for col in ("Accuracy(%)", "FAR(%)", "FRR(%)", "IDR(%)"):
        if col in show.columns:
            show[col] = show[col].apply(_fmt_pct)
    for col in ("TP", "TN", "FA", "FR", "MIS"):
        if col in show.columns:
            show[col] = show[col].apply(_fmt_int)

    show.columns = [c.replace("(%)", " (%)").replace("_", " ").title()
                    for c in show.columns]
    render_table(
        show, "Table 3 - Accuracy / FAR / FRR / IDR at Two Operating Points", out_path,
        footnote="FAR = stranger frames accepted as a registered user (security). "
                 "FRR = registered frames rejected (UX). "
                 "IDR = registered frames correctly named, among accepted frames.",
    )


# ---------------------------------------------------------------
# 33. Threshold sweep (FAR / FRR / Accuracy)
# ---------------------------------------------------------------
def export_threshold_sweep(df_sweep: pd.DataFrame, df_summary: pd.DataFrame,
                            out_path: str) -> None:
    if df_sweep.empty:
        print("[WARN] Threshold Sweep sheet is empty.")
        return
    apply_base_style()
    fig = plt.figure(figsize=(11, 6.4), dpi=150)
    fig.patch.set_facecolor("white")

    fig.text(0.5, 1 - M_TOP / 2, "Figure 1 - Threshold Sweep (FAR vs FRR)",
             ha="center", va="center", fontsize=FS_TITLE,
             fontweight="bold", color=C_TEXT)
    fig.text(0.5, 1 - M_TOP / 2 - 0.04,
             "Trade-off between security and usability as the cosine-distance threshold moves",
             ha="center", va="center", fontsize=FS_SUBTITLE, color=C_SUBTEXT)

    ax = fig.add_axes([M_LEFT + 0.04, M_BOTTOM + 0.10,
                       1 - M_LEFT - M_RIGHT - 0.08,
                       1 - M_TOP - M_BOTTOM - 0.20])

    thr = df_sweep["threshold"]
    far = df_sweep["FAR(%)"]
    frr = df_sweep["FRR(%)"]
    acc = df_sweep["Accuracy(%)"]

    # Recommended band: where FAR + FRR is within 1.5x of the minimum
    sum_err = far.fillna(0) + frr.fillna(0)
    min_se = float(sum_err.min())
    band_mask = sum_err <= min_se + 1.0  # within 1pp
    band_thrs = thr[band_mask]
    if len(band_thrs) >= 2:
        ax.axvspan(float(band_thrs.min()), float(band_thrs.max()),
                   facecolor=_tint(C_GREEN, 0.18), edgecolor="none",
                   zorder=0,
                   label=f"Recommended band ({float(band_thrs.min()):.2f}-{float(band_thrs.max()):.2f})")

    ax.plot(thr, far, color=C_FA,     lw=2.2, label="FAR (stranger accepted)")
    ax.plot(thr, frr, color=C_FR,     lw=2.2, label="FRR (registered rejected)")
    ax.plot(thr, acc, color=C_PURPLE, lw=2.0, linestyle="--", label="Accuracy")

    # Mark default + optimal points
    if not df_summary.empty:
        d_row = df_summary[df_summary["operating_point"] == "default"]
        o_row = df_summary[df_summary["operating_point"] == "optimal"]
        if not d_row.empty:
            t = float(d_row["threshold"].iloc[0])
            ax.axvline(t, color=C_TEXT, lw=1.0, alpha=0.4, linestyle="-.")
            ax.text(t, ax.get_ylim()[1] * 0.06, f" default {t:.2f}",
                    color=C_TEXT, fontsize=FS_NOTE)
        if not o_row.empty:
            t = float(o_row["threshold"].iloc[0])
            a = float(o_row["FAR(%)"].iloc[0]) if not pd.isna(o_row["FAR(%)"].iloc[0]) else 0.0
            ax.axvline(t, color=C_PURPLE, lw=1.2, alpha=0.55, linestyle=":")
            ax.scatter([t], [a], color=C_PURPLE, s=80, zorder=5,
                       edgecolor="white", linewidth=1.5)
            ax.annotate(f"Optimal\nthr={t:.2f}",
                        xy=(t, a), xytext=(t + 0.03, max(a + 8, 12)),
                        fontsize=FS_NOTE, color=C_PURPLE, fontweight="bold")

    ax.set_xlabel("Cosine-distance threshold", fontsize=FS_AXIS_LABEL, color=C_TEXT)
    ax.set_ylabel("Rate (%)", fontsize=FS_AXIS_LABEL, color=C_TEXT)
    ax.set_xlim(float(thr.min()), float(thr.max()))
    ax.set_ylim(0, 105)
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
             "Lower threshold = stricter (favours security). Higher threshold = more permissive (favours UX). "
             "The green band marks where FAR + FRR is near its minimum.",
             ha="center", va="center", fontsize=FS_NOTE,
             color=C_SUBTEXT, style="italic")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------
# 34. Distance distribution histogram (registered vs stranger)
# ---------------------------------------------------------------
def export_distance_distribution(df_log: pd.DataFrame, df_summary: pd.DataFrame,
                                  out_path: str) -> None:
    if df_log.empty:
        print("[WARN] Frame Log sheet is empty.")
        return
    apply_base_style()
    fig = plt.figure(figsize=(11, 5.8), dpi=150)
    fig.patch.set_facecolor("white")

    fig.text(0.5, 1 - M_TOP / 2,
             "Figure 2 - Distribution of Cosine Distances",
             ha="center", va="center", fontsize=FS_TITLE,
             fontweight="bold", color=C_TEXT)
    fig.text(0.5, 1 - M_TOP / 2 - 0.045,
             "Per-frame minimum cosine distance, separated by ground-truth identity class",
             ha="center", va="center", fontsize=FS_SUBTITLE, color=C_SUBTEXT)

    ax = fig.add_axes([M_LEFT + 0.04, M_BOTTOM + 0.10,
                       1 - M_LEFT - M_RIGHT - 0.08,
                       1 - M_TOP - M_BOTTOM - 0.22])

    # Use a sensible distance range -- cosine distance lives roughly in [0, 1.2]
    md = df_log["min_dist"].dropna()
    if md.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                fontsize=FS_BODY, color=C_SUBTEXT, transform=ax.transAxes)
    else:
        bins = np.linspace(0, max(1.0, float(md.max()) * 1.05), 31)
        reg = df_log[df_log["is_registered"] == 1]["min_dist"].dropna()
        stra = df_log[df_log["is_registered"] == 0]["min_dist"].dropna()
        if not reg.empty:
            ax.hist(reg, bins=bins, color=C_REGISTERED, alpha=0.55,
                    edgecolor="white", linewidth=0.6,
                    label=f"Registered (n={len(reg):,})")
        if not stra.empty:
            ax.hist(stra, bins=bins, color=C_STRANGER, alpha=0.55,
                    edgecolor="white", linewidth=0.6,
                    label=f"Stranger (n={len(stra):,})")

    # Mark default + optimal thresholds
    if not df_summary.empty:
        for op, ls, col in (("default", "-.", C_TEXT),
                            ("optimal", "--", C_PURPLE)):
            r = df_summary[df_summary["operating_point"] == op]
            if not r.empty:
                t = float(r["threshold"].iloc[0])
                ax.axvline(t, color=col, lw=1.8, linestyle=ls, alpha=0.85,
                           label=f"{op.title()} threshold = {t:.2f}")

    ax.set_xlabel("Cosine distance to closest DB embedding",
                  fontsize=FS_AXIS_LABEL, color=C_TEXT)
    ax.set_ylabel("Frame count", fontsize=FS_AXIS_LABEL, color=C_TEXT)
    ax.grid(alpha=0.25, color=C_BORDER, axis="y")
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
             "A clean gap between the two distributions means FaceNet separates "
             "registered users from strangers well.",
             ha="center", va="center", fontsize=FS_NOTE,
             color=C_SUBTEXT, style="italic")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------
# 35. Confusion matrices  (binary 2x2 + optional full N+1 x N+1)
# ---------------------------------------------------------------
def _draw_binary_cm(ax, cax, df_cm: pd.DataFrame, title: str,
                     title_color: str, vmax: float) -> None:
    """2x2 pastel-blue heatmap: rows=Actual (Registered, Stranger),
    cols=Predicted (Recognized, Unknown). One number per cell, no extras."""
    mat = df_cm[["pred_recognized", "pred_unknown"]].to_numpy(dtype=float)
    actual = df_cm["actual"].tolist()

    ax.grid(False)
    ax.set_axisbelow(False)
    im = ax.imshow(mat, cmap=CMAP_BLUE, vmin=0, vmax=vmax, aspect="equal")

    for r in range(2):
        for c in range(2):
            val = int(mat[r, c])
            intensity = mat[r, c] / vmax if vmax > 0 else 0
            txt_color = "white" if intensity > 0.55 else C_DARK_TEXT
            ax.text(c, r, f"{val:,}", ha="center", va="center",
                    fontsize=FS_KPI_VALUE, fontweight="bold", color=txt_color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["RECOGNIZED", "UNKNOWN"],
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
    if title:
        ax.set_title(title, fontsize=FS_SECTION, fontweight="bold",
                     color=title_color, pad=14)

    cbar = plt.colorbar(im, cax=cax)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=FS_TICK, colors=C_DARK_TEXT, length=0)
    cbar.ax.set_ylabel("Frame count", fontsize=FS_NOTE, color=C_DARK_TEXT,
                       rotation=270, labelpad=14)


def export_confusion_binary(df_cm_default: pd.DataFrame,
                              df_cm_optimal: pd.DataFrame,
                              df_summary: pd.DataFrame,
                              out_path: str) -> None:
    if df_cm_default.empty or df_cm_optimal.empty:
        print("[WARN] Confusion matrix data empty.")
        return
    apply_base_style()
    fig = plt.figure(figsize=(14, 6.8), dpi=150)
    fig.patch.set_facecolor("white")

    fig.text(0.5, 1 - M_TOP / 2,
             "Figure 3 - Binary Confusion Matrices",
             ha="center", va="center", fontsize=FS_TITLE,
             fontweight="bold", color=C_TEXT)
    fig.text(0.5, 1 - M_TOP / 2 - 0.04,
             "Frame-level Registered/Stranger decisions: default vs. optimal threshold",
             ha="center", va="center", fontsize=FS_SUBTITLE, color=C_SUBTEXT)

    mat_l = df_cm_default[["pred_recognized", "pred_unknown"]].to_numpy(dtype=float)
    mat_r = df_cm_optimal[["pred_recognized", "pred_unknown"]].to_numpy(dtype=float)
    vmax = float(max(mat_l.max(), mat_r.max()))

    ax_l  = fig.add_axes([0.10, 0.24, 0.30, 0.56])
    cax_l = fig.add_axes([0.405, 0.24, 0.012, 0.56])
    ax_r  = fig.add_axes([0.56, 0.24, 0.30, 0.56])
    cax_r = fig.add_axes([0.865, 0.24, 0.012, 0.56])

    def _stats(op_label: str) -> str:
        r = df_summary[df_summary["operating_point"] == op_label]
        if r.empty:
            return ""
        def f(c): return f"{float(r[c].iloc[0]):.2f}" if not pd.isna(r[c].iloc[0]) else "-"
        return (f"Accuracy {f('Accuracy(%)')}%   "
                f"FAR {f('FAR(%)')}%   "
                f"FRR {f('FRR(%)')}%")

    def _title(op_label: str) -> str:
        r = df_summary[df_summary["operating_point"] == op_label]
        if r.empty:
            return op_label.title()
        t = float(r["threshold"].iloc[0])
        return f"{op_label.title()} threshold ({t:.2f})"

    _draw_binary_cm(ax_l, cax_l, df_cm_default, _title("default"), C_HEADER, vmax)
    _draw_binary_cm(ax_r, cax_r, df_cm_optimal, _title("optimal"), C_PURPLE, vmax)

    fig.text(0.25, 0.05, _stats("default"),
             ha="center", fontsize=FS_BODY, color=C_DARK_TEXT, fontweight="bold")
    fig.text(0.71, 0.05, _stats("optimal"),
             ha="center", fontsize=FS_BODY, color=C_DARK_TEXT, fontweight="bold")

    fig.text(0.5, M_BOTTOM / 2,
             "Top-left and bottom-right are correct decisions. "
             "Bottom-left = FA (stranger accepted, security risk). "
             "Top-right = FR (registered rejected, UX issue).",
             ha="center", va="center", fontsize=FS_NOTE,
             color=C_SUBTEXT, style="italic")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_confusion_full(df_cm_full: pd.DataFrame,
                           df_summary: pd.DataFrame,
                           out_path: str) -> None:
    """(N+1) x (N+1) heatmap. Only call when N is small enough to be readable."""
    if df_cm_full.empty:
        print("[INFO] Full confusion matrix not present (too many identities or no data).")
        return
    apply_base_style()

    # Columns are pred_<name> -- preserve the original order from the report.
    pred_cols = [c for c in df_cm_full.columns if c.startswith("pred_")]
    actuals = df_cm_full["actual"].tolist()
    preds = [c[len("pred_"):] for c in pred_cols]

    mat = df_cm_full[pred_cols].to_numpy(dtype=float)
    vmax = float(mat.max()) if mat.size else 1.0

    n = len(actuals)
    # Size figure proportionally to matrix size
    side = max(8.0, 1.2 * n + 4.0)
    fig = plt.figure(figsize=(side + 1.0, side + 0.8), dpi=150)
    fig.patch.set_facecolor("white")

    # Title band at the very top
    fig.text(0.5, 0.955,
             "Figure 3b - Full Confusion Matrix (Per Identity)",
             ha="center", va="center", fontsize=FS_TITLE,
             fontweight="bold", color=C_TEXT)
    thr_txt = ""
    if not df_summary.empty:
        r = df_summary[df_summary["operating_point"] == "default"]
        if not r.empty:
            thr_txt = f"At default threshold = {float(r['threshold'].iloc[0]):.2f}"
    fig.text(0.5, 0.915,
             thr_txt or "Per-identity classification counts",
             ha="center", va="center", fontsize=FS_SUBTITLE, color=C_SUBTEXT)

    # Leave room below the matrix for the x-axis label + footnote.
    ax  = fig.add_axes([0.16, 0.18, 0.72, 0.68])
    cax = fig.add_axes([0.89, 0.18, 0.015, 0.68])

    ax.grid(False)
    im = ax.imshow(mat, cmap=CMAP_BLUE, vmin=0, vmax=vmax, aspect="equal")

    for r in range(n):
        for c in range(len(preds)):
            val = int(mat[r, c])
            intensity = mat[r, c] / vmax if vmax > 0 else 0
            txt_color = "white" if intensity > 0.55 else C_DARK_TEXT
            ax.text(c, r, f"{val:,}", ha="center", va="center",
                    fontsize=max(10, int(FS_BODY * 1.2)),
                    fontweight="bold", color=txt_color)

    ax.set_xticks(range(len(preds)))
    ax.set_xticklabels([p.upper() for p in preds],
                       fontsize=FS_BODY, color=C_DARK_TEXT,
                       rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels([a.upper() for a in actuals],
                       fontsize=FS_BODY, color=C_DARK_TEXT)
    ax.set_xlabel("Predicted Identity", fontsize=FS_BODY, color=C_DARK_TEXT, labelpad=18)
    ax.set_ylabel("Actual Identity", fontsize=FS_BODY, color=C_DARK_TEXT, labelpad=10)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    cbar = plt.colorbar(im, cax=cax)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=FS_TICK, colors=C_DARK_TEXT, length=0)
    cbar.ax.set_ylabel("Frame count", fontsize=FS_NOTE, color=C_DARK_TEXT,
                       rotation=270, labelpad=14)

    fig.text(0.5, 0.035,
             "Diagonal cells are correct identifications. Off-diagonals show identity confusions; "
             "the 'UNKNOWN' column counts rejections.",
             ha="center", va="center", fontsize=FS_NOTE,
             color=C_SUBTEXT, style="italic")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_single_confusion(df_cm: pd.DataFrame, df_summary: pd.DataFrame,
                              op_label: str, out_path: str,
                              title_suffix: str = "") -> None:
    """One large 2x2 confusion matrix for the final / production report."""
    if df_cm.empty:
        print("[WARN] Single confusion matrix data empty.")
        return
    apply_base_style()
    fig = plt.figure(figsize=(9, 7.4), dpi=150)
    fig.patch.set_facecolor("white")

    title = "Figure 3 - Confusion Matrix"
    if title_suffix:
        title = f"{title}  {title_suffix}"
    fig.text(0.5, 1 - M_TOP / 2, title,
             ha="center", va="center", fontsize=FS_TITLE,
             fontweight="bold", color=C_TEXT)
    fig.text(0.5, 1 - M_TOP / 2 - 0.045,
             "Frame-level Registered/Stranger decisions at the chosen threshold",
             ha="center", va="center", fontsize=FS_SUBTITLE, color=C_SUBTEXT)

    mat = df_cm[["pred_recognized", "pred_unknown"]].to_numpy(dtype=float)
    vmax = float(mat.max()) if mat.size else 1.0

    ax  = fig.add_axes([0.20, 0.20, 0.50, 0.56])
    cax = fig.add_axes([0.715, 0.20, 0.018, 0.56])
    _draw_binary_cm(ax, cax, df_cm, "", C_HEADER, vmax)

    r = df_summary[df_summary["operating_point"] == op_label]
    if not r.empty:
        def f(c): return f"{float(r[c].iloc[0]):.2f}" if not pd.isna(r[c].iloc[0]) else "-"
        stats = (f"Accuracy {f('Accuracy(%)')}%   "
                 f"FAR {f('FAR(%)')}%   "
                 f"FRR {f('FRR(%)')}%")
        fig.text(0.46, 0.10, stats,
                 ha="center", fontsize=FS_BODY + 1,
                 color=C_DARK_TEXT, fontweight="bold")

    fig.text(0.5, M_BOTTOM / 2,
             "Top-left and bottom-right are correct decisions; "
             "bottom-left = FA (stranger accepted), top-right = FR (registered rejected).",
             ha="center", va="center", fontsize=FS_NOTE,
             color=C_SUBTEXT, style="italic")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------
# 36. One-page infographic
# ---------------------------------------------------------------
def export_infographic(df_video: pd.DataFrame, df_ident: pd.DataFrame,
                        df_sweep: pd.DataFrame, df_summary: pd.DataFrame,
                        df_log: pd.DataFrame, out_path: str) -> None:
    apply_base_style()
    fig = plt.figure(figsize=(13, 16.5), dpi=150)
    fig.patch.set_facecolor("white")

    HEADER_H = 0.060
    GAP = 0.020
    KPI_H = 0.070
    PV_H = 0.180
    SWEEP_H = 0.180
    HIST_H = 0.180
    CONC_H = 0.180

    y_top = 1 - M_TOP
    header_bot = y_top - HEADER_H
    kpi_bot = header_bot - GAP - KPI_H
    pv_bot  = kpi_bot - GAP - PV_H
    sw_bot  = pv_bot - GAP - SWEEP_H
    hist_bot = sw_bot - GAP - HIST_H
    conc_bot = hist_bot - GAP - CONC_H

    # Title
    fig.text(0.5, header_bot + HEADER_H * 0.55,
             "Face-Recognition Evaluation Report",
             ha="center", va="center", fontsize=FS_TITLE + 2,
             fontweight="bold", color=C_TEXT)
    fig.text(0.5, header_bot + HEADER_H * 0.15,
             "Smart Locker System  -  FaceNet (cosine distance)  -  default threshold 0.55",
             ha="center", va="center", fontsize=FS_SUBTITLE, color=C_SUBTEXT)

    # ===== KPI row =====
    total_frames = int(df_log.shape[0]) if not df_log.empty else 0
    if not df_summary.empty:
        d_row = df_summary[df_summary["operating_point"] == "default"]
        o_row = df_summary[df_summary["operating_point"] == "optimal"]
        def _g(df, col):
            if df.empty or pd.isna(df[col].iloc[0]):
                return float("nan")
            return float(df[col].iloc[0])
        acc_d = _g(d_row, "Accuracy(%)")
        far_d = _g(d_row, "FAR(%)")
        frr_d = _g(d_row, "FRR(%)")
        thr_o = _g(o_row, "threshold")
        acc_o = _g(o_row, "Accuracy(%)")
    else:
        acc_d = far_d = frr_d = thr_o = acc_o = float("nan")

    ax_kpi = fig.add_axes([M_LEFT, kpi_bot, 1 - M_LEFT - M_RIGHT, KPI_H])
    ax_kpi.set_xlim(0, 1); ax_kpi.set_ylim(0, 1); ax_kpi.axis("off")
    cards = [
        (f"{total_frames:,}",  "Total Frames",        C_BLUE),
        (f"{acc_d:.2f}%",      "Accuracy (default)",  C_GREEN),
        (f"{far_d:.2f}%",      "FAR (default)",       C_FA),
        (f"{frr_d:.2f}%",      "FRR (default)",       C_FR),
        (f"{acc_o:.2f}%",      f"Accuracy (opt {thr_o:.2f})", C_PURPLE),
    ]
    n = len(cards); gap = 0.018
    cw = (1 - gap * (n - 1)) / n
    for i, (val, lab, col) in enumerate(cards):
        x = i * (cw + gap)
        draw_kpi_card(ax_kpi, x, 0.0, cw, 1.0, val, lab, stroke=col)

    # ===== Card: Per-video mean min-dist bar =====
    ax_pv_bg = fig.add_axes([M_LEFT, pv_bot, 1 - M_LEFT - M_RIGHT, PV_H])
    ax_pv_bg.set_xlim(0, 1); ax_pv_bg.set_ylim(0, 1); ax_pv_bg.axis("off")
    draw_card(ax_pv_bg, 0.0, 0.0, 1.0, 1.0, stroke=C_BLUE, fill="white",
              radius=0.025, lw=1.4)
    ax_pv_bg.text(0.03, 0.90, "Mean Min-Distance per Video",
                  fontsize=FS_SECTION, fontweight="bold", color=C_TEXT,
                  transform=ax_pv_bg.transAxes)
    # Legend chips
    legend_items = [(C_REGISTERED, "Registered"), (C_STRANGER, "Stranger")]
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
        df_s = df_video.sort_values(["is_registered", "video"], ascending=[False, True]).reset_index(drop=True)
        colors = [C_REGISTERED if int(r) else C_STRANGER
                  for r in df_s["is_registered"]]
        bars = ax_pv.bar(range(len(df_s)),
                         df_s["mean_min_dist"].astype(float),
                         color=colors, edgecolor="white", linewidth=1.2)
        for i, (b, v) in enumerate(zip(bars, df_s["mean_min_dist"].astype(float))):
            ax_pv.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                       f"{v:.2f}", ha="center", fontsize=FS_NOTE,
                       color=C_TEXT, fontweight="bold")
        ax_pv.set_xticks(range(len(df_s)))
        ax_pv.set_xticklabels(df_s["video"], rotation=30, ha="right",
                              fontsize=FS_TICK, color=C_TEXT)
        # Mark threshold lines
        if not df_summary.empty:
            d_row = df_summary[df_summary["operating_point"] == "default"]
            if not d_row.empty:
                t = float(d_row["threshold"].iloc[0])
                ax_pv.axhline(t, color=C_TEXT, lw=1.0,
                              linestyle="-.", alpha=0.5)
                ax_pv.text(len(df_s) - 0.5, t + 0.015,
                           f"default thr {t:.2f}", color=C_TEXT,
                           fontsize=FS_NOTE, ha="right", style="italic")
        ymax = max(1.0, float(df_s["mean_min_dist"].max()) * 1.25)
        ax_pv.set_ylim(0, ymax)
        ax_pv.set_ylabel("Mean cosine distance",
                         fontsize=FS_AXIS_LABEL, color=C_TEXT)
    ax_pv.tick_params(axis="y", labelsize=FS_TICK, colors=C_SUBTEXT)
    ax_pv.grid(axis="y", alpha=0.25, color=C_BORDER)
    ax_pv.set_axisbelow(True)
    ax_pv.set_facecolor("none")
    for sp in ["top", "right"]:
        ax_pv.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax_pv.spines[sp].set_color(C_BORDER)
        ax_pv.spines[sp].set_linewidth(0.8)

    # ===== Card: Threshold sweep =====
    ax_sw_bg = fig.add_axes([M_LEFT, sw_bot, 1 - M_LEFT - M_RIGHT, SWEEP_H])
    ax_sw_bg.set_xlim(0, 1); ax_sw_bg.set_ylim(0, 1); ax_sw_bg.axis("off")
    draw_card(ax_sw_bg, 0.0, 0.0, 1.0, 1.0, stroke=C_PURPLE, fill="white",
              radius=0.025, lw=1.4)
    ax_sw_bg.text(0.03, 0.90, "FAR vs FRR Threshold Sweep",
                   fontsize=FS_SECTION, fontweight="bold", color=C_TEXT,
                   transform=ax_sw_bg.transAxes)

    bx_y2 = sw_bot + chart_y_b * SWEEP_H
    bx_h2 = (chart_y_t - chart_y_b) * SWEEP_H
    ax_sw = fig.add_axes([bx_x, bx_y2, bx_w, bx_h2])
    if not df_sweep.empty:
        ax_sw.plot(df_sweep["threshold"], df_sweep["FAR(%)"], color=C_FA, lw=2,
                   label="FAR")
        ax_sw.plot(df_sweep["threshold"], df_sweep["FRR(%)"], color=C_FR, lw=2,
                   label="FRR")
        ax_sw.plot(df_sweep["threshold"], df_sweep["Accuracy(%)"], color=C_PURPLE,
                   lw=1.6, linestyle="--", label="Accuracy")
        if not df_summary.empty:
            o = df_summary[df_summary["operating_point"] == "optimal"]
            if not o.empty:
                t = float(o["threshold"].iloc[0])
                ax_sw.axvline(t, color=C_PURPLE, lw=1.2, alpha=0.5, linestyle=":")
    ax_sw.set_xlabel("Cosine-distance threshold",
                     fontsize=FS_AXIS_LABEL, color=C_TEXT)
    ax_sw.set_ylabel("Rate (%)", fontsize=FS_AXIS_LABEL, color=C_TEXT)
    ax_sw.set_ylim(0, 105)
    ax_sw.tick_params(labelsize=FS_TICK, colors=C_SUBTEXT)
    ax_sw.grid(alpha=0.25, color=C_BORDER)
    ax_sw.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax_sw.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax_sw.spines[sp].set_color(C_BORDER)
        ax_sw.spines[sp].set_linewidth(0.8)
    leg = ax_sw.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18),
                       ncol=3, frameon=False, fontsize=FS_NOTE)
    for txt in leg.get_texts():
        txt.set_color(C_TEXT)

    # ===== Card: Distance distribution =====
    ax_h_bg = fig.add_axes([M_LEFT, hist_bot, 1 - M_LEFT - M_RIGHT, HIST_H])
    ax_h_bg.set_xlim(0, 1); ax_h_bg.set_ylim(0, 1); ax_h_bg.axis("off")
    draw_card(ax_h_bg, 0.0, 0.0, 1.0, 1.0, stroke=C_GREEN, fill="white",
              radius=0.025, lw=1.4)
    ax_h_bg.text(0.03, 0.90, "Distance Distribution (Registered vs Stranger)",
                 fontsize=FS_SECTION, fontweight="bold", color=C_TEXT,
                 transform=ax_h_bg.transAxes)

    bx_y3 = hist_bot + chart_y_b * HIST_H
    bx_h3 = (chart_y_t - chart_y_b) * HIST_H
    ax_h = fig.add_axes([bx_x, bx_y3, bx_w, bx_h3])
    if not df_log.empty:
        md = df_log["min_dist"].dropna()
        if not md.empty:
            bins = np.linspace(0, max(1.0, float(md.max()) * 1.05), 31)
            reg = df_log[df_log["is_registered"] == 1]["min_dist"].dropna()
            stra = df_log[df_log["is_registered"] == 0]["min_dist"].dropna()
            if not reg.empty:
                ax_h.hist(reg, bins=bins, color=C_REGISTERED, alpha=0.55,
                          edgecolor="white", linewidth=0.6,
                          label=f"Registered (n={len(reg):,})")
            if not stra.empty:
                ax_h.hist(stra, bins=bins, color=C_STRANGER, alpha=0.55,
                          edgecolor="white", linewidth=0.6,
                          label=f"Stranger (n={len(stra):,})")
    ax_h.set_xlabel("Cosine distance", fontsize=FS_AXIS_LABEL, color=C_TEXT)
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

    bullets: List[Tuple[str, str, str]] = []

    # 1. Headline finding
    if not pd.isna(acc_d):
        if not pd.isna(far_d) and far_d > 2.0:
            bullets.append((C_FA, "RISK:",
                            f"Default threshold yields FAR {far_d:.2f}% -- strangers being accepted. "
                            f"Tighten the threshold or add more enrolment samples per user."))
        elif acc_d >= 90.0:
            bullets.append((C_GREEN, "PASS:",
                            f"Default threshold delivers Accuracy {acc_d:.2f}% "
                            f"(FAR {far_d:.2f}%, FRR {frr_d:.2f}%). System is usable."))
        else:
            bullets.append((C_FR, "REVIEW:",
                            f"Default Accuracy is {acc_d:.2f}% -- mostly driven by FRR ({frr_d:.2f}%). "
                            f"Consider loosening the threshold or improving enrolment lighting."))

    # 2. Threshold recommendation
    if not pd.isna(thr_o):
        bullets.append((C_PURPLE, "TUNE:",
                        f"Optimal threshold on this set is ~{thr_o:.2f} "
                        f"(Accuracy {acc_o:.2f}%). "
                        f"Validate on a held-out test split before changing app.py."))

    # 3. Data recommendation
    n_videos = len(df_video) if not df_video.empty else 0
    n_reg = int((df_video["is_registered"] == 1).sum()) if not df_video.empty else 0
    n_str = n_videos - n_reg
    data_bits = []
    if n_reg < 6:
        data_bits.append(f"Add more registered-user videos (now {n_reg}; "
                         "target 2-3 per enrolled identity covering different lighting).")
    if n_str < 4:
        data_bits.append(f"Add more stranger videos (now {n_str}; "
                         "target >=5 distinct strangers to make FAR statistically meaningful).")
    if data_bits:
        bullets.append((C_BLUE, "DATA:", " ".join(data_bits)))

    # 4. Deployment hardening
    bullets.append((C_ORANGE, "DEPLOY:",
                    "Use a dev/test split: tune the threshold on dev frames only, then "
                    "report the final confusion matrix from the held-out test set ONCE."))

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
# 37. Sample frames per identity (3 best + 2 worst)
# ---------------------------------------------------------------
def _pick_identity_samples(sample_root: str, video_names: List[str], n: int = 5
                            ) -> List[Tuple[str, str, str]]:
    bests, worsts = [], []
    for v in video_names:
        d = os.path.join(sample_root, v)
        for p in sorted(glob.glob(os.path.join(d, "best_*.jpg"))):
            bests.append((v, "BEST", p))
        for p in sorted(glob.glob(os.path.join(d, "worst_*.jpg"))):
            worsts.append((v, "WORST", p))

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


def export_sample_frames(sample_root: str, df_video: pd.DataFrame,
                          out_path: str) -> None:
    if df_video.empty:
        print("[WARN] No per-video data, skipping sample frames figure.")
        return
    if not os.path.isdir(sample_root):
        print(f"[WARN] No sample_root dir at {sample_root}; skipping figure 37.")
        return

    apply_base_style()

    identities = sorted(df_video["true_label"].unique().tolist())
    n_rows = len(identities)
    n_cols = 5
    if n_rows == 0:
        print("[WARN] no identities for sample frames")
        return

    # Auto-size: ~3.6 inches of vertical space per row, capped at 16
    fig_h = max(7.0, min(16.0, 3.4 * n_rows + 2.0))
    fig = plt.figure(figsize=(16, fig_h), dpi=150)
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.965, "Figure 4 - Sample Frames per Identity",
             ha="center", va="center", fontsize=FS_TITLE,
             fontweight="bold", color=C_TEXT)
    fig.text(0.5, 0.928,
             "3 best + 2 worst frames per identity  -  border colour encodes the confusion cell",
             ha="center", va="center", fontsize=FS_SUBTITLE, color=C_SUBTEXT)

    LEFT_LABEL_W = 0.07
    TOP_MARGIN = 0.880
    BOT_MARGIN = 0.06
    grid_h = TOP_MARGIN - BOT_MARGIN
    row_h = grid_h / n_rows

    for ri, ident in enumerate(identities):
        videos = (df_video[df_video["true_label"] == ident]["video"]
                  .str.replace(".mp4", "", regex=False).tolist())
        samples = _pick_identity_samples(sample_root, videos, n=n_cols)

        row_top = TOP_MARGIN - ri * row_h
        row_bot = row_top - row_h
        ax_lab = fig.add_axes([M_LEFT, row_bot + 0.06 * row_h,
                               LEFT_LABEL_W - M_LEFT - 0.005,
                               row_h - 0.12 * row_h])
        ax_lab.axis("off")
        is_unknown = (ident.lower() == "unknown")
        col = C_STRANGER if is_unknown else C_REGISTERED
        label_text = ("UNKNOWN\n(stranger)" if is_unknown
                      else ident.upper())
        ax_lab.add_patch(FancyBboxPatch(
            (0.0, 0.0), 1.0, 1.0,
            boxstyle="round,pad=0.0,rounding_size=0.08",
            facecolor=_tint(col, 0.18), edgecolor=col,
            linewidth=1.3, transform=ax_lab.transAxes))
        ax_lab.text(0.5, 0.5, label_text,
                    rotation=90, ha="center", va="center",
                    fontsize=FS_BODY, fontweight="bold", color=col,
                    transform=ax_lab.transAxes)

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
             "Green border = TP (registered correctly named). "
             "Blue border = TN (stranger rejected). "
             "Red border = FA (stranger accepted - critical). "
             "Orange border = FR (registered rejected).",
             ha="center", va="center", fontsize=FS_NOTE,
             color=C_SUBTEXT, style="italic")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------
# 38. Confusion-cell samples (TP / TN / FA / FR)
# ---------------------------------------------------------------
def export_confusion_samples_fig(sample_root: str, out_path: str) -> None:
    cdir = os.path.join(sample_root, "_confusion")
    if not os.path.isdir(cdir):
        print(f"[WARN] No confusion samples dir at {cdir}; skipping figure 38.")
        return

    apply_base_style()
    fig = plt.figure(figsize=(15, 9.2), dpi=150)
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.975, "Figure 5 - Per-Cell Confusion Samples",
             ha="center", va="center", fontsize=FS_TITLE,
             fontweight="bold", color=C_TEXT)
    fig.text(0.5, 0.945,
             "Two representative frames for each cell of the confusion matrix",
             ha="center", va="center", fontsize=FS_SUBTITLE, color=C_SUBTEXT)

    cells = [
        ("TP", "True Positive",  "registered correctly named",        C_TP),
        ("TN", "True Negative",  "stranger correctly rejected",       C_TN),
        ("FA", "False Accept",   "stranger accepted (security risk)", C_FA),
        ("FR", "False Reject",   "registered rejected (UX issue)",    C_FR),
    ]
    n_cols = 4
    n_rows = 2

    CHIP_TOP = 0.905
    CHIP_H   = 0.060
    BOT      = 0.06
    GRID_TOP = CHIP_TOP - CHIP_H - 0.020

    for ci, (cell, line1, line2, col) in enumerate(cells):
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
                msg = "no sample" if cell not in ("FA", "FR") else "no errors (ideal)"
                ax_im.text(0.5, 0.5, msg, ha="center", va="center",
                           fontsize=FS_NOTE, color=C_SUBTEXT, style="italic",
                           transform=ax_im.transAxes)

    fig.text(0.5, 0.025,
             "FA (false accept) is the most dangerous error -- a stranger was treated as a registered user. "
             "FR is the UX cost when a real user gets locked out.",
             ha="center", va="center", fontsize=FS_NOTE,
             color=C_SUBTEXT, style="italic")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------
# 29a. Dataset composition (stacked bar: identity x group)
# ---------------------------------------------------------------
def export_dataset_composition(df_manifest: pd.DataFrame,
                                 out_path: str) -> None:
    """Stacked-bar chart of frames per identity, coloured by group.

    Reads the 'Dataset Manifest' sheet produced by test_face_recognition_v2.
    Each identity contributes up to 3 stacked segments (normal / mask / spoof).
    """
    if df_manifest is None or df_manifest.empty:
        print("[INFO] Dataset Manifest empty -- skipping fig 29a.")
        return
    if "identity" not in df_manifest.columns or "group" not in df_manifest.columns:
        print("[INFO] Dataset Manifest missing identity/group -- skipping fig 29a.")
        return

    # Pivot: rows=identity, cols=group, values=frames_sampled
    frame_col = "frames_sampled" if "frames_sampled" in df_manifest.columns else "frames_with_face"
    df = df_manifest.copy()
    df[frame_col] = pd.to_numeric(df[frame_col], errors="coerce").fillna(0).astype(int)
    pivot = df.pivot_table(index="identity", columns="group",
                            values=frame_col, aggfunc="sum", fill_value=0)

    # Stable group order; only include groups actually present.
    group_order = [g for g in ("normal", "mask", "spoof", "unknown") if g in pivot.columns]
    if not group_order:
        print("[INFO] No known groups in Dataset Manifest -- skipping fig 29a.")
        return
    pivot = pivot[group_order]

    # Annotate role + split on x labels for context.
    role_lookup = (df.drop_duplicates("identity")
                     .set_index("identity")[["role", "split"]].to_dict("index"))
    # Order identities by role (registered first), then by name.
    def _key(name: str):
        meta = role_lookup.get(name, {})
        role = str(meta.get("role", "")).lower()
        return (0 if role == "registered" else 1, name)
    ordered_ids = sorted(pivot.index.tolist(), key=_key)
    pivot = pivot.loc[ordered_ids]

    apply_base_style()
    fig = plt.figure(figsize=(13.5, 6.6), dpi=150)
    fig.patch.set_facecolor("white")

    fig.text(0.5, 1 - M_TOP / 2,
             "Figure 0a - Dataset Composition",
             ha="center", va="center", fontsize=FS_TITLE,
             fontweight="bold", color=C_TEXT)
    fig.text(0.5, 1 - M_TOP / 2 - 0.04,
             "Frames sampled per identity, stacked by group (normal / mask / spoof)",
             ha="center", va="center", fontsize=FS_SUBTITLE, color=C_SUBTEXT)

    ax = fig.add_axes([M_LEFT + 0.04, M_BOTTOM + 0.16,
                       1 - M_LEFT - M_RIGHT - 0.08,
                       1 - M_TOP - M_BOTTOM - 0.28])

    group_colors = {"normal": C_BLUE, "mask": C_ORANGE,
                     "spoof": C_PINK, "unknown": C_PURPLE}
    xs = np.arange(len(ordered_ids))
    bottom = np.zeros(len(ordered_ids), dtype=float)
    for g in group_order:
        vals = pivot[g].values.astype(float)
        ax.bar(xs, vals, bottom=bottom,
               color=group_colors.get(g, C_BLUE),
               edgecolor="white", linewidth=1.2,
               width=0.7, alpha=0.92, label=g)
        # Per-segment label when segment is large enough.
        for i, v in enumerate(vals):
            if v >= max(pivot.values.max() * 0.06, 1):
                ax.text(xs[i], bottom[i] + v / 2, f"{int(v)}",
                        ha="center", va="center",
                        fontsize=FS_NOTE, color=C_DARK_TEXT, fontweight="bold")
        bottom += vals

    # Total label above each bar.
    totals = pivot.sum(axis=1).values
    top_y = max(totals.max(), 1)
    for i, t in enumerate(totals):
        ax.text(xs[i], t + top_y * 0.02, f"{int(t)}",
                ha="center", va="bottom",
                fontsize=FS_NOTE, color=C_TEXT, fontweight="bold")

    # X labels: name + role chip via two-line text.
    labels = []
    for n in ordered_ids:
        meta = role_lookup.get(n, {})
        role = str(meta.get("role", ""))
        split = str(meta.get("split", ""))
        role_tag = role.upper() if role else ""
        suffix = []
        if role_tag:
            suffix.append(role_tag[0])  # R / S
        if split and split.lower() in ("dev", "test"):
            suffix.append(split.lower())
        tag = (" / ".join(suffix)) if suffix else ""
        labels.append(f"{n}\n{tag}" if tag else n)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=FS_TICK, color=C_TEXT)
    ax.set_ylabel("Frames sampled", fontsize=FS_AXIS_LABEL, color=C_TEXT)
    ax.set_ylim(0, top_y * 1.18)
    ax.grid(axis="y", alpha=0.25, color=C_BORDER)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color(C_BORDER)
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(axis="y", labelsize=FS_TICK, colors=C_SUBTEXT)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10),
                     ncol=len(group_order), frameon=False, fontsize=FS_NOTE)
    for txt in leg.get_texts():
        txt.set_color(C_TEXT)

    # Summary footnote
    n_ids = len(ordered_ids)
    n_reg = sum(1 for n in ordered_ids
                 if str(role_lookup.get(n, {}).get("role", "")).lower() == "registered")
    n_str = n_ids - n_reg
    total_frames = int(pivot.values.sum())
    foot = (f"Corpus: {n_ids} identities ({n_reg} registered + {n_str} stranger), "
            f"{total_frames:,} frames total. "
            f"R = registered, S = stranger; dev/test indicates the split partition.")
    fig.text(0.5, M_BOTTOM / 2, foot,
             ha="center", va="center", fontsize=FS_NOTE,
             color=C_SUBTEXT, style="italic")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------
# 29b. Split overview (12 -> 8R + 4S -> dev / test)
# ---------------------------------------------------------------
def export_split_overview(df_manifest: pd.DataFrame, out_path: str) -> None:
    """Two-panel figure: left = role pie (R vs S), right = grouped bar of
    frames per (role, split).  Shows how 12 identities split into 8R + 4S
    and how each half is further split into dev / test.
    """
    if df_manifest is None or df_manifest.empty:
        print("[INFO] Dataset Manifest empty -- skipping fig 29b.")
        return
    for col in ("identity", "role", "split"):
        if col not in df_manifest.columns:
            print(f"[INFO] Dataset Manifest missing '{col}' -- skipping fig 29b.")
            return

    frame_col = "frames_sampled" if "frames_sampled" in df_manifest.columns else "frames_with_face"
    df = df_manifest.copy()
    df[frame_col] = pd.to_numeric(df[frame_col], errors="coerce").fillna(0).astype(int)

    # Unique identity-level role/split (one row per identity)
    id_role = (df.drop_duplicates("identity")
                  .set_index("identity")[["role", "split"]])
    n_reg = int((id_role["role"].astype(str).str.lower() == "registered").sum())
    n_str = int((id_role["role"].astype(str).str.lower() == "stranger").sum())

    # Frames grouped by (role, split)
    df["role_l"]  = df["role"].astype(str).str.lower()
    df["split_l"] = df["split"].astype(str).str.lower()
    grp_frames = df.groupby(["role_l", "split_l"])[frame_col].sum().to_dict()

    # Identities grouped by (role, split)
    df_id_split = (df.drop_duplicates("identity")
                       [["identity", "role", "split"]])
    df_id_split["role_l"]  = df_id_split["role"].astype(str).str.lower()
    df_id_split["split_l"] = df_id_split["split"].astype(str).str.lower()
    grp_ids = df_id_split.groupby(["role_l", "split_l"])["identity"].apply(list).to_dict()

    apply_base_style()
    fig = plt.figure(figsize=(14, 6.4), dpi=150)
    fig.patch.set_facecolor("white")

    fig.text(0.5, 1 - M_TOP / 2,
             "Figure 0b - Identity Split Overview",
             ha="center", va="center", fontsize=FS_TITLE,
             fontweight="bold", color=C_TEXT)
    fig.text(0.5, 1 - M_TOP / 2 - 0.04,
             "How identities and frames flow into registered / stranger and dev / test partitions",
             ha="center", va="center", fontsize=FS_SUBTITLE, color=C_SUBTEXT)

    # ---- Left: role donut ------------------------------------------------
    ax_pie = fig.add_axes([0.06, M_BOTTOM + 0.08, 0.32,
                            1 - M_TOP - M_BOTTOM - 0.18])
    sizes = [n_reg, n_str]
    labels_pie = [f"Registered\n({n_reg} id)", f"Stranger\n({n_str} id)"]
    colors_pie = [C_BLUE, C_PINK]
    if sum(sizes) > 0:
        wedges, texts, autotexts = ax_pie.pie(
            sizes,
            labels=labels_pie,
            colors=colors_pie,
            startangle=90,
            wedgeprops=dict(width=0.36, edgecolor="white", linewidth=2),
            autopct=lambda p: f"{p:.0f}%",
            pctdistance=0.78,
            textprops=dict(color=C_TEXT, fontsize=FS_BODY, fontweight="bold"),
        )
        for at in autotexts:
            at.set_color(C_DARK_TEXT)
            at.set_fontsize(FS_NOTE)
    fig.text(0.22, 1 - M_TOP / 2 - 0.10,
              f"Identity roles  (N = {n_reg + n_str})",
              ha="center", va="center",
              fontsize=FS_SECTION, fontweight="bold", color=C_TEXT)

    # ---- Right: grouped bar of frames per (role, split) ------------------
    ax_bar = fig.add_axes([0.46, M_BOTTOM + 0.10, 0.50,
                            1 - M_TOP - M_BOTTOM - 0.26])
    role_keys  = [("registered", "Registered"), ("stranger", "Stranger")]
    split_keys = [("dev", "Dev"), ("test", "Test")]
    split_colors = {"dev": C_TEAL, "test": C_PURPLE}

    x = np.arange(len(role_keys))
    bar_w = 0.36
    max_h = 0.0
    for j, (sk, sl) in enumerate(split_keys):
        heights = []
        for rk, _ in role_keys:
            heights.append(grp_frames.get((rk, sk), 0))
        offs = (j - 0.5) * bar_w
        bars = ax_bar.bar(x + offs, heights, bar_w,
                            color=split_colors[sk], alpha=0.92,
                            edgecolor="white", linewidth=1.4, label=sl)
        # Label: frames + identity count
        for i, (rk, _) in enumerate(role_keys):
            n_id_here = len(grp_ids.get((rk, sk), []))
            h = heights[i]
            if h > 0:
                ax_bar.text(x[i] + offs, h + max(heights + [1]) * 0.02,
                            f"{int(h):,}\n({n_id_here} id)",
                            ha="center", va="bottom",
                            fontsize=FS_NOTE, color=C_DARK_TEXT, fontweight="bold")
        if heights:
            max_h = max(max_h, max(heights))

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([lbl for _, lbl in role_keys],
                             fontsize=FS_TICK, color=C_TEXT)
    ax_bar.set_ylabel("Frames", fontsize=FS_AXIS_LABEL, color=C_TEXT)
    ax_bar.set_ylim(0, (max_h or 1) * 1.28)
    ax_bar.grid(axis="y", alpha=0.25, color=C_BORDER)
    ax_bar.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax_bar.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax_bar.spines[sp].set_color(C_BORDER)
        ax_bar.spines[sp].set_linewidth(0.8)
    ax_bar.tick_params(axis="y", labelsize=FS_TICK, colors=C_SUBTEXT)
    fig.text(0.71, 1 - M_TOP / 2 - 0.10,
              "Frames per (role, split)",
              ha="center", va="center",
              fontsize=FS_SECTION, fontweight="bold", color=C_TEXT)
    leg = ax_bar.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0),
                          ncol=2, frameon=False, fontsize=FS_NOTE)
    for txt in leg.get_texts():
        txt.set_color(C_TEXT)

    # Footnote: list the identities per cell so the reader can audit the split
    parts = []
    for rk, rl in role_keys:
        for sk, sl in split_keys:
            ids = grp_ids.get((rk, sk), [])
            if ids:
                parts.append(f"{rl} {sl}: " + ", ".join(sorted(ids)))
    foot = " | ".join(parts) if parts else ""
    if foot:
        fig.text(0.5, M_BOTTOM / 2 - 0.005, foot,
                  ha="center", va="center", fontsize=FS_NOTE,
                  color=C_SUBTEXT, style="italic", wrap=True)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------
# 31b. Per-group breakdown (normal / mask / spoof / ALL)
# ---------------------------------------------------------------
def export_per_group_breakdown(df_per_group: pd.DataFrame,
                                 df_summary: pd.DataFrame,
                                 out_path: str) -> None:
    """Per-group bar chart + table reading the 'Per Group' sheet.

    Reads df_per_group with columns operating_point, group, Accuracy(%),
    FAR(%), FRR(%), IDR(%), TP, TN, FA, FR, MIS.  Plots grouped bars
    (Accuracy, FAR, FRR, IDR) per group at the DEFAULT operating point
    and renders a compact metrics table below.
    """
    if df_per_group is None or df_per_group.empty:
        print("[INFO] Per Group sheet empty -- skipping fig 31b.")
        return
    if "operating_point" in df_per_group.columns:
        df = df_per_group[df_per_group["operating_point"] == "default"].copy()
        if df.empty:
            df = df_per_group.copy()
    else:
        df = df_per_group.copy()
    if "group" not in df.columns:
        print("[INFO] Per Group missing 'group' column -- skipping fig 31b.")
        return

    group_order = [g for g in ("normal", "mask", "spoof", "unknown", "ALL")
                    if g in df["group"].astype(str).tolist()]
    if not group_order:
        return
    df = df.set_index("group").loc[group_order].reset_index()

    metric_cols = [
        ("Accuracy(%)", "Accuracy", C_GREEN),
        ("FAR(%)",      "FAR",      C_FA),
        ("FRR(%)",      "FRR",      C_FR),
        ("IDR(%)",      "IDR",      C_PURPLE),
    ]
    metric_cols = [(c, l, col) for (c, l, col) in metric_cols if c in df.columns]

    apply_base_style()
    fig = plt.figure(figsize=(13.5, 7.6), dpi=150)
    fig.patch.set_facecolor("white")

    fig.text(0.5, 1 - M_TOP / 2,
             "Figure 2b - Per-Group Recognition Breakdown",
             ha="center", va="center", fontsize=FS_TITLE,
             fontweight="bold", color=C_TEXT)
    fig.text(0.5, 1 - M_TOP / 2 - 0.035,
             "Accuracy and error rates by acquisition group at the default threshold",
             ha="center", va="center", fontsize=FS_SUBTITLE, color=C_SUBTEXT)

    # ---- Top: grouped bar chart ------------------------------------------
    ax = fig.add_axes([M_LEFT + 0.04, 0.42, 1 - M_LEFT - M_RIGHT - 0.08, 0.40])
    xs = np.arange(len(group_order))
    bar_w = 0.78 / max(len(metric_cols), 1)
    max_h = 0.0
    for j, (col, lbl, color) in enumerate(metric_cols):
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0).values
        offs = (j - (len(metric_cols) - 1) / 2) * bar_w
        bars = ax.bar(xs + offs, vals, bar_w, color=color, alpha=0.92,
                       edgecolor="white", linewidth=1.2, label=lbl)
        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(xs[i] + offs, v + 1.5, f"{float(v):.1f}",
                         ha="center", va="bottom",
                         fontsize=FS_NOTE, color=C_DARK_TEXT, fontweight="bold")
        if len(vals):
            max_h = max(max_h, float(np.nanmax(vals)))
    ax.set_xticks(xs)
    ax.set_xticklabels(group_order, fontsize=FS_TICK, color=C_TEXT)
    ax.set_ylabel("Rate (%)", fontsize=FS_AXIS_LABEL, color=C_TEXT)
    ax.set_ylim(0, max((max_h or 1) * 1.18, 5))
    ax.grid(axis="y", alpha=0.25, color=C_BORDER)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color(C_BORDER)
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(axis="y", labelsize=FS_TICK, colors=C_SUBTEXT)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10),
                     ncol=len(metric_cols), frameon=False, fontsize=FS_NOTE)
    for txt in leg.get_texts():
        txt.set_color(C_TEXT)

    # ---- Bottom: compact metrics table -----------------------------------
    show = df.copy()
    keep = ["group"]
    for c in ("n_total", "n_registered", "n_stranger",
               "Accuracy(%)", "FAR(%)", "FRR(%)", "IDR(%)",
               "TP", "TN", "FA", "FR", "MIS"):
        if c in show.columns:
            keep.append(c)
    show = show[keep]
    for c in show.columns:
        if c == "group":
            continue
        if pd.api.types.is_float_dtype(show[c]):
            show[c] = show[c].apply(lambda v: "-" if pd.isna(v) else f"{float(v):.2f}")
        elif pd.api.types.is_integer_dtype(show[c]):
            show[c] = show[c].apply(lambda v: f"{int(v):,}")
    show.columns = [c.replace("(%)", " (%)").replace("_", " ").title()
                     for c in show.columns]

    ax_tbl = fig.add_axes([M_LEFT + 0.02, 0.08, 1 - M_LEFT - M_RIGHT - 0.04, 0.28])
    ax_tbl.axis("off")
    # Shorten column names so they fit in narrow cells (12 columns wide)
    _SHORT = {
        "Group": "Group",
        "N Total": "N",
        "N Registered": "N Reg",
        "N Stranger": "N Strg",
        "Accuracy (%)": "Acc (%)",
        "Far (%)": "FAR (%)",
        "Frr (%)": "FRR (%)",
        "Idr (%)": "IDR (%)",
        "Tp": "TP", "Tn": "TN", "Fa": "FA",
        "Fr": "FR", "Mis": "MIS",
    }
    short_cols = [_SHORT.get(c, c) for c in show.columns]
    n_rows = len(show)
    n_cols = len(show.columns)
    if n_rows > 0 and n_cols > 0:
        cell_w = 1.0 / n_cols
        row_h  = 1.0 / (n_rows + 1)
        # Header
        for j, col_name in enumerate(short_cols):
            ax_tbl.add_patch(plt.Rectangle((j * cell_w, 1 - row_h),
                                            cell_w, row_h,
                                            facecolor=C_HEADER,
                                            edgecolor="white", linewidth=1.0,
                                            transform=ax_tbl.transAxes))
            ax_tbl.text(j * cell_w + cell_w / 2, 1 - row_h / 2,
                          str(col_name), ha="center", va="center",
                          fontsize=FS_NOTE, fontweight="bold", color="white",
                          transform=ax_tbl.transAxes)
        # Body
        orig_cols = list(show.columns)
        for i, (_, r) in enumerate(show.iterrows()):
            y = 1 - (i + 2) * row_h
            face = C_BG_ALT if i % 2 == 1 else "white"
            for j, col_name in enumerate(orig_cols):
                ax_tbl.add_patch(plt.Rectangle((j * cell_w, y),
                                                cell_w, row_h,
                                                facecolor=face,
                                                edgecolor=C_BORDER, linewidth=0.5,
                                                transform=ax_tbl.transAxes))
                ax_tbl.text(j * cell_w + cell_w / 2, y + row_h / 2,
                              str(r[col_name]), ha="center", va="center",
                              fontsize=FS_NOTE, color=C_DARK_TEXT,
                              transform=ax_tbl.transAxes)

    fig.text(0.5, 0.03,
              "Comparing rates across groups reveals which acquisition condition the recogniser handles best. "
              "Mask frames typically show elevated FRR; spoof frames test cross-pipeline robustness when anti-spoof fails.",
              ha="center", va="center", fontsize=FS_NOTE,
              color=C_SUBTEXT, style="italic")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Render figures for the face-recognition report.")
    parser.add_argument(
        "--single", choices=["default", "optimal"], default=None,
        help=("Also render ONE confusion matrix for the final / production report "
              "(use after threshold has been frozen on a dev set). "
              "'optimal' = use the sweep-optimal threshold; 'default' = use 0.55."))
    parser.add_argument(
        "--single-only", action="store_true",
        help="Skip the dev figures and render only the single confusion matrix.")
    args = parser.parse_args()

    if not os.path.exists(REPORT_XLSX):
        print(f"[ERROR] {REPORT_XLSX} not found. Run test_face_recognition_v2.py first.")
        sys.exit(1)
    os.makedirs(IMG_DIR, exist_ok=True)

    xw = pd.ExcelFile(REPORT_XLSX)
    def _read(name):
        return pd.read_excel(xw, name) if name in xw.sheet_names else pd.DataFrame()
    df_video      = _read("Per Video")
    df_ident      = _read("Per Identity")
    df_cm_default = _read("Confusion @ Default")
    df_cm_optimal = _read("Confusion @ Optimal")
    df_cm_full    = _read("Confusion Full @ Default")
    df_sweep      = _read("Threshold Sweep")
    df_summary    = _read("Summary")
    df_log        = _read("Frame Log")
    df_manifest   = _read("Dataset Manifest")
    df_per_group  = _read("Per Group")

    if not args.single_only:
        export_dataset_composition (df_manifest,
                                     os.path.join(IMG_DIR, "29a_dataset_composition.png"))
        export_split_overview      (df_manifest,
                                     os.path.join(IMG_DIR, "29b_split_overview.png"))
        export_per_video_table     (df_video,
                                     os.path.join(IMG_DIR, "30_summary_table.png"))
        export_identity_table      (df_ident,
                                     os.path.join(IMG_DIR, "31_group_summary_table.png"))
        export_per_group_breakdown (df_per_group, df_summary,
                                     os.path.join(IMG_DIR, "31b_per_group_breakdown.png"))
        export_metrics_table       (df_summary,
                                     os.path.join(IMG_DIR, "32_metrics_table.png"))
        export_threshold_sweep     (df_sweep, df_summary,
                                     os.path.join(IMG_DIR, "33_threshold_sweep.png"))
        export_distance_distribution(df_log, df_summary,
                                     os.path.join(IMG_DIR, "34_distance_distribution.png"))
        export_confusion_binary    (df_cm_default, df_cm_optimal, df_summary,
                                     os.path.join(IMG_DIR, "35a_confusion_binary.png"))
        if not df_cm_full.empty:
            export_confusion_full  (df_cm_full, df_summary,
                                     os.path.join(IMG_DIR, "35b_confusion_full.png"))
        export_infographic         (df_video, df_ident, df_sweep, df_summary, df_log,
                                     os.path.join(IMG_DIR, "36_infographic.png"))
        export_sample_frames       (SAMPLE_ROOT, df_video,
                                     os.path.join(IMG_DIR, "37_sample_frames.png"))
        export_confusion_samples_fig(SAMPLE_ROOT,
                                     os.path.join(IMG_DIR, "38_confusion_samples.png"))

    if args.single:
        op_label = args.single
        df_cm = df_cm_default if op_label == "default" else df_cm_optimal
        thr_txt = ""
        if not df_summary.empty:
            r = df_summary[df_summary["operating_point"] == op_label]
            if not r.empty:
                t = r["threshold"].iloc[0]
                if not pd.isna(t):
                    thr_txt = f"(threshold = {float(t):.2f})"
        out_path = os.path.join(IMG_DIR, "39_confusion_matrix_single.png")
        export_single_confusion(df_cm, df_summary, op_label, out_path, thr_txt)
        print(f"[OK] Single confusion matrix -> {out_path}")

    print(f"\n[OK] Face-recognition figures exported to: {IMG_DIR}/")


if __name__ == "__main__":
    main()
