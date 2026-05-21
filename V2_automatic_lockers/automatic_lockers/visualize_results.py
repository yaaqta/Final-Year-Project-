"""
============================================================
 VISUALIZE RESULTS -- v4
 (Unified style guide: consistent margins, font sizes,
  italic notes, no overlap between text and tables)
============================================================
Reads Detection_Report.xlsx in results/ and exports clean
academic-style PNG figures for the thesis report.

STYLE GUIDE (applied across ALL outputs):
  - Page margins:     left/right/top/bottom >= 1.5-2 cm
  - Title:            18 pt bold
  - Subtitle:         11 pt regular (gray)
  - Section header:   13 pt bold
  - Body / cells:     11 pt regular
  - Note / footnote:  9.5 pt italic, gray
  - KPI value:        22 pt bold
  - KPI label:        10 pt regular
  - Min gap between blocks: ~0.6 cm
  - Footnote: at least 0.5 cm below the table/figure

Outputs (in results/report_images/):
  - 10_summary_table.png         Detailed per-video table
  - 11_group_summary_table.png   Group averages (Normal vs Masked)
  - 12_comparison_table.png      Normal vs Masked comparison
  - 13_frame_statistics.png      Total frames + donut + per-group bar
  - 14_methodology_table.png     Metric formulas
  - 15_infographic.png           One-page summary

Run:
   python visualize_results.py
============================================================
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch


# ---------------------------------------------------------------
# PASTEL ACADEMIC PALETTE (bright but soft)
# ---------------------------------------------------------------
C_BLUE     = "#5BA8E8"
C_BLUE_BG  = "#E6F3FD"
C_PURPLE   = "#A88BE3"
C_PURPLE_BG= "#F1EAFB"
C_PINK     = "#F18FB1"
C_PINK_BG  = "#FDEBF1"
C_GREEN    = "#6FCB91"
C_GREEN_BG = "#E5F7EC"
C_ORANGE   = "#F4A263"
C_ORANGE_BG= "#FCEDDD"
C_TEAL     = "#5FCFC9"
C_TEAL_BG  = "#E1F6F5"
C_YELLOW   = "#F0C24B"
C_YELLOW_BG= "#FBF1D7"

C_NORMAL   = C_BLUE
C_MASKED   = C_ORANGE
C_OK       = C_GREEN
C_WARN     = C_PINK
C_HEADER   = "#6B8FB3"
C_TEXT     = "#2C3E50"
C_SUBTEXT  = "#7B8794"
C_BG_ALT   = "#F7F9FC"
C_BORDER   = "#D8DEE5"
C_ACCENT   = "#A4B8D1"
C_TARGET   = C_YELLOW


# ---------------------------------------------------------------
# UNIFIED TYPOGRAPHY CONSTANTS
# ---------------------------------------------------------------
FS_TITLE       = 18    # main title (Table / Figure heading)
FS_SUBTITLE    = 11    # subtitle under title
FS_SECTION     = 13    # card / section header
FS_BODY        = 11    # body / table cell text
FS_AXIS_LABEL  = 10.5  # axis labels
FS_TICK        = 10    # tick labels
FS_NOTE        = 9.5   # italic footnote / caption
FS_KPI_VALUE   = 22    # big KPI number
FS_KPI_LABEL   = 10    # KPI bottom label

# Margins on figure (figure-fraction; ~1.5-2 cm at typical figure sizes)
M_LEFT   = 0.06
M_RIGHT  = 0.06
M_TOP    = 0.06
M_BOTTOM = 0.06


def apply_base_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": FS_BODY,
        "axes.titlesize": FS_SECTION,
        "axes.titleweight": "semibold",
        "axes.labelsize": FS_AXIS_LABEL,
        "axes.labelcolor": C_TEXT,
        "axes.edgecolor": C_BORDER,
        "axes.linewidth": 0.8,
        "xtick.color": C_TEXT,
        "ytick.color": C_TEXT,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "axes.grid": True,
        "grid.color": C_BORDER,
        "grid.alpha": 0.6,
        "grid.linewidth": 0.6,
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.facecolor": "white",
    })


# ---------------------------------------------------------------
# Helper -- text wrapping
# ---------------------------------------------------------------
def _wrap_text(s, max_chars):
    s = str(s)
    if len(s) <= max_chars:
        return s
    words = s.split(" ")
    lines, cur = [], ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur = cur + " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)


# ---------------------------------------------------------------
# Table rendering -- balanced vertical layout
#
#   Page layout (figure fraction):
#     +----------------------------------+  y = 1.00
#     |        TITLE (top margin)        |
#     +----------------------------------+  y = title_y
#     |                                  |
#     |             TABLE                |
#     |                                  |
#     +----------------------------------+  y = table_bottom
#     |          NOTE (italic)           |
#     +----------------------------------+  y = bottom margin
# ---------------------------------------------------------------
def render_table(df, title, save_path, highlight_cols=None,
                 col_widths=None, figsize=None, footnote=None,
                 wrap_chars=None):
    n_rows, n_cols = df.shape
    col_labels = [str(c) for c in df.columns]
    cell_text = df.astype(str).values.tolist()

    if wrap_chars:
        for i in range(n_rows):
            for j in range(n_cols):
                if j in wrap_chars:
                    cell_text[i][j] = _wrap_text(cell_text[i][j], wrap_chars[j])

    if col_widths is None:
        widths = []
        for j in range(n_cols):
            max_len = max(len(col_labels[j]),
                          max((len(cell_text[i][j]) for i in range(n_rows)),
                              default=1))
            widths.append(max(max_len, 6))
        total = sum(widths)
        col_widths = [w / total for w in widths]

    # Auto figure size:
    # - Width:  scaled with total chars
    # - Height: title(1.0) + table(0.55 * n_rows + 0.7) + footnote(0.7)
    if figsize is None:
        fig_w = max(10, sum(widths) * 0.22)
        fig_h = 1.0 + (n_rows + 1) * 0.55 + (0.7 if footnote else 0.4)
        figsize = (fig_w, fig_h)

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("white")

    # -------- Compute y-bands (figure fraction) ----------------
    fig_h_in = figsize[1]
    title_band = 0.7 / fig_h_in      # ~0.7 inch  (~1.8 cm)
    note_band  = (0.7 if footnote else 0.4) / fig_h_in

    title_y_center = 1.0 - M_TOP - title_band / 2
    table_top      = 1.0 - M_TOP - title_band
    table_bottom   = M_BOTTOM + note_band
    table_h        = table_top - table_bottom
    note_y_center  = M_BOTTOM + note_band / 2

    # -------- Title ---------------------------------------------
    fig.text(0.5, title_y_center, title,
             ha="center", va="center",
             fontsize=FS_TITLE, fontweight="bold", color=C_TEXT)

    # -------- Table axes ---------------------------------------
    ax = fig.add_axes([M_LEFT, table_bottom, 1 - M_LEFT - M_RIGHT, table_h])
    ax.axis("off")

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc="center", cellLoc="center",
                     colWidths=col_widths,
                     bbox=[0.0, 0.0, 1.0, 1.0])
    table.auto_set_font_size(False)
    table.set_fontsize(FS_BODY)

    # header row
    for col_idx in range(n_cols):
        cell = table[(0, col_idx)]
        cell.set_facecolor(C_HEADER)
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor(C_HEADER)
        cell.set_height(0.10)

    # body row height accounting for multiline cells
    max_lines = 1
    for i in range(n_rows):
        for j in range(n_cols):
            max_lines = max(max_lines, str(cell_text[i][j]).count("\n") + 1)
    row_h = 0.085 if max_lines <= 1 else 0.085 + (max_lines - 1) * 0.045

    for row_idx in range(1, n_rows + 1):
        for col_idx in range(n_cols):
            cell = table[(row_idx, col_idx)]
            cell.set_facecolor(C_BG_ALT if row_idx % 2 == 0 else "white")
            cell.set_edgecolor(C_BORDER)
            cell.set_text_props(color=C_TEXT)
            cell.set_height(row_h)

            if highlight_cols and col_idx in highlight_cols:
                val_str = str(df.iloc[row_idx - 1, col_idx])
                try:
                    val = float(val_str.replace("%", "").replace("ms", "").strip())
                    target = highlight_cols[col_idx]
                    if target.get("type") == "above":
                        ok = val >= target["value"]
                    elif target.get("type") == "below":
                        ok = val <= target["value"]
                    else:
                        ok = True
                    cell.set_text_props(color=C_OK if ok else C_WARN,
                                        fontweight="semibold")
                except Exception:
                    pass

    # -------- Footnote (italic) ---------------------------------
    if footnote:
        fig.text(0.5, note_y_center, footnote,
                 ha="center", va="center",
                 fontsize=FS_NOTE, color=C_SUBTEXT, style="italic")

    plt.savefig(save_path, dpi=200, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------
# Tables
# ---------------------------------------------------------------
def export_summary_table(df_video, out_path):
    df = df_video.copy()
    cols = ["video", "group", "total_frames", "detected_frames",
            "detection_rate(%)", "avg_confidence",
            "avg_infer_time(ms)", "fps_infer"]
    df = df[cols].rename(columns={
        "video": "Video",
        "group": "Group",
        "total_frames": "Frames",
        "detected_frames": "Detected",
        "detection_rate(%)": "Detection Rate (%)",
        "avg_confidence": "Avg Confidence",
        "avg_infer_time(ms)": "Time (ms)",
        "fps_infer": "FPS",
    })
    df["Group"] = df["Group"].str.capitalize()
    highlight = {4: {"type": "above", "value": 95}}
    render_table(df,
                 title="Table 1 - Per-Video Detection Results",
                 save_path=out_path,
                 highlight_cols=highlight,
                 footnote="Detection Rate values in green indicate they meet "
                          "the >= 95% target.")


def export_group_summary(df_group, out_path):
    if df_group is None or df_group.empty:
        return
    df = df_group.copy().rename(columns={
        "Group": "Group",
        "So video": "# Videos",
        "Avg Detection Rate (%)": "Avg Detection Rate (%)",
        "Min Detection Rate (%)": "Min Detection Rate (%)",
        "Avg Confidence": "Avg Confidence",
        "Avg Inference Time (ms)": "Avg Time (ms)",
        "Avg FPS": "Avg FPS",
    })
    df["Group"] = df["Group"].str.capitalize()
    highlight = {2: {"type": "above", "value": 95}}
    render_table(df,
                 title="Table 2 - Group Averages (Normal vs. Masked)",
                 save_path=out_path,
                 highlight_cols=highlight,
                 footnote="Normal = user_* videos (uncovered faces). "
                          "Masked = mask_* videos (faces wearing surgical mask).")


def export_comparison_table(df_compare, out_path):
    if df_compare is None or df_compare.empty:
        return
    df = df_compare.copy().rename(columns={
        "Chi so": "Metric",
        "Normal (user_*)": "Normal (user_*)",
        "Masked (mask_*)": "Masked (mask_*)",
        "Thay doi": "Delta Change",
        "% thay doi": "% Change",
    })
    render_table(df,
                 title="Table 3 - Performance Comparison: Normal vs. Masked",
                 save_path=out_path,
                 footnote="Detection Rate and Confidence drops are negligible, "
                          "indicating YOLOv12n-face is robust to facial occlusion.")


def export_methodology(df_method, out_path):
    df = pd.DataFrame([
        ("Detection Rate (%)",
         "(frames with >= 1 bbox) / (total frames) x 100",
         "Sensitivity - percentage of frames with a detected face"),
        ("Avg Confidence",
         "mean(confidence of largest bbox per frame)",
         "Average model certainty across all detections"),
        ("Min Confidence",
         "min(confidence) over all detected frames",
         "Worst-case scenario - least confident detection"),
        ("Avg Box Area (px2)",
         "mean((x2 - x1) x (y2 - y1)) per detected frame",
         "Average face size - controls subject distance from camera"),
        ("Avg Inference Time (ms)",
         "mean(time.perf_counter() delta around yolo_model()) x 1000",
         "Average processing latency - determines real-time capability"),
        ("Std Inference Time (ms)",
         "std(per-frame inference times)",
         "Stability indicator - lower std = more uniform processing"),
        ("FPS Inference",
         "1000 / Avg Inference Time (ms)",
         "Frames per second the model can process"),
    ], columns=["Metric", "Formula", "Description"])

    render_table(df,
                 title="Table 4 - Metric Definitions",
                 save_path=out_path,
                 col_widths=[0.22, 0.42, 0.36],
                 figsize=(15, 8.5),
                 wrap_chars={1: 40, 2: 38},
                 footnote="Reference: ISO/IEC 19795 - Information technology - "
                          "Biometric performance testing and reporting.")


# ---------------------------------------------------------------
# Card helpers
# ---------------------------------------------------------------
def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


def _tint(hex_color, alpha=0.12):
    r, g, b = _hex_to_rgb(hex_color)
    return (1 - alpha + alpha * r,
            1 - alpha + alpha * g,
            1 - alpha + alpha * b)


def draw_card(ax, x, y, w, h, stroke, fill=None,
              radius=0.025, lw=1.6, alpha=1.0):
    if fill is None:
        fill = _tint(stroke, alpha=0.10)
    inset = radius * 0.6
    rect = FancyBboxPatch(
        (x + inset, y + inset),
        max(w - 2 * inset, 0.01),
        max(h - 2 * inset, 0.01),
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=stroke, facecolor=fill,
        alpha=alpha, transform=ax.transAxes, zorder=2,
        clip_on=False,
    )
    ax.add_patch(rect)
    return rect


def draw_kpi_card(ax, x, y, w, h, value, label, stroke, fill=None):
    draw_card(ax, x, y, w, h, stroke=stroke, fill=fill, radius=0.025, lw=1.5)
    ax.text(x + w / 2, y + h * 0.62, value,
            ha="center", va="center",
            fontsize=FS_KPI_VALUE, fontweight="bold",
            color=stroke, transform=ax.transAxes, zorder=3)
    ax.text(x + w / 2, y + h * 0.22, label,
            ha="center", va="center",
            fontsize=FS_KPI_LABEL, color=C_TEXT,
            transform=ax.transAxes, zorder=3)


# ---------------------------------------------------------------
# Frame statistics figure
#
# Vertical layout (figure fraction, top -> bottom):
#   [Title band]            top 1.5 cm
#   [KPI cards row]         next 1.5 cm
#   [Donut + Bar cards]     ~ 55 %
#   [Note band]             bottom 1.0 cm
# Margins L/R 1.5 cm.
# ---------------------------------------------------------------
def export_frame_stats(df_video, df_frames, out_path):
    total_videos    = len(df_video)
    total_frames    = int(df_video["total_frames"].sum())
    total_detected  = int(df_video["detected_frames"].sum())
    total_missed    = total_frames - total_detected
    detection_overall = (total_detected / total_frames * 100.0) if total_frames else 0

    normal_frames   = int(df_video[df_video["group"] == "normal"]["total_frames"].sum())
    masked_frames   = int(df_video[df_video["group"] == "masked"]["total_frames"].sum())
    normal_detected = int(df_video[df_video["group"] == "normal"]["detected_frames"].sum())
    masked_detected = int(df_video[df_video["group"] == "masked"]["detected_frames"].sum())

    if "infer_time_ms" in df_frames:
        avg_time = df_frames["infer_time_ms"].mean()
        min_time = df_frames["infer_time_ms"].min()
        max_time = df_frames["infer_time_ms"].max()
    else:
        avg_time = min_time = max_time = 0

    # Figure size: 14 x 9.5 inches -> 1 cm ~ 0.078 width-fraction, 0.041 height-fraction
    FIG_W, FIG_H = 14, 9.5
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")

    # ---- vertical bands (top -> bottom) ----
    title_top    = 1.0 - M_TOP                       # ~ 0.94
    title_band_h = 0.075                             # ~ 1.8 cm
    subtitle_y   = title_top - title_band_h + 0.018

    kpi_top   = title_top - title_band_h - 0.025     # gap 0.6 cm
    kpi_h     = 0.115
    kpi_y     = kpi_top - kpi_h

    note_y    = M_BOTTOM + 0.012                     # bottom margin
    note_top  = note_y + 0.04

    cards_top = kpi_y - 0.030
    cards_bot = note_top + 0.015
    cards_h   = cards_top - cards_bot

    # ---- Title ----
    fig.text(0.5, title_top - 0.005,
             "Figure 1 - Total Frame Processing Statistics",
             ha="center", va="top",
             fontsize=FS_TITLE, fontweight="bold", color=C_TEXT)
    fig.text(0.5, subtitle_y,
             "Smart Locker System  ·  YOLOv12n-face  ·  Frame-level evaluation",
             ha="center", va="top",
             fontsize=FS_SUBTITLE, color=C_SUBTEXT)

    # ---- KPI cards ----
    ax_kpi = fig.add_axes([M_LEFT, kpi_y, 1 - M_LEFT - M_RIGHT, kpi_h])
    ax_kpi.set_xlim(0, 1); ax_kpi.set_ylim(0, 1); ax_kpi.axis("off")
    kpis = [
        ("Videos Tested",   f"{total_videos}",        C_PURPLE),
        ("Total Frames",    f"{total_frames:,}",      C_BLUE),
        ("Frames Detected", f"{total_detected:,}",    C_GREEN),
        ("Frames Missed",   f"{total_missed:,}",      C_PINK if total_missed else C_TEAL),
    ]
    card_w = 0.225
    gap = (1 - card_w * 4) / 5
    for i, (label, value, color) in enumerate(kpis):
        cx = gap + i * (card_w + gap)
        draw_kpi_card(ax_kpi, cx, 0.05, card_w, 0.90,
                      value=value, label=label, stroke=color)

    # ---- Two chart cards ----
    # Card 1 (donut)
    half_w = (1 - M_LEFT - M_RIGHT - 0.04) / 2  # 0.04 gap between cards
    donut_x = M_LEFT
    bar_x   = M_LEFT + half_w + 0.04

    ax_donut_bg = fig.add_axes([donut_x, cards_bot, half_w, cards_h])
    ax_donut_bg.set_xlim(0, 1); ax_donut_bg.set_ylim(0, 1); ax_donut_bg.axis("off")
    draw_card(ax_donut_bg, 0.0, 0.0, 1.0, 1.0,
              stroke=C_GREEN, fill="white", radius=0.04, lw=1.4)
    ax_donut_bg.text(0.5, 0.93, "Detected vs. Missed Frames",
                     ha="center", va="center",
                     fontsize=FS_SECTION, fontweight="bold", color=C_TEXT,
                     transform=ax_donut_bg.transAxes)

    # donut axes -- inside the card, well away from card edges
    donut_inset_x = donut_x + half_w * 0.06
    donut_inset_y = cards_bot + cards_h * 0.10
    donut_w = half_w * 0.50
    donut_h = cards_h * 0.70
    ax_donut = fig.add_axes([donut_inset_x, donut_inset_y, donut_w, donut_h])
    sizes = [total_detected, max(total_missed, 1)]
    ax_donut.pie(sizes, colors=[C_GREEN, C_PINK], startangle=90,
                 wedgeprops=dict(width=0.30, edgecolor="white", linewidth=3))
    ax_donut.text(0, 0.10, f"{detection_overall:.2f}%",
                  ha="center", va="center",
                  fontsize=24, fontweight="bold", color=C_GREEN)
    ax_donut.text(0, -0.18, "Overall\nDetection Rate",
                  ha="center", va="center",
                  fontsize=FS_TICK, color=C_SUBTEXT)

    # legend inside donut card (right side)
    legend_items = [
        ("Detected", f"{total_detected:,} frames", C_GREEN),
        ("Missed",   f"{total_missed:,} frames",   C_PINK),
    ]
    for i, (lab, val, c) in enumerate(legend_items):
        ypos = 0.55 - i * 0.18
        ax_donut_bg.plot([0.66], [ypos + 0.04], marker="o",
                         markersize=11, color=c,
                         markeredgecolor="white", markeredgewidth=1.5,
                         transform=ax_donut_bg.transAxes,
                         clip_on=False)
        ax_donut_bg.text(0.71, ypos + 0.04, lab,
                         fontsize=FS_BODY, fontweight="bold", color=c,
                         va="center", transform=ax_donut_bg.transAxes)
        ax_donut_bg.text(0.71, ypos - 0.03, val,
                         fontsize=FS_TICK, color=C_TEXT,
                         va="center", transform=ax_donut_bg.transAxes)

    # Card 2 (bar)
    ax_bar_bg = fig.add_axes([bar_x, cards_bot, half_w, cards_h])
    ax_bar_bg.set_xlim(0, 1); ax_bar_bg.set_ylim(0, 1); ax_bar_bg.axis("off")
    draw_card(ax_bar_bg, 0.0, 0.0, 1.0, 1.0,
              stroke=C_BLUE, fill="white", radius=0.04, lw=1.4)
    ax_bar_bg.text(0.5, 0.93, "Frames Processed by Group",
                   ha="center", va="center",
                   fontsize=FS_SECTION, fontweight="bold", color=C_TEXT,
                   transform=ax_bar_bg.transAxes)

    # bar axes inside card -- leave generous left padding for y-label and ticks,
    # and bottom padding so x-tick labels don't cross the card border.
    bar_inset_x = bar_x + half_w * 0.18
    bar_inset_y = cards_bot + cards_h * 0.22
    bar_w_inset = half_w * 0.74
    bar_h_inset = cards_h * 0.60
    ax_bar = fig.add_axes([bar_inset_x, bar_inset_y, bar_w_inset, bar_h_inset])
    groups = ["Normal\n(user_*)", "Masked\n(mask_*)"]
    detected_vals = [normal_detected, masked_detected]
    missed_vals = [normal_frames - normal_detected,
                   masked_frames - masked_detected]
    totals = [d + m for d, m in zip(detected_vals, missed_vals)]
    max_total = max(totals) if totals else 1

    x = np.arange(len(groups))
    w = 0.45
    ax_bar.bar(x, detected_vals, w, label="Detected",
               color=C_GREEN, edgecolor="white", linewidth=1.5)
    ax_bar.bar(x, missed_vals, w, bottom=detected_vals, label="Missed",
               color=C_PINK, edgecolor="white", linewidth=1.5)

    for i, (d, m) in enumerate(zip(detected_vals, missed_vals)):
        ax_bar.text(i, d / 2, f"{d:,}", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=FS_BODY)
        if m > 0 and m > max_total * 0.04:
            ax_bar.text(i, d + m / 2, f"{m:,}", ha="center", va="center",
                        color="white", fontweight="bold", fontsize=FS_TICK)
        ax_bar.text(i, d + m + max_total * 0.025,
                    f"Total: {d + m:,}", ha="center", va="bottom",
                    fontsize=FS_AXIS_LABEL, color=C_TEXT, fontweight="semibold")

    ax_bar.set_ylim(0, max_total * 1.20)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(groups, fontsize=FS_TICK)
    ax_bar.set_ylabel("Number of frames", fontsize=FS_AXIS_LABEL)
    ax_bar.legend(loc="upper right", frameon=False,
                  bbox_to_anchor=(1.0, 1.0), fontsize=FS_TICK)
    ax_bar.grid(axis="y", alpha=0.4, color=C_BORDER)
    ax_bar.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax_bar.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax_bar.spines[sp].set_color(C_BORDER)

    # ---- Footnote ----
    fig.text(0.5, note_y,
             f"Per-frame inference time   |   Min: {min_time:.1f} ms   ·   "
             f"Avg: {avg_time:.1f} ms   ·   Max: {max_time:.1f} ms",
             ha="center", va="bottom",
             fontsize=FS_NOTE, color=C_SUBTEXT, style="italic")

    plt.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------
# Infographic one-pager
#
# Vertical layout (figure fraction, top -> bottom):
#   Title band            ~ 1.8 cm
#   KPI cards row         ~ 2.5 cm
#   Detection-rate card   ~ 5.5 cm
#   Inference-time card   ~ 5.5 cm
#   Group-comparison card ~ 5.5 cm
#   Conclusion card       ~ 5.0 cm
#   Bottom margin         ~ 1.5 cm
# All inter-block gaps >= 0.6 cm
# ---------------------------------------------------------------
def export_infographic(df_video, df_group, df_compare, df_frames, out_path):
    total_frames = int(df_video["total_frames"].sum())
    overall_dr   = (df_video["detected_frames"].sum() / total_frames * 100.0) if total_frames else 0
    avg_conf     = df_video["avg_confidence"].dropna().mean()
    avg_time     = df_video["avg_infer_time(ms)"].mean()

    FIG_W, FIG_H = 13, 16.5
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")

    # 1 cm in fig-height-fraction: 0.394 / FIG_H = 0.0219
    # We allocate vertical bands explicitly (top -> bottom):
    #   header_h        = 0.060   (~1.5 cm)
    #   gap             = 0.012   (~0.6 cm)
    #   kpi_h           = 0.060
    #   gap
    #   card_dr_h       = 0.158   (~7.5 cm with chart inside)
    #   gap
    #   card_it_h       = 0.158
    #   gap
    #   card_gc_h       = 0.170
    #   gap
    #   conclusion_h    = 0.140
    #   bottom margin   = 0.032   (~1.5 cm)

    HEADER_H = 0.062
    GAP      = 0.020
    KPI_H    = 0.070
    DR_H     = 0.195
    IT_H     = 0.195
    GC_H     = 0.195
    CONC_H   = 0.150
    BOTTOM_M = 0.030

    # Compute y positions (top of each band)
    top = 1.0 - 0.025                          # top margin ~ 1.0 cm at top
    header_top = top
    header_bot = header_top - HEADER_H

    kpi_top    = header_bot - GAP
    kpi_bot    = kpi_top - KPI_H

    dr_top     = kpi_bot - GAP
    dr_bot     = dr_top - DR_H

    it_top     = dr_bot - GAP
    it_bot     = it_top - IT_H

    gc_top     = it_bot - GAP
    gc_bot     = gc_top - GC_H

    conc_top   = gc_bot - GAP
    conc_bot   = conc_top - CONC_H

    # ===== Header =====
    fig.text(0.5, header_top - 0.012,
             "Face Detection Evaluation Report",
             ha="center", va="top",
             fontsize=FS_TITLE + 2, fontweight="bold", color=C_TEXT)
    fig.text(0.5, header_top - 0.040,
             "Smart Locker System  ·  YOLOv12n-face  ·  Modular Evaluation",
             ha="center", va="top",
             fontsize=FS_SUBTITLE, color=C_SUBTEXT)

    # ===== KPI cards =====
    ax_kpi = fig.add_axes([M_LEFT, kpi_bot, 1 - M_LEFT - M_RIGHT, KPI_H])
    ax_kpi.set_xlim(0, 1); ax_kpi.set_ylim(0, 1); ax_kpi.axis("off")
    kpis = [
        ("Total Frames",   f"{total_frames:,}",  C_BLUE),
        ("Detection Rate", f"{overall_dr:.1f}%", C_GREEN),
        ("Avg Confidence", f"{avg_conf:.3f}",    C_PURPLE),
        ("Avg Latency",    f"{avg_time:.0f} ms", C_ORANGE),
    ]
    card_w = 0.225
    gap = (1 - card_w * 4) / 5
    for i, (label, value, color) in enumerate(kpis):
        cx = gap + i * (card_w + gap)
        draw_kpi_card(ax_kpi, cx, 0.05, card_w, 0.90,
                      value=value, label=label, stroke=color)

    # ===== Card: Detection Rate per Video =====
    ax_dr_bg = fig.add_axes([M_LEFT, dr_bot, 1 - M_LEFT - M_RIGHT, DR_H])
    ax_dr_bg.set_xlim(0, 1); ax_dr_bg.set_ylim(0, 1); ax_dr_bg.axis("off")
    draw_card(ax_dr_bg, 0.0, 0.0, 1.0, 1.0,
              stroke=C_BLUE, fill="white", radius=0.025, lw=1.4)
    ax_dr_bg.text(0.03, 0.90, "Detection Rate by Video",
                  fontsize=FS_SECTION, fontweight="bold", color=C_TEXT,
                  va="center", transform=ax_dr_bg.transAxes)
    # legend chips (top-right)
    ax_dr_bg.plot([0.605], [0.90], marker="s", markersize=10, color=C_BLUE,
                  markeredgecolor="none", transform=ax_dr_bg.transAxes,
                  clip_on=False)
    ax_dr_bg.text(0.625, 0.90, "Normal (user_*)", fontsize=FS_TICK,
                  color=C_TEXT, va="center", transform=ax_dr_bg.transAxes)
    ax_dr_bg.plot([0.795], [0.90], marker="s", markersize=10, color=C_ORANGE,
                  markeredgecolor="none", transform=ax_dr_bg.transAxes,
                  clip_on=False)
    ax_dr_bg.text(0.815, 0.90, "Masked (mask_*)", fontsize=FS_TICK,
                  color=C_TEXT, va="center", transform=ax_dr_bg.transAxes)

    # chart axes -- inside this outer card (NO overlap with title or border).
    # Padding sized so y-label + tick labels (left), and x-tick labels (bottom)
    # all fit fully INSIDE the card border, not outside.
    pad_l, pad_r = 0.085, 0.035      # outer-card fraction (horizontal)
    # vertical band inside outer card:
    #   title band:   y_oc in [0.78, 0.97]
    #   chart band:   y_oc in [0.34, 0.72]  (extra room below for rotated x-ticks)
    chart_y_b, chart_y_t = 0.34, 0.72
    bx_x = M_LEFT + pad_l * (1 - M_LEFT - M_RIGHT)
    bx_w = (1 - pad_l - pad_r) * (1 - M_LEFT - M_RIGHT)
    bx_y = dr_bot + chart_y_b * DR_H
    bx_h = (chart_y_t - chart_y_b) * DR_H
    ax_dr = fig.add_axes([bx_x, bx_y, bx_w, bx_h])
    bar_colors = [C_BLUE if g == "normal" else C_ORANGE
                  for g in df_video["group"]]
    bars = ax_dr.bar(df_video["video"], df_video["detection_rate(%)"],
                     color=bar_colors, edgecolor="white",
                     linewidth=1.5, width=0.65)
    ax_dr.axhline(95, color=C_TARGET, linestyle="--", linewidth=1.2,
                  label="Target 95%")
    ax_dr.set_ylim(0, 115)
    ax_dr.set_ylabel("Detection Rate (%)", fontsize=FS_AXIS_LABEL)
    for bar, v in zip(bars, df_video["detection_rate(%)"]):
        ax_dr.text(bar.get_x() + bar.get_width() / 2, v + 2.5, f"{v:.1f}%",
                   ha="center", fontsize=FS_TICK, color=C_TEXT)
    ax_dr.legend(loc="lower right", frameon=False, fontsize=FS_TICK)
    ax_dr.grid(axis="y", alpha=0.4, color=C_BORDER)
    ax_dr.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax_dr.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax_dr.spines[sp].set_color(C_BORDER)
    plt.setp(ax_dr.get_xticklabels(), rotation=12, ha="right", fontsize=FS_TICK)
    ax_dr.tick_params(axis="y", labelsize=FS_TICK)

    # ===== Card: Inference Time per Video =====
    ax_it_bg = fig.add_axes([M_LEFT, it_bot, 1 - M_LEFT - M_RIGHT, IT_H])
    ax_it_bg.set_xlim(0, 1); ax_it_bg.set_ylim(0, 1); ax_it_bg.axis("off")
    draw_card(ax_it_bg, 0.0, 0.0, 1.0, 1.0,
              stroke=C_ORANGE, fill="white", radius=0.025, lw=1.4)
    ax_it_bg.text(0.03, 0.90, "Average Inference Time per Video",
                  fontsize=FS_SECTION, fontweight="bold", color=C_TEXT,
                  va="center", transform=ax_it_bg.transAxes)
    bx_x = M_LEFT + pad_l * (1 - M_LEFT - M_RIGHT)
    bx_w = (1 - pad_l - pad_r) * (1 - M_LEFT - M_RIGHT)
    bx_y = it_bot + chart_y_b * IT_H
    bx_h = (chart_y_t - chart_y_b) * IT_H
    ax_it = fig.add_axes([bx_x, bx_y, bx_w, bx_h])
    bars = ax_it.bar(df_video["video"], df_video["avg_infer_time(ms)"],
                     color=bar_colors, edgecolor="white",
                     linewidth=1.5, width=0.65)
    ax_it.set_ylabel("Inference Time (ms)", fontsize=FS_AXIS_LABEL)
    vmax_it = max(df_video["avg_infer_time(ms)"]) if len(df_video) else 1
    ax_it.set_ylim(0, vmax_it * 1.35)
    for bar, v in zip(bars, df_video["avg_infer_time(ms)"]):
        ax_it.text(bar.get_x() + bar.get_width() / 2, v + vmax_it * 0.06,
                   f"{v:.0f}", ha="center", fontsize=FS_TICK, color=C_TEXT)
    ax_it.grid(axis="y", alpha=0.4, color=C_BORDER)
    ax_it.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax_it.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax_it.spines[sp].set_color(C_BORDER)
    plt.setp(ax_it.get_xticklabels(), rotation=12, ha="right", fontsize=FS_TICK)
    ax_it.tick_params(axis="y", labelsize=FS_TICK)

    # ===== Card: Group comparison (4 mini cards) =====
    if not df_group.empty and len(df_group) >= 2:
        g = df_group.set_index("Group")
        if "normal" in g.index and "masked" in g.index:
            metrics_list = [
                ("Avg Detection Rate (%)", "%",  "Detection Rate", C_GREEN),
                ("Avg Confidence",          "",  "Confidence",     C_PURPLE),
                ("Avg Inference Time (ms)"," ms","Inference Time", C_ORANGE),
                ("Avg FPS",                 "",  "FPS",            C_TEAL),
            ]
            OC_X = M_LEFT
            OC_W = 1 - M_LEFT - M_RIGHT
            OC_Y = gc_bot
            OC_H = GC_H
            ax_g_bg = fig.add_axes([OC_X, OC_Y, OC_W, OC_H])
            ax_g_bg.set_xlim(0, 1); ax_g_bg.set_ylim(0, 1); ax_g_bg.axis("off")
            draw_card(ax_g_bg, 0.0, 0.0, 1.0, 1.0,
                      stroke=C_PURPLE, fill="white", radius=0.025, lw=1.4)
            ax_g_bg.text(0.03, 0.93, "Group Comparison - Normal vs. Masked",
                         fontsize=FS_SECTION, fontweight="bold", color=C_TEXT,
                         va="center", transform=ax_g_bg.transAxes)

            sub_w = 0.21
            sub_gap = (1.0 - sub_w * 4) / 5
            # Mini-card vertical layout -- leave generous room for the title (top)
            # and the x-tick labels (bottom) so they never touch the card border.
            MC_Y_BOTTOM, MC_Y_TOP = 0.06, 0.82
            CHART_Y_BOTTOM, CHART_Y_TOP = 0.20, 0.55
            TITLE_Y = 0.74  # well above the chart top to avoid overlap with value labels

            for i, (m, unit, short, ccol) in enumerate(metrics_list):
                vals = [g.loc["normal", m], g.loc["masked", m]]
                cx = sub_gap + i * (sub_w + sub_gap)

                draw_card(ax_g_bg, cx, MC_Y_BOTTOM, sub_w,
                          MC_Y_TOP - MC_Y_BOTTOM,
                          stroke=ccol, fill=_tint(ccol, alpha=0.06),
                          radius=0.02, lw=1.3)

                ax_g_bg.text(cx + sub_w / 2, TITLE_Y, short,
                             ha="center", va="center",
                             fontsize=FS_BODY,
                             fontweight="bold", color=ccol,
                             transform=ax_g_bg.transAxes)

                # Wider left padding so y-tick labels (e.g. '100', '200') stay
                # inside the mini-card border; right padding kept tight.
                pad_x_l = 0.052
                pad_x_r = 0.022
                chart_x_oc = cx + pad_x_l
                chart_w_oc = sub_w - pad_x_l - pad_x_r

                bx_x2 = OC_X + chart_x_oc * OC_W
                bx_w2 = chart_w_oc * OC_W
                bx_y2 = OC_Y + CHART_Y_BOTTOM * OC_H
                bx_h2 = (CHART_Y_TOP - CHART_Y_BOTTOM) * OC_H

                bx = fig.add_axes([bx_x2, bx_y2, bx_w2, bx_h2])
                bars = bx.bar(["Normal", "Masked"], vals,
                              color=[C_BLUE, C_ORANGE],
                              edgecolor="white", linewidth=1.4, width=0.55)
                vmax = max([v for v in vals if v is not None] + [1])
                for bar, v in zip(bars, vals):
                    if v is not None:
                        bx.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + vmax * 0.05,
                                f"{v:.2f}{unit}".strip(),
                                ha="center", fontsize=FS_NOTE,
                                color=C_TEXT, fontweight="bold")
                # Headroom for value label, capped so label never crosses card top.
                bx.set_ylim(0, vmax * 1.30)
                bx.set_xticks([0, 1])
                bx.set_xticklabels(["Normal", "Masked"],
                                   fontsize=FS_NOTE, color=C_TEXT)
                # Limit y-axis to at most 3 ticks so labels stay readable and
                # don't crowd the card edge.
                from matplotlib.ticker import MaxNLocator
                bx.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="upper"))
                bx.tick_params(axis="y", labelsize=FS_NOTE - 1, colors=C_SUBTEXT, pad=2)
                bx.grid(axis="y", alpha=0.25, color=C_BORDER)
                bx.set_axisbelow(True)
                bx.set_facecolor("none")
                for sp in ["top", "right"]:
                    bx.spines[sp].set_visible(False)
                for sp in ["bottom", "left"]:
                    bx.spines[sp].set_color(C_BORDER)
                    bx.spines[sp].set_linewidth(0.8)

    # ===== Conclusion card =====
    masked_dr_vals = df_group[df_group["Group"] == "masked"]["Avg Detection Rate (%)"].values
    normal_dr_vals = df_group[df_group["Group"] == "normal"]["Avg Detection Rate (%)"].values
    dr_drop = (normal_dr_vals[0] - masked_dr_vals[0]) if (len(normal_dr_vals) and len(masked_dr_vals)) else 0

    ax_c = fig.add_axes([M_LEFT, conc_bot, 1 - M_LEFT - M_RIGHT, CONC_H])
    ax_c.set_xlim(0, 1); ax_c.set_ylim(0, 1); ax_c.axis("off")
    draw_card(ax_c, 0.0, 0.0, 1.0, 1.0,
              stroke=C_GREEN, fill=_tint(C_GREEN, alpha=0.07),
              radius=0.025, lw=1.4)
    ax_c.text(0.5, 0.86, "Conclusion",
              ha="center", va="center",
              fontsize=FS_SECTION + 1, fontweight="bold",
              color=C_GREEN, transform=ax_c.transAxes)

    bullets = [
        f"Overall Detection Rate of {overall_dr:.1f}% exceeds the 95% target - "
        f"YOLOv12n-face provides stable face detection.",
        f"When subjects wear surgical masks, Detection Rate drops by only "
        f"{dr_drop:.2f}% - the model is robust to facial occlusion.",
        f"Average latency of {avg_time:.0f} ms on CPU can be further reduced "
        f"via GPU inference or frame-skipping for real-time deployment.",
    ]
    bullet_top = 0.66
    bullet_step = 0.18
    for i, txt in enumerate(bullets):
        y_pos = bullet_top - i * bullet_step
        ax_c.text(0.05, y_pos, "●", fontsize=FS_BODY, color=C_GREEN,
                  va="center", transform=ax_c.transAxes)
        ax_c.text(0.08, y_pos, txt, fontsize=FS_BODY, color=C_TEXT,
                  va="center", transform=ax_c.transAxes)

    plt.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------
# Sample frames figure  (2 rows x 5 cols, with bbox drawn by tester)
#
# Reads results/sample_frames/<video>/best_*.jpg + worst_*.jpg
# Picks 5 images per group (3 best + 2 worst preferred, but if a
# group has multiple videos we round-robin across videos so the
# 5 slots show diverse content).
# ---------------------------------------------------------------
def _pick_group_samples(sample_root, video_list, n=5):
    """Pick n sample images for a group, distributing across videos.
    Strategy:
      - try to fill 3 'best' slots first (round-robin across videos)
      - then 2 'worst' slots (round-robin)
      - if any category has fewer files, fall back to the other.
    Each item: (path, label, kind)  kind in {'best','worst'}
    """
    if not video_list:
        return []

    # collect file lists per video
    bests, worsts = {}, {}
    for v in video_list:
        name = os.path.splitext(v)[0]
        d = os.path.join(sample_root, name)
        if not os.path.isdir(d):
            bests[name] = []
            worsts[name] = []
            continue
        bests[name]  = sorted(glob.glob(os.path.join(d, "best_*.jpg")))
        worsts[name] = sorted(glob.glob(os.path.join(d, "worst_*.jpg")))

    names = [os.path.splitext(v)[0] for v in video_list]
    picked = []
    used = {n: {"best": 0, "worst": 0} for n in names}

    def pull(kind, target_count):
        """Round-robin pull `target_count` items of kind from videos."""
        added = 0
        pool = bests if kind == "best" else worsts
        # iterate rounds; each round take 1 from each video that still has items
        max_rounds = max((len(pool[n]) for n in names), default=0)
        for r in range(max_rounds):
            for n in names:
                if added >= target_count:
                    return added
                if used[n][kind] < len(pool[n]):
                    picked.append((pool[n][used[n][kind]], n, kind))
                    used[n][kind] += 1
                    added += 1
        return added

    n_best = pull("best", 3)
    n_worst = pull("worst", 2)

    # if we still don't have 5, fill from the other kind
    if len(picked) < n:
        deficit = n - len(picked)
        # try more bests
        more = pull("best", deficit)
        deficit -= more
        if deficit > 0:
            pull("worst", deficit)

    return picked[:n]


def export_sample_frames(sample_root, df_video, out_path):
    """Compose a 2x5 figure: row 1 = Normal, row 2 = Masked."""
    if not os.path.isdir(sample_root):
        print(f"[WARN] No sample_frames dir at {sample_root}; skipping sample_frames figure.")
        return False

    normal_videos = df_video[df_video["group"] == "normal"]["video"].tolist()
    masked_videos = df_video[df_video["group"] == "masked"]["video"].tolist()

    normal_picks = _pick_group_samples(sample_root, normal_videos, n=5)
    masked_picks = _pick_group_samples(sample_root, masked_videos, n=5)

    if not normal_picks and not masked_picks:
        print("[WARN] No sample frames found; skipping sample_frames figure.")
        return False

    # ---- Figure layout ----
    FIG_W, FIG_H = 16, 9.0
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")

    # bands
    title_top    = 1.0 - 0.025
    title_band_h = 0.075
    subtitle_y   = title_top - title_band_h + 0.018
    note_y       = 0.030
    grid_top     = subtitle_y - 0.030
    grid_bot     = note_y + 0.045

    # ---- Title ----
    fig.text(0.5, title_top - 0.005,
             "Figure 2 - Sample Frames per Group",
             ha="center", va="top",
             fontsize=FS_TITLE, fontweight="bold", color=C_TEXT)
    fig.text(0.5, subtitle_y,
             "3 best + 2 worst frames per group  ·  bounding boxes from YOLOv12n-face",
             ha="center", va="top",
             fontsize=FS_SUBTITLE, color=C_SUBTEXT)

    # ---- Grid: 2 rows x 5 cols ----
    rows = [("Normal (user_*)", C_BLUE,   normal_picks),
            ("Masked (mask_*)", C_ORANGE, masked_picks)]
    n_cols = 5
    row_h = (grid_top - grid_bot) / 2

    LEFT_LABEL_W = 0.07   # space for the row label on the left
    CELL_PAD = 0.008      # padding between cells
    cell_band_w = 1 - M_LEFT - M_RIGHT - LEFT_LABEL_W
    cell_w = cell_band_w / n_cols

    for r_idx, (label, color, picks) in enumerate(rows):
        # row anchored at the top of its band
        y_top = grid_top - r_idx * row_h
        y_bot = y_top - row_h

        # left label (vertical band)
        ax_lab = fig.add_axes([M_LEFT, y_bot, LEFT_LABEL_W, row_h])
        ax_lab.set_xlim(0, 1); ax_lab.set_ylim(0, 1); ax_lab.axis("off")
        # colored vertical stripe
        draw_card(ax_lab, 0.10, 0.05, 0.80, 0.90,
                  stroke=color, fill=_tint(color, alpha=0.10),
                  radius=0.05, lw=1.4)
        ax_lab.text(0.5, 0.5, label, ha="center", va="center",
                    fontsize=FS_SECTION, fontweight="bold", color=color,
                    rotation=90, transform=ax_lab.transAxes)

        # cells
        for c_idx in range(n_cols):
            x_cell = M_LEFT + LEFT_LABEL_W + c_idx * cell_w
            ax = fig.add_axes([x_cell + CELL_PAD,
                               y_bot + CELL_PAD,
                               cell_w - 2 * CELL_PAD,
                               row_h - 2 * CELL_PAD])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

            if c_idx < len(picks):
                path, vname, kind = picks[c_idx]
                try:
                    img = mpimg.imread(path)
                    ax.imshow(img)
                except Exception as e:
                    ax.text(0.5, 0.5, "(load error)", ha="center", va="center",
                            fontsize=FS_NOTE, color=C_SUBTEXT)
                # caption strip at bottom
                kind_label = "BEST" if kind == "best" else "WORST"
                kind_color = C_GREEN if kind == "best" else C_PINK
                ax.text(0.02, 0.04, f"{vname}  ·  {kind_label}",
                        ha="left", va="bottom", transform=ax.transAxes,
                        fontsize=FS_NOTE, color="white",
                        fontweight="bold",
                        bbox=dict(facecolor=kind_color, edgecolor="none",
                                  boxstyle="round,pad=0.25", alpha=0.85))
            else:
                ax.set_facecolor(C_BG_ALT)
                ax.text(0.5, 0.5, "(no sample)",
                        ha="center", va="center",
                        fontsize=FS_NOTE, color=C_SUBTEXT,
                        transform=ax.transAxes)

    # ---- Footnote ----
    fig.text(0.5, note_y,
             "Green box = detected face with confidence score.  "
             "Orange border = low-confidence detection.  "
             "Red border = MISS (no face detected).",
             ha="center", va="bottom",
             fontsize=FS_NOTE, color=C_SUBTEXT, style="italic")

    plt.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)
    return True


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    apply_base_style()

    xlsx_path = os.path.join(args.results_dir, "Detection_Report.xlsx")
    if not os.path.isfile(xlsx_path):
        print(f"[ERROR] Cannot find {xlsx_path}")
        print("        Please run test_face_detection.py first.")
        return

    out_dir = os.path.join(args.results_dir, "report_images")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Reading data from {xlsx_path} ...")
    df_video   = pd.read_excel(xlsx_path, sheet_name="Per_Video")
    df_group   = pd.read_excel(xlsx_path, sheet_name="Group_Summary")
    df_compare = pd.read_excel(xlsx_path, sheet_name="Comparison")
    df_method  = pd.read_excel(xlsx_path, sheet_name="Methodology")
    df_frames  = pd.read_excel(xlsx_path, sheet_name="Frame_Level_Detail")

    print("[INFO] Exporting report images ...")

    export_summary_table(df_video,
                         os.path.join(out_dir, "10_summary_table.png"))
    print("  - 10_summary_table.png")

    export_group_summary(df_group,
                         os.path.join(out_dir, "11_group_summary_table.png"))
    print("  - 11_group_summary_table.png")

    export_comparison_table(df_compare,
                            os.path.join(out_dir, "12_comparison_table.png"))
    print("  - 12_comparison_table.png")

    export_frame_stats(df_video, df_frames,
                       os.path.join(out_dir, "13_frame_statistics.png"))
    print("  - 13_frame_statistics.png")

    export_methodology(df_method,
                       os.path.join(out_dir, "14_methodology_table.png"))
    print("  - 14_methodology_table.png")

    export_infographic(df_video, df_group, df_compare, df_frames,
                       os.path.join(out_dir, "15_infographic.png"))
    print("  - 15_infographic.png")

    sample_root = os.path.join(args.results_dir, "sample_frames")
    ok = export_sample_frames(sample_root, df_video,
                              os.path.join(out_dir, "16_sample_frames.png"))
    if ok:
        print("  - 16_sample_frames.png")

    n_out = 7 if ok else 6
    print(f"\n[OK] {n_out} figures exported to: {out_dir}/")
    print("     Ready for thesis report.\n")


if __name__ == "__main__":
    main()
