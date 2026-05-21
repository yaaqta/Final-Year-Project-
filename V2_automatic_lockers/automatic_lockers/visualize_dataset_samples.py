"""
visualize_dataset_samples.py
════════════════════════════
Generates a publication-ready figure showing one representative sample frame
from each video in test_videos/, arranged in a labelled grid by dataset group:

    Group 1 – Normal (user_*_normal)
    Group 2 – Mask   (mask_*)
    Group 3 – Spoof  (spoof_*_phone  +  spoof_*_print, shown side-by-side)

For each video the script:
  1. Opens the clip with OpenCV.
  2. Tries the middle frame first; if YOLO finds no face there it scans
     forward / backward in steps of 10 frames and takes the first frame
     that has a face.  Falls back to the raw middle frame if no face is found.
  3. Draws the YOLO bounding box (if found) and a small label strip.

Outputs (in results/reportimages/):
    00a_dataset_normal.png
    00b_dataset_mask.png
    00c_dataset_spoof.png
    00d_dataset_overview.png   ← combined 4-row poster

Usage:
    python visualize_dataset_samples.py
    python visualize_dataset_samples.py --videodir my_clips --outdir my_out

Dependencies: same as the rest of the project (ultralytics, opencv-python,
matplotlib).  YOLO model path is resolved the same way as app.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import glob
from typing import List, Optional, Tuple

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch

# ── resolve project root so we can re-use app.py's YOLO model ─────────────────
THISDIR = os.path.dirname(os.path.abspath(__file__))
if THISDIR not in sys.path:
    sys.path.insert(0, THISDIR)

# ── style constants (shared with visualize_results.py) ────────────────────────
CBLUE    = "#5BA8E8"
CORANGE  = "#F4A263"
CPINK    = "#F18FB1"
CGREEN   = "#6FCB91"
CPURPLE  = "#A88BE3"
CTEAL    = "#5FCFC9"
CYELLOW  = "#F0C24B"
CHEADER  = "#6B8FB3"
CTEXT    = "#2C3E50"
CSUBTEXT = "#7B8794"
CBGALT   = "#F7F9FC"
CBORDER  = "#D8DEE5"

FSTITLE    = 18
FSSUBTITLE = 11
FSSECTION  = 13
FSBODY     = 11
FSNOTE     = 9.5
FSTICK     = 10

MLEFT = MRIGHT = MTOP = MBOTTOM = 0.06

# Group colour / label mapping
GROUP_CFG = {
    "normal": {
        "color": CBLUE,
        "label": "Normal\n(user_*_normal)",
        "title": "Group 1 – Normal (Bona Fide, No Mask)",
        "subtitle": "Cooperative users presenting an uncovered face to the camera",
    },
    "mask": {
        "color": CORANGE,
        "label": "Mask\n(mask_*)",
        "title": "Group 2 – Masked Bona Fide (mask_*)",
        "subtitle": "Same registered users wearing a surgical / cloth face mask",
    },
    "spoof_phone": {
        "color": CPINK,
        "label": "Spoof – Phone\n(spoof_*_phone)",
        "title": "Group 3a – Replay Attack (spoof_*_phone)",
        "subtitle": "Genuine face displayed on a smartphone screen held toward the camera",
    },
    "spoof_print": {
        "color": CPURPLE,
        "label": "Spoof – Print\n(spoof_*_print)",
        "title": "Group 3b – Print Attack (spoof_*_print)",
        "subtitle": "Printed photograph of a genuine face presented to the camera",
    },
}


# ── helpers ────────────────────────────────────────────────────────────────────

def apply_base_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": FSBODY,
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.facecolor": "white",
    })


def hex_to_rgb(h: str) -> Tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


def tint(hex_color: str, alpha: float = 0.12) -> Tuple[float, float, float]:
    r, g, b = hex_to_rgb(hex_color)
    return (1 - alpha + alpha * r, 1 - alpha + alpha * g, 1 - alpha + alpha * b)


def draw_card(ax, x, y, w, h, stroke, fill=None, radius=0.025, lw=1.6):
    if fill is None:
        fill = tint(stroke, 0.10)
    inset = radius * 0.6
    rect = FancyBboxPatch(
        (x + inset, y + inset),
        max(w - 2 * inset, 0.01),
        max(h - 2 * inset, 0.01),
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=stroke, facecolor=fill,
        transform=ax.transAxes, zorder=2, clip_on=False,
    )
    ax.add_patch(rect)


def classify_video(basename: str) -> Optional[str]:
    """Return group key or None if unrecognised."""
    b = os.path.basename(basename).lower()
    if b.startswith("user_") and "_normal" in b:
        return "normal"
    if b.startswith("mask_"):
        return "mask"
    if b.startswith("spoof_"):
        if "_phone" in b:
            return "spoof_phone"
        if "_print" in b:
            return "spoof_print"
        return "spoof_phone"          # fallback for plain spoof_*
    return None


def identity_from_basename(basename: str) -> str:
    """Extract the person name from the filename."""
    b = os.path.splitext(os.path.basename(basename))[0]  # strip .mp4
    # Patterns: user_<name>_normal, mask_<name>, spoof_<name>_phone/print
    parts = b.split("_")
    if b.startswith("user_") and b.endswith("_normal"):
        return "_".join(parts[1:-1])
    if b.startswith("mask_"):
        return "_".join(parts[1:])
    if b.startswith("spoof_"):
        # spoof_<name>_phone / spoof_<name>_print
        if parts[-1] in ("phone", "print"):
            return "_".join(parts[1:-1])
        return "_".join(parts[1:])
    return b


def load_yolo(model_path: str):
    """Load YOLOv12n-face (or any ultralytics model)."""
    from ultralytics import YOLO          # noqa: PLC0415
    return YOLO(model_path)


def detect_face(yolo_model, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return (x1,y1,x2,y2) of highest-confidence face, or None."""
    if yolo_model is None:
        return None
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = yolo_model(rgb, verbose=False)
    if not results:
        return None
    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    confs = res.boxes.conf.cpu().numpy()
    xyxy  = res.boxes.xyxy.cpu().numpy()
    i = int(np.argmax(confs))
    x1, y1, x2, y2 = xyxy[i]
    return int(x1), int(y1), int(x2), int(y2)


def pick_representative_frame(
    video_path: str,
    yolo_model,
    scan_radius: int = 80,
    scan_step: int = 10,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Return (frame_rgb, bbox_or_None).
    Tries the middle frame, then scans ±scan_radius frames in scan_step jumps.
    Falls back to the middle frame if no face is found anywhere.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None, None

    mid = total // 2
    candidates = [mid]
    for d in range(scan_step, scan_radius + 1, scan_step):
        if mid + d < total:
            candidates.append(mid + d)
        if mid - d >= 0:
            candidates.append(mid - d)

    best_frame, best_box = None, None
    for idx in candidates:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_bgr = cap.read()
        if not ret:
            continue
        if best_frame is None:
            best_frame = frame_bgr          # keep as fallback
        box = detect_face(yolo_model, frame_bgr)
        if box is not None:
            best_frame = frame_bgr
            best_box   = box
            break

    cap.release()
    if best_frame is None:
        return None, None
    rgb = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
    return rgb, best_box


def annotate_frame(
    frame_rgb: np.ndarray,
    box: Optional[Tuple[int, int, int, int]],
    color_hex: str,
    label: str,
) -> np.ndarray:
    """Draw bounding box and identity label on the frame (returns a copy)."""
    img = frame_rgb.copy()
    color_bgr = tuple(int(c * 255) for c in reversed(hex_to_rgb(color_hex)))

    if box is not None:
        x1, y1, x2, y2 = box
        thick = max(2, img.shape[1] // 220)
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, thick)

    # Label strip at top-left
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, img.shape[1] / 900)
    font_thick = max(1, int(font_scale * 1.8))
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)
    pad = max(3, int(font_scale * 4))
    x0, y0 = 6, 6
    cv2.rectangle(img, (x0, y0), (x0 + tw + pad * 2, y0 + th + pad * 2),
                  color_bgr, -1)
    cv2.putText(img, label, (x0 + pad, y0 + th + pad),
                font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)
    return img


def make_uniform_thumbnail(frame_rgb: np.ndarray, target_w: int = 640, target_h: int = 420) -> np.ndarray:
    """
    Convert arbitrary video frames into a uniform landscape thumbnail suitable
    for thesis grids. The full frame is preserved as much as possible using a
    contain-fit on a light canvas, so samples look consistent without becoming
    tiny portrait slivers.
    """
    h, w = frame_rgb.shape[:2]
    canvas = np.full((target_h, target_w, 3), 247, dtype=np.uint8)
    if h <= 0 or w <= 0:
        return canvas
    scale = min(target_w / w, target_h / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    ox = (target_w - nw) // 2
    oy = (target_h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = img
    return canvas


# ── per-group grid figure ──────────────────────────────────────────────────────

def build_group_figure(
    group_key: str,
    samples: List[Tuple[str, np.ndarray, Optional[Tuple]]],
    ncols: int = 4,
) -> plt.Figure:
    """
    Build a clean academic grid with wide, uniform image panels.
    Each sample uses the same thumbnail area and the same label band below.
    """
    cfg = GROUP_CFG[group_key]
    color = cfg["color"]
    n = len(samples)
    if n == 0:
        return None

    two_row_groups = {"normal", "mask", "spoof_phone"}
    if group_key in two_row_groups and n > 0:
        nrows = 2
        ncols = (n + 1) // 2
    else:
        if n <= 4:
            ncols = n
        elif n <= 8:
            ncols = 4
        else:
            ncols = 4
        nrows = (n + ncols - 1) // ncols

    apply_base_style()
    fig_w = max(15.2, 2.7 * ncols + 1.8)
    fig_h = 2.95 * nrows + 1.8
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.968, cfg["title"],
             ha="center", va="top", fontsize=FSTITLE, fontweight="bold", color=CTEXT)
    fig.text(0.5, 0.93, cfg["subtitle"],
             ha="center", va="top", fontsize=FSSUBTITLE, color=CSUBTEXT, style="italic")

    GRID_TOP = 0.86
    GRID_BOT = 0.08
    GRID_LEFT = 0.045
    GRID_RIGHT = 0.045
    GRID_W = 1.0 - GRID_LEFT - GRID_RIGHT
    GRID_H = GRID_TOP - GRID_BOT

    gap_x = 0.012
    gap_y = 0.04
    cell_w = (GRID_W - gap_x * (ncols - 1)) / ncols
    cell_h = (GRID_H - gap_y * (nrows - 1)) / nrows

    for idx, (identity, frame_rgb, box) in enumerate(samples):
        row = idx // ncols
        col = idx % ncols
        x0 = GRID_LEFT + col * (cell_w + gap_x)
        y0 = GRID_TOP - (row + 1) * cell_h - row * gap_y

        card = fig.add_axes([x0, y0, cell_w, cell_h])
        card.set_xlim(0, 1)
        card.set_ylim(0, 1)
        card.axis("off")

        # image region (dominant)
        img_ax = fig.add_axes([x0, y0 + 0.105 * cell_h, cell_w, 0.79 * cell_h])
        img_ax.axis("off")
        annotated = annotate_frame(frame_rgb, box, color, identity.upper())
        thumb = make_uniform_thumbnail(annotated, target_w=640, target_h=420)
        img_ax.imshow(thumb)
        for sp in img_ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(CBORDER)
            sp.set_linewidth(0.9)

        # bottom name band
        band = fig.add_axes([x0, y0, cell_w, 0.095 * cell_h])
        band.set_xlim(0, 1)
        band.set_ylim(0, 1)
        band.axis("off")
        band.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='none', transform=band.transAxes))
        band.text(0.5, 0.42, identity, ha='center', va='center',
                  fontsize=FSBODY + 0.2, color=CTEXT, fontweight='bold', transform=band.transAxes)

    fig.text(0.5, 0.02,
             f"{n} videos · one representative frame each · standardised panel size",
             ha="center", va="bottom", fontsize=FSNOTE - 0.5, color=CSUBTEXT, style="italic")
    return fig


# ── combined overview poster ───────────────────────────────────────────────────

def build_overview_figure(
    all_groups: dict,   # group_key → list of (identity, frame_rgb, box)
) -> plt.Figure:
    """
    4-row poster: Normal / Mask / Spoof-Phone / Spoof-Print.
    Each row shows up to OVERVIEW_COLS thumbnail cells.
    """
    OVERVIEW_COLS = 6
    CELL_W = 2.2
    CELL_H = 1.9
    row_order = ["normal", "mask", "spoof_phone", "spoof_print"]

    apply_base_style()
    nrows = len(row_order)
    FIG_W = OVERVIEW_COLS * CELL_W + 2.0
    FIG_H = nrows * CELL_H + 2.2

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.985, "Dataset Overview – Sample Frames by Group",
             ha="center", va="top",
             fontsize=FSTITLE + 1, fontweight="bold", color=CTEXT)
    fig.text(0.5, 0.960,
             "One representative frame per video · test_videos/ · "
             "bounding box from YOLOv12n-face",
             ha="center", va="top",
             fontsize=FSSUBTITLE, color=CSUBTEXT, style="italic")

    LABEL_W  = 0.09      # fraction of figure width for row label
    GRID_LEFT = LABEL_W + 0.01
    GRID_W    = 1.0 - GRID_LEFT - 0.01
    GRID_TOP  = 0.940
    GRID_BOT  = 0.018
    total_h   = GRID_TOP - GRID_BOT
    row_h     = total_h / nrows
    cell_w    = GRID_W / OVERVIEW_COLS

    for ri, gk in enumerate(row_order):
        cfg      = GROUP_CFG[gk]
        color    = cfg["color"]
        samples  = all_groups.get(gk, [])[:OVERVIEW_COLS]
        y_row    = GRID_BOT + (nrows - 1 - ri) * row_h

        # ── row label chip ───────────────────────────────────────────────────
        axl = fig.add_axes([0.002, y_row + 0.005, LABEL_W - 0.012, row_h - 0.01])
        axl.axis("off")
        draw_card(axl, 0.0, 0.0, 1.0, 1.0, stroke=color,
                  fill=tint(color, 0.18), radius=0.08, lw=1.4)
        axl.text(0.5, 0.5, cfg["label"],
                 ha="center", va="center",
                 fontsize=FSNOTE, fontweight="bold", color=color,
                 transform=axl.transAxes, multialignment="center")

        # ── cell images ──────────────────────────────────────────────────────
        for ci in range(OVERVIEW_COLS):
            x0 = GRID_LEFT + ci * cell_w + 0.003
            PAD = 0.004
            ax = fig.add_axes([x0 + PAD, y_row + 0.03,
                               cell_w - 2*PAD, row_h - 0.05])
            ax.axis("off")

            if ci < len(samples):
                identity, frame_rgb, box = samples[ci]
                annotated = annotate_frame(frame_rgb, box, color,
                                           identity.upper()[:10])
                ax.imshow(annotated)
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_edgecolor(color)
                    sp.set_linewidth(1.4)
                # tiny identity label
                fig.text(
                    x0 + cell_w / 2,
                    y_row + 0.012,
                    identity,
                    ha="center", va="bottom",
                    fontsize=max(6.5, FSNOTE - 1.5), color=CTEXT,
                )
            else:
                ax.set_facecolor(CBGALT)
                ax.text(0.5, 0.5, "–", ha="center", va="center",
                        fontsize=FSNOTE, color=CSUBTEXT,
                        transform=ax.transAxes)

    return fig


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise dataset sample frames by group for thesis Chapter 4."
    )
    parser.add_argument("--videodir", default="test_videos",
                        help="Folder containing test video clips")
    parser.add_argument("--outdir",   default=os.path.join("results", "reportimages"),
                        help="Output folder for PNG figures")
    parser.add_argument("--model",    default=None,
                        help="Path to YOLOv12n-face.pt (auto-detected if omitted)")
    parser.add_argument("--ncols",    type=int, default=4,
                        help="Number of columns in each group grid")
    parser.add_argument("--no-yolo",  action="store_true",
                        help="Skip face detection (no bounding boxes)")
    args = parser.parse_args()

    # ── locate videos ─────────────────────────────────────────────────────────
    video_dir = args.videodir
    if not os.path.isdir(video_dir):
        print(f"ERROR: video directory not found: {video_dir}")
        sys.exit(1)

    patterns = ["*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.AVI"]
    all_videos = []
    for pat in patterns:
        all_videos.extend(glob.glob(os.path.join(video_dir, pat)))
    all_videos = sorted(set(all_videos))
    if not all_videos:
        print(f"ERROR: no video files found in {video_dir}")
        sys.exit(1)
    print(f"INFO: found {len(all_videos)} videos in {video_dir}")

    # ── load YOLO (optional) ──────────────────────────────────────────────────
    yolo_model = None
    if not args.no_yolo:
        model_candidates = [
            args.model or "",
            os.path.join(THISDIR, "yolov12n-face.pt"),
            "yolov12n-face.pt",
        ]
        for mp in model_candidates:
            if mp and os.path.isfile(mp):
                print(f"INFO: loading YOLO from {mp}")
                try:
                    yolo_model = load_yolo(mp)
                    print("INFO: YOLO ready")
                except Exception as e:
                    print(f"WARN: could not load YOLO ({e}) – running without bbox")
                    yolo_model = None
                break
        if yolo_model is None:
            print("WARN: YOLO model not found – frames will be shown without bounding boxes")

    # ── classify and extract frames ───────────────────────────────────────────
    groups: dict[str, list] = {k: [] for k in GROUP_CFG}

    for vpath in all_videos:
        gk = classify_video(vpath)
        if gk is None:
            print(f"  SKIP (unrecognised prefix): {os.path.basename(vpath)}")
            continue

        identity = identity_from_basename(vpath)
        print(f"  → {gk:12s}  {identity:20s}  {os.path.basename(vpath)}")

        frame_rgb, box = pick_representative_frame(vpath, yolo_model)
        if frame_rgb is None:
            print(f"    WARN: could not read frames from {vpath}")
            continue

        groups[gk].append((identity, frame_rgb, box))

    os.makedirs(args.outdir, exist_ok=True)

    # ── per-group figures ─────────────────────────────────────────────────────
    group_file_map = {
        "normal":      "00a_dataset_normal.png",
        "mask":        "00b_dataset_mask.png",
        "spoof_phone": "00c_dataset_spoof_phone.png",
        "spoof_print": "00d_dataset_spoof_print.png",
    }

    for gk, fname in group_file_map.items():
        samples = groups[gk]
        if not samples:
            print(f"  SKIP (no videos): group={gk}")
            continue
        fig = build_group_figure(gk, samples, ncols=args.ncols)
        if fig is None:
            continue
        out_path = os.path.join(args.outdir, fname)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  OK  {fname}  ({len(samples)} videos)")

    # ── combined poster ───────────────────────────────────────────────────────
    total = sum(len(v) for v in groups.values())
    if total > 0:
        fig_ov = build_overview_figure(groups)
        ov_path = os.path.join(args.outdir, "00e_dataset_overview.png")
        fig_ov.savefig(ov_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig_ov)
        print(f"  OK  00e_dataset_overview.png  (combined poster, {total} clips)")

    print(f"\nDone. Figures saved to: {args.outdir}")


if __name__ == "__main__":
    main()
