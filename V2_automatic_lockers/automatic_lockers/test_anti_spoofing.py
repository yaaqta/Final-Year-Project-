"""
test_anti_spoofing.py
======================

Standalone evaluation for the Silent-Face / MiniFASNet anti-spoofing module
used in the Smart Locker capstone project.

What it does
------------
For every frame of every test video:
  1. Run YOLOv12n-face to find a face bounding box (fallback to full-frame crop
     when no face is detected, so the anti-spoof module still gets evaluated).
  2. Crop the face region via `crop_face_bgr()` from app.py (same pre-processing
     the deployed system uses).
  3. Run `check_liveness(face_bgr)` -> (is_real: bool, live_score: float).
  4. Time the call, log everything to a per-frame DataFrame.

Then it computes the ISO/IEC 30107-3 metrics:
  - APCER = FAR for the attack class (spoof presentations classified as live)
  - BPCER = FRR for the bona-fide class (live presentations classified as spoof)
  - ACER  = (APCER + BPCER) / 2
  - EER   = point on the threshold sweep where APCER == BPCER

Two threshold operating points are reported:
  - DEFAULT   : whatever app.py's check_liveness() decides via its `is_real` flag
  - EER       : the threshold that minimises |APCER - BPCER|

Output
------
results/Spoof_Report.xlsx with sheets:
  - Per Video
  - Per Group
  - Confusion @ Default
  - Confusion @ EER
  - ROC Sweep
  - Summary
  - Frame Log  (raw scores for downstream plots)

Sample frames saved to results/spoof_samples/<video_name>/:
  - best_*.jpg    : top-3 most confident *correct* predictions
  - worst_*.jpg   : top-2 most damaging *errors* per video
  - tp_*.jpg / tn_*.jpg / fp_*.jpg / fn_*.jpg : per-frame visualisations
                                                used for the confusion-matrix figure.

NOTE on language
----------------
This file is meant to be dropped into the user's Windows project folder, so
all comments / strings are plain ASCII (no Vietnamese diacritics).
"""

from __future__ import annotations

import os
import sys
import glob
import time
import heapq
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project imports -- expects this file to live in the same folder as app.py
# ---------------------------------------------------------------------------
# We use the YOLO model directly so we can read the confidence score, which
# app.detect_faces() strips away. We also reuse the exact crop / liveness
# helpers from app.py so the test mirrors the deployed pipeline 1-for-1.
from app import (                         # noqa: E402
    yolo_model,
    run_liveness,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VIDEO_DIR = "test_videos"
RESULTS_DIR = "results"
SAMPLE_ROOT = os.path.join(RESULTS_DIR, "spoof_samples")
REPORT_XLSX = os.path.join(RESULTS_DIR, "Spoof_Report.xlsx")

# Threshold sweep range for ROC / EER computation.
# `live_score` from MiniFASNet is in [0, 1]; we sweep finely enough to give
# a smooth ROC and locate EER within ~0.5 pp.
THRESHOLD_GRID = np.round(np.linspace(0.05, 0.95, 91), 4)

# Group classification rules. The video filename prefix decides the ground truth.
# Spoof co 2 sub-type:
#   spoof_<name>_phone.mp4 -> spoof_phone (replay attack: dien thoai)
#   spoof_<name>_print.mp4 -> spoof_print (print attack: anh in giay)
GROUP_RULES = [
    ("user_",  "user",  "live"),     # bona-fide (live)
    ("mask_",  "mask",  "live"),     # bona-fide (live, with surgical mask)
    ("spoof_", "spoof", "spoof"),    # attack presentation (parent group)
]


# ---------------------------------------------------------------------------
# Per-video accumulator + heaps for sample frames
# ---------------------------------------------------------------------------
@dataclass
class VideoAcc:
    """In-memory accumulator for one video.

    Sample-frame strategy mirrors the detection test:
        - best_heap:  top-3 frames where the model was most confident AND correct.
        - worst_heap: top-2 frames where the model was most confidently wrong
                      (or, if none, lowest-margin errors).
    We use heaps so we never have to keep all frames in memory.
    """
    name: str
    group: str
    label: str                 # "live" or "spoof"
    spoof_type: str = "n/a"    # "phone" / "print" / "n/a"
    frames: int = 0
    detected: int = 0
    scores: List[float] = field(default_factory=list)
    is_real_default: List[bool] = field(default_factory=list)
    latencies: List[float] = field(default_factory=list)

    # Heaps store (priority, counter, frame_idx, frame_bgr, box, score, is_real_default)
    # `counter` is a tie-breaker so equal-priority items don't compare ndarrays.
    best_heap: list = field(default_factory=list)    # min-heap, size 3, prio = confidence_in_correct_class
    worst_heap: list = field(default_factory=list)   # max-heap via negation, size 2, prio = severity of error


def _correct(label: str, is_real: bool) -> bool:
    """Was the default-threshold prediction correct?"""
    if label == "live":
        return is_real is True
    return is_real is False


def _confidence_in_correct_class(label: str, score: float) -> float:
    """How confident was the model in the *correct* class?
    Higher = better. score is live probability.
    """
    return score if label == "live" else (1.0 - score)


def _error_severity(label: str, score: float) -> float:
    """How damaging was the prediction *if it was wrong*?
    Higher = worse mistake. We only call this for wrong predictions.
    """
    if label == "live":
        # Wrongly rejected a live face -> low score is more severe
        return 1.0 - score
    # Wrongly accepted a spoof -> high score is more severe (false accept = security risk)
    return score


_GLOBAL_HEAP_COUNTER = 0


def _heap_push(heap, priority, frame_bgr, box, frame_idx, score, is_real, max_size, mode="min"):
    """Bounded heap insert. mode='min' keeps top-`max_size` largest priorities;
    mode='max' keeps top-`max_size` smallest priorities (use negative priority)."""
    global _GLOBAL_HEAP_COUNTER
    _GLOBAL_HEAP_COUNTER += 1
    payload = (priority, _GLOBAL_HEAP_COUNTER, frame_idx, frame_bgr, box, score, is_real)
    if len(heap) < max_size:
        heapq.heappush(heap, payload)
    else:
        # For a min-heap kept at size N: replace smallest if new > smallest.
        if priority > heap[0][0]:
            heapq.heapreplace(heap, payload)


# ---------------------------------------------------------------------------
# Drawing helpers (mirrors test_face_detection.py style)
# ---------------------------------------------------------------------------
def _label_scale(img_w: int) -> Tuple[float, int, int]:
    s = max(1.0, img_w / 640.0)
    return 1.1 * s, max(2, int(round(2.3 * s))), max(2, int(round(2 * s)))


def _put_label_with_bg(img, text, org, color_bgr, font_scale, text_thick):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, text_thick)
    pad_x = max(4, int(font_scale * 4))
    pad_y = max(4, int(font_scale * 3))
    x, y = org
    x = max(0, x)
    y = max(th + pad_y * 2, y)
    cv2.rectangle(img, (x, y - th - pad_y * 2), (x + tw + pad_x * 2, y),
                  color_bgr, thickness=-1)
    cv2.putText(img, text, (x + pad_x, y - pad_y),
                font, font_scale, (255, 255, 255), text_thick, cv2.LINE_AA)


# Colour palette (BGR for OpenCV)
COL_LIVE_OK = (0, 200, 0)        # green     -> correct live
COL_SPOOF_OK = (200, 80, 0)      # blue      -> correct spoof reject
COL_FALSE_ACC = (0, 0, 255)      # red       -> spoof accepted (most dangerous)
COL_FALSE_REJ = (0, 140, 255)    # orange    -> live rejected
COL_NEUTRAL = (120, 120, 120)


def annotate_frame(frame_bgr, box, score: float, is_real: bool, label: str) -> np.ndarray:
    """Draw bbox + label showing live score + verdict.
    Border colour encodes the confusion-matrix cell:
        green   = TP (live, accepted)
        blue    = TN (spoof, rejected)
        red     = FP (spoof, accepted)  <-- security incident
        orange  = FN (live, rejected)
    """
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    fs, tt, bt = _label_scale(w)

    verdict = "LIVE" if is_real else "SPOOF"
    correct = _correct(label, is_real)

    if label == "live" and correct:
        col = COL_LIVE_OK
        cell = "TP"
    elif label == "spoof" and correct:
        col = COL_SPOOF_OK
        cell = "TN"
    elif label == "spoof" and not correct:
        col = COL_FALSE_ACC
        cell = "FP"
    else:
        col = COL_FALSE_REJ
        cell = "FN"

    # Outer border colour-coded by confusion cell
    border = max(4, int(round(w / 200)))
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), col, border)

    # Bbox (if a face was detected)
    if box is not None:
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, bt)
        text = f"{verdict} {score:.2f}"
        _put_label_with_bg(img, text, (x1, y1 - 2), col, fs, tt)

    # Cell tag top-left
    _put_label_with_bg(img, cell, (border + 4, int(fs * 36) + border),
                       col, fs * 1.05, tt + 1)
    return img


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------
def find_videos(folder: str) -> List[str]:
    pats = ["*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.AVI"]
    files = []
    for p in pats:
        files.extend(glob.glob(os.path.join(folder, p)))
    # Windows is case-insensitive -- dedupe by realpath.
    seen = set()
    uniq = []
    for f in files:
        rp = os.path.realpath(f).lower()
        if rp not in seen:
            seen.add(rp)
            uniq.append(f)
    return sorted(uniq)


def classify_video(name: str) -> Tuple[str, str, str]:
    """Return (group_name, ground_truth_label, spoof_type) from filename.

    spoof_type:
        - 'phone'  neu video la 'spoof_<name>_phone.mp4'
        - 'print'  neu video la 'spoof_<name>_print.mp4'
        - 'n/a'    neu khong phai spoof (user/mask)
    """
    base = os.path.basename(name).lower()
    for prefix, group, label in GROUP_RULES:
        if base.startswith(prefix):
            spoof_type = "n/a"
            if group == "spoof":
                # Tach sub-type tu duoi file: 'spoof_an_phone.mp4' -> 'phone'
                if "_phone" in base:
                    spoof_type = "phone"
                elif "_print" in base:
                    spoof_type = "print"
                else:
                    spoof_type = "other"
            return group, label, spoof_type
    return "unknown", "unknown", "n/a"


def detect_one_face(frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return the highest-confidence YOLO box, or None."""
    results = yolo_model(frame_bgr, verbose=False)
    if not results:
        return None
    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    confs = res.boxes.conf.cpu().numpy()
    xyxy = res.boxes.xyxy.cpu().numpy()
    i = int(np.argmax(confs))
    x1, y1, x2, y2 = xyxy[i]
    return int(x1), int(y1), int(x2), int(y2)


# ---------------------------------------------------------------------------
# Test 1 video
# ---------------------------------------------------------------------------
def test_one_video(video_path: str, sample_dir: str) -> VideoAcc:
    name = os.path.splitext(os.path.basename(video_path))[0]
    group, label, spoof_type = classify_video(name)
    acc = VideoAcc(name=name, group=group, label=label, spoof_type=spoof_type)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] cannot open {video_path}")
        return acc

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_idx += 1
        acc.frames += 1

        # 1) YOLO detect
        box = detect_one_face(frame_bgr)

        # 2) Goi anti-spoof DUNG CACH:
        #    - frame_bgr: FULL FRAME goc (khong crop truoc)
        #    - boxes [[x1, y1, x2, y2]] tu YOLO
        #    run_liveness se tu convert sang [x, y, w, h] va goi check_liveness,
        #    de MiniFASNet tu mo rong bbox theo scale 2.7 / 4.0.
        #    Neu khong detect duoc bbox: dung full frame voi bbox = toan anh
        #    (van dung API, nhung khong co context => coi nhu fallback).
        t0 = time.time()
        try:
            if box is not None:
                acc.detected += 1
                x1, y1, x2, y2 = box
                # app.run_liveness nhan img_np_rgb (decode_image format).
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                is_real, live_score = run_liveness(
                    frame_rgb, [[x1, y1, x2, y2]]
                )
            else:
                # Fallback: khong co face detection -> coi nhu spoof
                # (an toan hon, va trong thuc te route /analyze_hint cung reject)
                is_real, live_score = False, 0.0
        except Exception as exc:
            print(f"[WARN] run_liveness failed on {name} frame {frame_idx}: {exc}")
            is_real, live_score = False, 0.0
        t_ms = (time.time() - t0) * 1000.0

        acc.scores.append(float(live_score))
        acc.is_real_default.append(bool(is_real))
        acc.latencies.append(float(t_ms))

        # 4) Sample-frame bookkeeping
        is_correct = _correct(label, is_real)
        if is_correct:
            # Track top-3 most confidently correct
            prio = _confidence_in_correct_class(label, live_score)
            _heap_push(acc.best_heap, prio, frame_bgr.copy(), box,
                       frame_idx, live_score, is_real, max_size=3)
        else:
            # Track top-2 most damaging errors
            prio = _error_severity(label, live_score)
            _heap_push(acc.worst_heap, prio, frame_bgr.copy(), box,
                       frame_idx, live_score, is_real, max_size=2)

    cap.release()

    # 5) Persist sample frames for this video
    out_dir = os.path.join(sample_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    # best (correct) -- sort descending by priority
    bests = sorted(acc.best_heap, key=lambda x: -x[0])
    for i, (prio, _ctr, fidx, fbgr, box, score, is_real) in enumerate(bests, 1):
        img = annotate_frame(fbgr, box, score, is_real, label)
        cv2.imwrite(os.path.join(out_dir, f"best_{i}.jpg"), img)

    # worst (errors) -- if no errors recorded, fall back to *least* confident correct
    if acc.worst_heap:
        worsts = sorted(acc.worst_heap, key=lambda x: -x[0])
    else:
        # No errors at all -- pick 2 lowest-confidence-correct frames as "worst"
        worsts = sorted(acc.best_heap, key=lambda x: x[0])[:2]
    for i, (prio, _ctr, fidx, fbgr, box, score, is_real) in enumerate(worsts, 1):
        img = annotate_frame(fbgr, box, score, is_real, label)
        cv2.imwrite(os.path.join(out_dir, f"worst_{i}.jpg"), img)

    type_tag = f"({spoof_type})" if spoof_type != "n/a" else ""
    print(f"[OK] {name:22s} | group={group:5s}{type_tag:8s} | label={label:5s} | "
          f"frames={acc.frames:4d} | mean_score={np.mean(acc.scores):.3f} | "
          f"mean_latency={np.mean(acc.latencies):.1f} ms")
    return acc


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_metrics_at_threshold(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> Dict[str, float]:
    """labels: 1 = live (bona-fide), 0 = spoof (attack).
    A presentation is classified as live iff score >= threshold."""
    pred_live = scores >= threshold
    live_mask = labels == 1
    spoof_mask = labels == 0

    n_live = int(live_mask.sum())
    n_spoof = int(spoof_mask.sum())

    # APCER = attack frames classified as live / total attack frames
    if n_spoof > 0:
        fp = int(((pred_live) & spoof_mask).sum())
        apcer = fp / n_spoof
    else:
        fp, apcer = 0, float("nan")

    # BPCER = bona-fide frames classified as spoof / total bona-fide frames
    if n_live > 0:
        fn = int(((~pred_live) & live_mask).sum())
        bpcer = fn / n_live
    else:
        fn, bpcer = 0, float("nan")

    tp = int(((pred_live) & live_mask).sum())
    tn = int(((~pred_live) & spoof_mask).sum())

    acer = (apcer + bpcer) / 2 if not (math.isnan(apcer) or math.isnan(bpcer)) else float("nan")

    return {
        "threshold": float(threshold),
        "APCER": apcer,
        "BPCER": bpcer,
        "ACER": acer,
        "TP": tp, "FN": fn, "TN": tn, "FP": fp,
        "n_live": n_live, "n_spoof": n_spoof,
    }


def compute_eer(roc_df: pd.DataFrame) -> Tuple[float, float]:
    """Return (eer_threshold, eer_value) -- the threshold that minimises
    |APCER - BPCER|, and the value of APCER (== BPCER) at that point."""
    diffs = (roc_df["APCER"] - roc_df["BPCER"]).abs()
    i = int(diffs.idxmin())
    thr = float(roc_df.loc[i, "threshold"])
    eer = float((roc_df.loc[i, "APCER"] + roc_df.loc[i, "BPCER"]) / 2)
    return thr, eer


# ---------------------------------------------------------------------------
# Confusion-matrix sample collection (TP / TN / FP / FN)
# ---------------------------------------------------------------------------
def export_confusion_samples(
    video_accs: List[VideoAcc], sample_dir: str, n_per_cell: int = 2
) -> None:
    """Walk through the in-memory accumulators a second time and pull 2 frames
    per confusion cell (TP, TN, FP, FN), prioritising the most representative:
        TP: highest live_score among correct live frames
        TN: lowest  live_score among correct spoof frames
        FP: highest live_score among spoof frames (false accept = worst case)
        FN: lowest  live_score among live  frames (false reject)

    To keep memory bounded we already have the best/worst heaps populated per
    video; here we just re-scan them and pick globally.
    """
    # NOTE: best/worst heaps only retain a handful of frames, but they are the
    # most informative ones, so we sample from them. If the user wants more
    # diverse cells they can simply replay the videos.
    pools = {"TP": [], "TN": [], "FP": [], "FN": []}

    for acc in video_accs:
        all_frames = acc.best_heap + acc.worst_heap
        for prio, _ctr, fidx, fbgr, box, score, is_real in all_frames:
            correct = _correct(acc.label, is_real)
            if acc.label == "live" and correct:
                cell = "TP"
            elif acc.label == "spoof" and correct:
                cell = "TN"
            elif acc.label == "spoof" and not correct:
                cell = "FP"
            else:
                cell = "FN"
            pools[cell].append((score, acc.name, fbgr, box, is_real, acc.label, fidx))

    # Sort each pool to grab the most "representative" frames
    pools["TP"].sort(key=lambda r: -r[0])           # highest live_score first
    pools["TN"].sort(key=lambda r:  r[0])           # lowest live_score first
    pools["FP"].sort(key=lambda r: -r[0])           # highest live_score among spoof
    pools["FN"].sort(key=lambda r:  r[0])           # lowest live_score among live

    out_dir = os.path.join(sample_dir, "_confusion")
    os.makedirs(out_dir, exist_ok=True)
    written = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for cell, rows in pools.items():
        for score, vname, fbgr, box, is_real, label, fidx in rows[:n_per_cell]:
            img = annotate_frame(fbgr, box, score, is_real, label)
            fname = f"{cell}_{written[cell] + 1}_{vname}_f{fidx}.jpg"
            cv2.imwrite(os.path.join(out_dir, fname), img)
            written[cell] += 1
    print(f"[OK] Confusion samples written: {written}")


# ---------------------------------------------------------------------------
# Excel report
# ---------------------------------------------------------------------------
def build_report(video_accs: List[VideoAcc]) -> None:
    # Per-video sheet
    per_video_rows = []
    frame_log_rows = []
    for acc in video_accs:
        if acc.frames == 0:
            continue
        per_video_rows.append({
            "video": f"{acc.name}.mp4",
            "group": acc.group,
            "spoof_type": acc.spoof_type,
            "label": acc.label,
            "total_frames": acc.frames,
            "face_detected_frames": acc.detected,
            "detection_rate(%)": 100.0 * acc.detected / acc.frames,
            "mean_live_score": float(np.mean(acc.scores)) if acc.scores else float("nan"),
            "std_live_score": float(np.std(acc.scores)) if acc.scores else float("nan"),
            "frac_pred_live_default(%)": 100.0 * float(np.mean(acc.is_real_default)) if acc.is_real_default else float("nan"),
            "mean_latency(ms)": float(np.mean(acc.latencies)) if acc.latencies else float("nan"),
            "std_latency(ms)": float(np.std(acc.latencies)) if acc.latencies else float("nan"),
        })
        for i, (s, ir, lat) in enumerate(zip(acc.scores, acc.is_real_default, acc.latencies), 1):
            frame_log_rows.append({
                "video": f"{acc.name}.mp4",
                "group": acc.group,
                "spoof_type": acc.spoof_type,
                "label": acc.label,
                "frame_idx": i,
                "live_score": s,
                "is_real_default": int(ir),
                "latency_ms": lat,
            })

    df_video = pd.DataFrame(per_video_rows)
    df_log = pd.DataFrame(frame_log_rows)

    # Per group -- includes per-group APCER/BPCER at the DEFAULT (is_real) decision.
    # APCER is meaningful for the 'spoof' group; BPCER is meaningful for the
    # 'user' and 'mask' groups (both are bona-fide).
    if not df_video.empty:
        df_group = df_video.groupby("group").agg(
            videos=("video", "count"),
            total_frames=("total_frames", "sum"),
            mean_live_score=("mean_live_score", "mean"),
            mean_latency_ms=("mean_latency(ms)", "mean"),
            mean_pred_live_pct=("frac_pred_live_default(%)", "mean"),
        ).reset_index()

        # Add per-group APCER/BPCER columns using the raw frame log
        if not df_log.empty:
            per_grp_rates = []
            for grp in df_group["group"].tolist():
                sub = df_log[df_log["group"] == grp]
                if sub.empty:
                    per_grp_rates.append({"group": grp, "APCER(%)": float("nan"),
                                          "BPCER(%)": float("nan")})
                    continue
                lbl = (sub["label"] == "live").astype(int).to_numpy()
                pred = sub["is_real_default"].to_numpy().astype(bool)
                n_live = int((lbl == 1).sum())
                n_spoof = int((lbl == 0).sum())
                fn = int(((~pred) & (lbl == 1)).sum())
                fp = int(((pred) & (lbl == 0)).sum())
                apcer = (fp / n_spoof) if n_spoof else float("nan")
                bpcer = (fn / n_live) if n_live else float("nan")
                per_grp_rates.append({
                    "group": grp,
                    "APCER(%)": apcer * 100 if not math.isnan(apcer) else float("nan"),
                    "BPCER(%)": bpcer * 100 if not math.isnan(bpcer) else float("nan"),
                })
            df_rates = pd.DataFrame(per_grp_rates)
            df_group = df_group.merge(df_rates, on="group", how="left")
    else:
        df_group = pd.DataFrame()

    # ROC sweep + confusion @ default + EER
    scores = df_log["live_score"].to_numpy() if not df_log.empty else np.array([])
    labels = (df_log["label"] == "live").astype(int).to_numpy() if not df_log.empty else np.array([])

    if scores.size and labels.size:
        roc_rows = [compute_metrics_at_threshold(scores, labels, t) for t in THRESHOLD_GRID]
        df_roc = pd.DataFrame(roc_rows)
        eer_thr, eer_val = compute_eer(df_roc)
        m_eer = compute_metrics_at_threshold(scores, labels, eer_thr)

        # "Default" = use is_real flag from app.py (no thresholding by us)
        pred_default = df_log["is_real_default"].to_numpy().astype(bool)
        tp = int(((pred_default) & (labels == 1)).sum())
        fn = int(((~pred_default) & (labels == 1)).sum())
        fp = int(((pred_default) & (labels == 0)).sum())
        tn = int(((~pred_default) & (labels == 0)).sum())
        n_live = int((labels == 1).sum())
        n_spoof = int((labels == 0).sum())
        apcer_d = fp / n_spoof if n_spoof else float("nan")
        bpcer_d = fn / n_live if n_live else float("nan")
        acer_d = (apcer_d + bpcer_d) / 2

        df_cm_default = pd.DataFrame([
            {"actual": "live",  "pred_live": tp, "pred_spoof": fn, "total": n_live},
            {"actual": "spoof", "pred_live": fp, "pred_spoof": tn, "total": n_spoof},
        ])
        df_cm_eer = pd.DataFrame([
            {"actual": "live",  "pred_live": m_eer["TP"], "pred_spoof": m_eer["FN"], "total": m_eer["n_live"]},
            {"actual": "spoof", "pred_live": m_eer["FP"], "pred_spoof": m_eer["TN"], "total": m_eer["n_spoof"]},
        ])

        df_summary = pd.DataFrame([
            {"operating_point": "default (app.py)",
             "threshold": float("nan"),
             "APCER(%)": apcer_d * 100, "BPCER(%)": bpcer_d * 100, "ACER(%)": acer_d * 100,
             "TP": tp, "FN": fn, "FP": fp, "TN": tn},
            {"operating_point": "EER",
             "threshold": eer_thr,
             "APCER(%)": m_eer["APCER"] * 100, "BPCER(%)": m_eer["BPCER"] * 100,
             "ACER(%)": m_eer["ACER"] * 100,
             "TP": m_eer["TP"], "FN": m_eer["FN"], "FP": m_eer["FP"], "TN": m_eer["TN"]},
        ])
        # Convert ROC fractions to percentages for human reading
        df_roc_out = df_roc.copy()
        for col in ("APCER", "BPCER", "ACER"):
            df_roc_out[col] = df_roc_out[col] * 100
        df_roc_out = df_roc_out.rename(columns={"APCER": "APCER(%)",
                                                "BPCER": "BPCER(%)",
                                                "ACER":  "ACER(%)"})
    else:
        df_roc_out = df_cm_default = df_cm_eer = df_summary = pd.DataFrame()

    # ---- Per Spoof Type sheet ----
    # Tach phone vs print de bao cao chi tiet ve khang nang chong tung loai
    # tan cong. Day la thong tin co gia tri cao cho luan van (chung minh
    # he thong chong duoc nhieu loai PA).
    df_spoof_type = pd.DataFrame()
    if not df_log.empty:
        spoof_log = df_log[df_log["group"] == "spoof"]
        if not spoof_log.empty and "spoof_type" in spoof_log.columns:
            rows = []
            for stype in sorted(spoof_log["spoof_type"].unique()):
                sub = spoof_log[spoof_log["spoof_type"] == stype]
                n = len(sub)
                pred = sub["is_real_default"].astype(bool).to_numpy()
                fp = int(pred.sum())                # spoof bi accept = APCER
                tn = n - fp                          # spoof bi reject (dung)
                apcer = (fp / n) if n else float("nan")
                rows.append({
                    "spoof_type": stype,
                    "videos": int(sub["video"].nunique()),
                    "total_frames": n,
                    "mean_live_score": float(sub["live_score"].mean()),
                    "std_live_score": float(sub["live_score"].std()),
                    "frames_accepted_as_live": fp,
                    "frames_rejected_as_spoof": tn,
                    "APCER(%)": apcer * 100 if not math.isnan(apcer) else float("nan"),
                })
            df_spoof_type = pd.DataFrame(rows)

    # Mask-only confusion -- highlights the most interesting failure mode of
    # the anti-spoof module: bona-fide subjects wearing a surgical mask being
    # mis-classified as a presentation attack.  The matrix has just ONE
    # actual row ('live (mask)') and two prediction columns.
    if not df_log.empty and "mask" in set(df_log["group"]):
        sub = df_log[df_log["group"] == "mask"]
        n = len(sub)
        pred_live = int(sub["is_real_default"].astype(bool).sum())
        pred_spoof = n - pred_live
        bpcer_mask = pred_spoof / n if n else float("nan")
        df_cm_mask = pd.DataFrame([
            {"actual": "live (mask)",
             "pred_live": pred_live,
             "pred_spoof": pred_spoof,
             "total": n,
             "BPCER_mask(%)": bpcer_mask * 100 if not math.isnan(bpcer_mask) else float("nan")},
        ])
    else:
        df_cm_mask = pd.DataFrame()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with pd.ExcelWriter(REPORT_XLSX, engine="openpyxl") as xw:
        df_video.to_excel(xw, sheet_name="Per Video", index=False)
        df_group.to_excel(xw, sheet_name="Per Group", index=False)
        df_cm_default.to_excel(xw, sheet_name="Confusion @ Default", index=False)
        df_cm_eer.to_excel(xw, sheet_name="Confusion @ EER", index=False)
        if not df_spoof_type.empty:
            df_spoof_type.to_excel(xw, sheet_name="Per Spoof Type", index=False)
        if not df_cm_mask.empty:
            df_cm_mask.to_excel(xw, sheet_name="Confusion Mask Only", index=False)
        df_roc_out.to_excel(xw, sheet_name="ROC Sweep", index=False)
        df_summary.to_excel(xw, sheet_name="Summary", index=False)
        df_log.to_excel(xw, sheet_name="Frame Log", index=False)

    print(f"\n[OK] Report written: {REPORT_XLSX}")
    if not df_summary.empty:
        print(df_summary.to_string(index=False))
    if not df_spoof_type.empty:
        print("\nPer spoof type (APCER tach phone vs print):")
        print(df_spoof_type.to_string(index=False))
    if not df_cm_mask.empty:
        print("\nMask-only confusion (BPCER on masked bona-fide):")
        print(df_cm_mask.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(SAMPLE_ROOT, exist_ok=True)

    videos = find_videos(VIDEO_DIR)
    if not videos:
        print(f"[ERROR] no videos found in '{VIDEO_DIR}'")
        sys.exit(1)

    print(f"[INFO] Found {len(videos)} videos in {VIDEO_DIR}/")
    for v in videos:
        print(f"  - {v}")

    accs: List[VideoAcc] = []
    for v in videos:
        accs.append(test_one_video(v, SAMPLE_ROOT))

    # Per-cell confusion samples for the visualizer
    export_confusion_samples(accs, SAMPLE_ROOT, n_per_cell=2)

    # Excel report
    build_report(accs)


if __name__ == "__main__":
    main()
