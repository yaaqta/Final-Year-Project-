"""
test_face_recognition_v2.py
============================

Standalone evaluation for the FaceNet-based face recognition module used in
the Smart Locker capstone project. Replaces the single-video test_face_recognition.py
with a video-folder sweep that scales from 1 to ~20 registered identities.

What it does
------------
For every frame of every test video in `test_videos/`:
  1. Run YOLOv12n-face to find a face bounding box (full-frame fallback if no
     detection so we still get a recognition decision).
  2. Pass the full RGB frame + bbox into `get_face_embedding_from_image()`,
     which internally crops the face -> 512-d FaceNet embedding.
  4. Run `recognize_face(embedding, threshold=DEFAULT_THR)` ->
        (best_user: str|None, confidence: float|None)
     where `confidence = 1 - min_cosine_distance` when accepted, None when
     the closest match is farther than threshold.
  5. Also compute the *raw* minimum cosine distance to every DB embedding so
     the threshold sweep can re-decide later without re-running FaceNet.

Ground truth
------------
Filename prefix decides the true identity:
  - `<name>_<idx>.mp4`  -> true_label = "<name>"  (lower-case)
  - `unknown_<idx>.mp4` -> true_label = "Unknown"
  - `stranger_<idx>.mp4`-> true_label = "Unknown"

A frame is "registered" when true_label is a key in load_embeddings();
otherwise it is a "stranger". This logic scales from 1 to N enrolled users.

Metrics (per frame, then aggregated)
------------------------------------
  - Accuracy  = (correct_name + correct_unknown) / total
  - FAR       = strangers accepted as some user / total stranger frames     (security)
  - FRR       = registered rejected (predicted Unknown) / total registered  (UX)
  - IDR       = registered correctly named, among registered-accepted frames

A threshold sweep on cosine *distance* 0.30 -> 0.80 (step 0.01) lets us find
the optimal operating point (min FAR + FRR).

Output
------
results/Recog_Report.xlsx with sheets:
  - Per Video
  - Per Identity
  - Confusion @ Default        (binary: Registered/Stranger vs Recognized/Unknown)
  - Confusion @ Optimal        (binary, at swept-optimal threshold)
  - Confusion Full @ Default   (N+1 x N+1, only when n_identities <= 8)
  - Threshold Sweep
  - Summary
  - Frame Log                  (raw per-frame distances + decisions)

Sample frames saved to results/recog_samples/<video_name>/:
  - best_*.jpg / worst_*.jpg per video
  - results/recog_samples/_confusion/{TP,TN,FA,FR}_*.jpg for the figure

NOTE
----
Plain ASCII only (no Vietnamese diacritics) so the file drops into the
user's Windows project folder without UTF-8 surprises.
"""

from __future__ import annotations

import argparse
import json
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
from app import (                          # noqa: E402
    yolo_model,
    get_face_embedding_from_image,
    recognize_face,
    load_embeddings,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VIDEO_DIR = "test_videos"
RESULTS_DIR = "results"
SAMPLE_ROOT = os.path.join(RESULTS_DIR, "recog_samples")
REPORT_XLSX = os.path.join(RESULTS_DIR, "Recog_Report.xlsx")

# Default threshold matches app.py's recognize_face(threshold=0.55).
# This is on *cosine distance* (lower = more similar).
DEFAULT_THR = 0.85

# Threshold sweep range for FAR / FRR / Accuracy. Cosine distance lives
# roughly in [0, 1] for FaceNet embeddings, with same-identity ~0.2-0.4 and
# different-identity ~0.5-0.8.
THRESHOLD_GRID = np.round(np.arange(0.30, 0.801, 0.01), 4)

# Show the full N+1 x N+1 confusion matrix when it is still readable.
MAX_FULL_CM_SIZE = 8

UNKNOWN_LABEL = "Unknown"
STRANGER_PREFIXES = ("unknown_", "stranger_")

# When a recognition_split.json manifest is provided, ground-truth labels,
# roles, splits, and groups come from the manifest instead of being parsed
# from the filename. The manifest is produced by prepare_recognition_split.py.
GROUP_NORMAL = "normal"
GROUP_MASK = "mask"
GROUP_SPOOF = "spoof"
GROUP_UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Per-video accumulator + bounded heaps for sample frames
# ---------------------------------------------------------------------------
@dataclass
class VideoAcc:
    """In-memory accumulator for one video.

    For each frame we keep:
      - min_dist            : the cosine distance to the closest DB entry
      - closest_name        : which DB user that was
      - pred_default        : prediction at DEFAULT_THR (None means "Unknown")
      - confidence_default  : 1 - min_dist if accepted, else None
      - latency_ms          : wall-clock time for the (embed + recognize) call

    Sample-frame heaps mirror test_anti_spoofing.py:
      - best_heap  : top-3 frames most confidently correct (priority = correctness margin)
      - worst_heap : top-2 frames most damaging mistakes  (priority = error severity)
    """
    name: str
    true_label: str               # lower-case identity, or "Unknown"
    is_registered: bool           # True if true_label is in the DB at start of run
    group: str = GROUP_UNKNOWN    # normal | mask | spoof | unknown (from manifest if present)
    role: str = ""                # registered | stranger | ""
    split: str = ""               # dev | test | ""
    frames: int = 0
    detected: int = 0
    min_dists: List[float] = field(default_factory=list)
    closest_names: List[str] = field(default_factory=list)
    pred_defaults: List[Optional[str]] = field(default_factory=list)
    latencies: List[float] = field(default_factory=list)

    # Heaps store (priority, counter, frame_idx, frame_bgr, box,
    #              min_dist, closest, pred_default)
    best_heap: list = field(default_factory=list)
    worst_heap: list = field(default_factory=list)


def _classify_outcome(true_label: str, is_registered: bool,
                       pred: Optional[str]) -> str:
    """Return one of: TP, TN, FA, FR.

      - TP  : registered face correctly named             (true == pred, != None)
      - TN  : stranger correctly rejected                  (pred is None, not registered)
      - FA  : stranger accepted as some user (false ACCEPT, security incident)
      - FR  : registered face rejected (false REJECT, UX issue)
      - MIS : registered face mis-identified as a different user
              (treated as FA for security accounting but tagged differently here)
    """
    if is_registered:
        if pred is None:
            return "FR"
        if pred.lower() == true_label.lower():
            return "TP"
        return "MIS"           # registered but wrong name -> bad for both UX & security
    # stranger
    if pred is None:
        return "TN"
    return "FA"


def _correctness_priority(true_label: str, is_registered: bool,
                           pred: Optional[str], min_dist: float) -> float:
    """How confidently *correct* was this frame? Higher = better.
    Used to keep the top-N best frames per video.
    """
    outcome = _classify_outcome(true_label, is_registered, pred)
    if outcome == "TP":
        # Lower distance -> more confident match. Convert to similarity-like score.
        return 1.0 - min_dist
    if outcome == "TN":
        # Higher distance -> more confident rejection of stranger.
        return min_dist
    # Wrong frames have priority 0 here -- they go in the worst_heap instead.
    return -1.0


def _error_severity(true_label: str, is_registered: bool,
                     pred: Optional[str], min_dist: float) -> float:
    """How damaging was this mistake? Higher = worse. Only meaningful for errors."""
    outcome = _classify_outcome(true_label, is_registered, pred)
    if outcome == "FA":
        # Stranger accepted: a closer match (smaller distance) is more dangerous
        # because the system was more confident in the wrong answer.
        return 1.0 - min_dist
    if outcome == "MIS":
        # Registered but named as someone else: same severity logic.
        return 1.0 - min_dist
    if outcome == "FR":
        # Registered rejected: a *low* distance with rejection is weird;
        # but the more common case is that distance was just above threshold.
        # We weight by how close it got -- nearer to threshold = less severe.
        return min_dist
    # Correct frames -> low severity sentinel
    return -1.0


_GLOBAL_HEAP_COUNTER = 0


def _heap_push(heap, priority, payload_extras, max_size):
    """Bounded min-heap: keep top-`max_size` largest priorities."""
    global _GLOBAL_HEAP_COUNTER
    _GLOBAL_HEAP_COUNTER += 1
    item = (priority, _GLOBAL_HEAP_COUNTER) + tuple(payload_extras)
    if len(heap) < max_size:
        heapq.heappush(heap, item)
    elif priority > heap[0][0]:
        heapq.heapreplace(heap, item)


# ---------------------------------------------------------------------------
# Drawing helpers (mirrors test_anti_spoofing.py style)
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


# Confusion-cell colour palette (BGR for OpenCV)
COL_TP = (0, 200, 0)        # green   -> registered correctly named
COL_TN = (200, 80, 0)       # blue    -> stranger correctly rejected
COL_FA = (0, 0, 255)        # red     -> stranger accepted (security incident)
COL_FR = (0, 140, 255)      # orange  -> registered rejected
COL_MIS = (180, 0, 180)     # magenta -> registered but wrong name


def _cell_color(outcome: str):
    return {
        "TP": COL_TP, "TN": COL_TN,
        "FA": COL_FA, "FR": COL_FR,
        "MIS": COL_MIS,
    }.get(outcome, (120, 120, 120))


def annotate_frame(frame_bgr, box, min_dist: float, closest: str,
                   pred: Optional[str], outcome: str) -> np.ndarray:
    """Draw bbox + label showing predicted name + confidence.
    Border colour encodes the confusion-matrix cell.
    """
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    fs, tt, bt = _label_scale(w)

    col = _cell_color(outcome)
    border = max(4, int(round(w / 200)))
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), col, border)

    name_disp = pred if pred else UNKNOWN_LABEL
    conf = 1.0 - min_dist
    text = f"{name_disp} d={min_dist:.2f}"

    if box is not None:
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, bt)
        _put_label_with_bg(img, text, (x1, y1 - 2), col, fs, tt)

    # Cell tag top-left
    _put_label_with_bg(img, outcome, (border + 4, int(fs * 36) + border),
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
    seen, uniq = set(), []
    for f in files:
        rp = os.path.realpath(f).lower()
        if rp not in seen:
            seen.add(rp)
            uniq.append(f)
    return sorted(uniq)


def parse_true_label(video_basename: str) -> str:
    """Return the lower-case identity encoded in the filename prefix,
    or 'Unknown' for stranger-tagged videos.

    Used as a *fallback* when no recognition_split.json manifest is given.
    The manifest is preferred for the 12-user multi-group dataset.
    """
    base = os.path.basename(video_basename).lower()
    for sp in STRANGER_PREFIXES:
        if base.startswith(sp):
            return UNKNOWN_LABEL
    # Take prefix up to last underscore-then-digit chunk. e.g. an_1.mp4 -> "an",
    # nguyen_van_a_3.mp4 -> "nguyen_van_a".
    stem = os.path.splitext(base)[0]
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def parse_group_from_filename(video_basename: str) -> str:
    """Return normal / mask / spoof / unknown from the filename prefix."""
    base = os.path.basename(video_basename).lower()
    if base.startswith("user_"):
        return GROUP_NORMAL
    if base.startswith("mask_"):
        return GROUP_MASK
    if base.startswith("spoof_"):
        return GROUP_SPOOF
    return GROUP_UNKNOWN


def load_manifest(path: str) -> Dict:
    """Load a recognition_split.json. Returns {} if the path is empty/missing."""
    if not path:
        return {}
    if not os.path.isfile(path):
        print(f"[ERROR] manifest not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as fh:
        m = json.load(fh)
    print(f"[INFO] Loaded manifest: {path}")
    print(f"       registered = {m.get('registered_identities', [])}")
    print(f"       stranger   = {m.get('stranger_identities', [])}")
    return m


def lookup_video_meta(manifest: Dict, video_basename: str,
                       db_keys_lower: set) -> Dict[str, str]:
    """Resolve (true_label, is_registered, group, role, split) for one video.

    When a manifest entry exists for the file we trust it.  Otherwise we fall
    back to filename parsing so legacy folders still work.
    """
    fname = os.path.basename(video_basename)
    entries = manifest.get("videos", {}) if manifest else {}
    if fname in entries:
        meta = entries[fname]
        true_label = str(meta.get("ground_truth", "")).strip()
        if true_label.lower() == UNKNOWN_LABEL.lower() or not true_label:
            true_label = UNKNOWN_LABEL
        is_reg = (meta.get("role", "") == "registered")
        return {
            "true_label": true_label,
            "is_registered": is_reg,
            "group": meta.get("group", GROUP_UNKNOWN),
            "role": meta.get("role", "stranger" if not is_reg else "registered"),
            "split": meta.get("split", ""),
        }
    # Fallback path (legacy 1-user demo without manifest)
    parsed = parse_true_label(fname)
    true_label = UNKNOWN_LABEL if parsed == UNKNOWN_LABEL else parsed
    is_reg = (true_label.lower() in db_keys_lower)
    return {
        "true_label": true_label,
        "is_registered": is_reg,
        "group": parse_group_from_filename(fname),
        "role": "registered" if is_reg else "stranger",
        "split": "",
    }


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


def _all_distances(embedding: np.ndarray, db: Dict[str, np.ndarray]
                    ) -> Tuple[float, str, Dict[str, float]]:
    """Return (min_dist, closest_name, all_distances_dict).

    Mirrors app.recognize_face but also exposes per-user distances so the
    threshold sweep can re-decide without re-embedding every frame.
    """
    dists: Dict[str, float] = {}
    min_d, best = float("inf"), ""
    emb = np.asarray(embedding, dtype=np.float32).ravel()
    en = np.linalg.norm(emb) + 1e-12
    for user, db_emb in db.items():
        v = np.asarray(db_emb, dtype=np.float32).ravel()
        vn = np.linalg.norm(v) + 1e-12
        d = float(1.0 - np.dot(emb, v) / (en * vn))
        dists[user] = d
        if d < min_d:
            min_d = d
            best = user
    return min_d, best, dists


# ---------------------------------------------------------------------------
# Test one video
# ---------------------------------------------------------------------------
def test_one_video(video_path: str, sample_dir: str,
                    db: Dict[str, np.ndarray],
                    meta: Dict[str, str],
                    frame_stride: int = 1) -> Tuple[VideoAcc, pd.DataFrame]:
    name = os.path.splitext(os.path.basename(video_path))[0]
    true_label = meta["true_label"]
    is_registered = bool(meta["is_registered"])
    acc = VideoAcc(
        name=name,
        true_label=true_label,
        is_registered=is_registered,
        group=meta.get("group", GROUP_UNKNOWN),
        role=meta.get("role", ""),
        split=meta.get("split", ""),
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] cannot open {video_path}")
        return acc, pd.DataFrame()

    per_frame_rows: List[Dict] = []
    frame_idx = 0
    stride = max(1, int(frame_stride))
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_idx += 1
        # Frame-stride sampling (every N-th frame). Keeps runtime tractable
        # for 36 videos x 200-500 frames each.
        if stride > 1 and (frame_idx - 1) % stride != 0:
            continue
        acc.frames += 1

        # 1) YOLO detect
        box = detect_one_face(frame_bgr)

        # 2) Embed via app.get_face_embedding_from_image(frame_rgb, [[x1,y1,x2,y2]])
        #    which internally crops the face from the full RGB frame.
        t0 = time.time()
        emb = None
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if box is not None:
            acc.detected += 1
            x1, y1, x2, y2 = box
            try:
                emb = get_face_embedding_from_image(
                    frame_rgb, [[int(x1), int(y1), int(x2), int(y2)]]
                )
            except Exception as exc:
                print(f"[WARN] embedding failed on {name} frame {frame_idx}: {exc}")
                emb = None
        if emb is not None and db:
            min_d, closest, all_d = _all_distances(emb, db)
        else:
            min_d, closest, all_d = float("nan"), "", {}
        t_ms = (time.time() - t0) * 1000.0

        if math.isfinite(min_d):
            pred_default = closest if min_d < DEFAULT_THR else None
        else:
            pred_default = None
        outcome = _classify_outcome(true_label, is_registered, pred_default)

        acc.min_dists.append(min_d)
        acc.closest_names.append(closest)
        acc.pred_defaults.append(pred_default)
        acc.latencies.append(t_ms)

        # Heaps for sample frames
        if outcome in ("TP", "TN"):
            prio = _correctness_priority(true_label, is_registered,
                                          pred_default, min_d)
            _heap_push(acc.best_heap, prio,
                       (frame_idx, frame_bgr.copy(), box,
                        min_d, closest, pred_default, outcome),
                       max_size=3)
        else:
            prio = _error_severity(true_label, is_registered,
                                    pred_default, min_d)
            _heap_push(acc.worst_heap, prio,
                       (frame_idx, frame_bgr.copy(), box,
                        min_d, closest, pred_default, outcome),
                       max_size=2)

        # Per-frame row for the threshold sweep + frame log
        row = {
            "video": f"{name}.mp4",
            "true_label": true_label,
            "is_registered": int(is_registered),
            "group": acc.group,
            "role": acc.role,
            "split": acc.split,
            "frame_idx": frame_idx,
            "face_detected": int(box is not None),
            "min_dist": min_d,
            "closest": closest,
            "pred_default": pred_default if pred_default else "",
            "outcome_default": outcome,
            "latency_ms": t_ms,
        }
        # Stash per-user distances so the sweep can pick the *closest* user
        # at any threshold without re-embedding. JSON-string-encode to fit
        # in one Excel cell (typically <8 users).
        if all_d:
            row["dist_json"] = "{" + ",".join(
                f'"{k}":{v:.4f}' for k, v in sorted(all_d.items())) + "}"
        else:
            row["dist_json"] = "{}"
        per_frame_rows.append(row)

    cap.release()

    # Persist best/worst sample frames
    out_dir = os.path.join(sample_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    bests = sorted(acc.best_heap, key=lambda x: -x[0])
    for i, item in enumerate(bests, 1):
        _, _ctr, fidx, fbgr, box, mind, closest, pred, outcome = item
        img = annotate_frame(fbgr, box, mind, closest, pred, outcome)
        cv2.imwrite(os.path.join(out_dir, f"best_{i}.jpg"), img)

    if acc.worst_heap:
        worsts = sorted(acc.worst_heap, key=lambda x: -x[0])
    else:
        # No errors -- fall back to least-confident correct frames
        worsts = sorted(acc.best_heap, key=lambda x: x[0])[:2]
    for i, item in enumerate(worsts, 1):
        _, _ctr, fidx, fbgr, box, mind, closest, pred, outcome = item
        img = annotate_frame(fbgr, box, mind, closest, pred, outcome)
        cv2.imwrite(os.path.join(out_dir, f"worst_{i}.jpg"), img)

    valid = [d for d in acc.min_dists if math.isfinite(d)]
    mean_d = float(np.mean(valid)) if valid else float("nan")
    lat = float(np.mean(acc.latencies)) if acc.latencies else float("nan")
    print(f"[OK] {name:18s} | true={true_label:10s} | "
          f"registered={int(is_registered)} | frames={acc.frames:4d} | "
          f"mean_dist={mean_d:.3f} | mean_latency={lat:.1f} ms")
    return acc, pd.DataFrame(per_frame_rows)


# ---------------------------------------------------------------------------
# Threshold sweep + metric helpers
# ---------------------------------------------------------------------------
def _decide_at_threshold(min_dist: float, closest: str, thr: float) -> Optional[str]:
    if not math.isfinite(min_dist):
        return None
    return closest if min_dist < thr else None


def compute_metrics_at_threshold(df_log: pd.DataFrame, db_keys: List[str],
                                  thr: float) -> Dict[str, float]:
    """Compute Accuracy / FAR / FRR / IDR at a given cosine-distance threshold.

    Definitions (frame-level):
      - FAR = stranger frames accepted as some user / total stranger frames
      - FRR = registered frames rejected (Unknown) / total registered frames
      - IDR = registered frames correctly named, among registered-accepted frames
      - Accuracy = correctly classified frames / total
    """
    n = len(df_log)
    if n == 0:
        return {"threshold": thr, "Accuracy": float("nan"),
                "FAR": float("nan"), "FRR": float("nan"),
                "IDR": float("nan"), "n_total": 0,
                "TP": 0, "TN": 0, "FA": 0, "FR": 0, "MIS": 0}

    db_lower = {k.lower() for k in db_keys}
    tp = tn = fa = fr = mis = 0
    for _, row in df_log.iterrows():
        true_label = str(row["true_label"]).lower()
        is_reg = bool(int(row["is_registered"]))
        md = float(row["min_dist"])
        closest = str(row["closest"])
        pred = _decide_at_threshold(md, closest, thr)
        oc = _classify_outcome(true_label, is_reg, pred)
        if oc == "TP":   tp += 1
        elif oc == "TN": tn += 1
        elif oc == "FA": fa += 1
        elif oc == "FR": fr += 1
        elif oc == "MIS": mis += 1

    n_reg = tp + fr + mis
    n_str = tn + fa
    n_reg_accepted = tp + mis

    accuracy = (tp + tn) / n
    far = fa / n_str if n_str else float("nan")
    frr = fr / n_reg if n_reg else float("nan")
    idr = tp / n_reg_accepted if n_reg_accepted else float("nan")

    return {
        "threshold": float(thr),
        "Accuracy": accuracy,
        "FAR": far,
        "FRR": frr,
        "IDR": idr,
        "TP": tp, "TN": tn, "FA": fa, "FR": fr, "MIS": mis,
        "n_total": n, "n_registered": n_reg, "n_stranger": n_str,
    }


def find_optimal_threshold(df_sweep: pd.DataFrame) -> Tuple[float, float]:
    """Optimal = threshold that minimises (FAR + FRR), ignoring NaNs.
    Returns (threshold, FAR+FRR at that threshold).
    """
    s = df_sweep.copy()
    s["FAR_f"] = s["FAR"].fillna(0.0)
    s["FRR_f"] = s["FRR"].fillna(0.0)
    s["sum_err"] = s["FAR_f"] + s["FRR_f"]
    i = int(s["sum_err"].idxmin())
    return float(s.loc[i, "threshold"]), float(s.loc[i, "sum_err"])


# ---------------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------------
def build_binary_confusion(df_log: pd.DataFrame, db_keys: List[str],
                            thr: float) -> pd.DataFrame:
    """2x2 confusion matrix at a given threshold.
        rows = Actual {Registered, Stranger}
        cols = Predicted {Recognized, Unknown}
    """
    m = compute_metrics_at_threshold(df_log, db_keys, thr)
    return pd.DataFrame([
        {"actual": "Registered",
         "pred_recognized": m["TP"] + m["MIS"],
         "pred_unknown":    m["FR"],
         "total":           m["n_registered"]},
        {"actual": "Stranger",
         "pred_recognized": m["FA"],
         "pred_unknown":    m["TN"],
         "total":           m["n_stranger"]},
    ])


def build_full_confusion(df_log: pd.DataFrame, db_keys: List[str],
                          thr: float) -> pd.DataFrame:
    """(N+1) x (N+1) confusion matrix at a given threshold.

    Rows = Actual identities (every registered name + "Unknown")
    Cols = Predicted identities ( same set )
    """
    db_lower = sorted({k.lower() for k in db_keys})
    actuals = db_lower + [UNKNOWN_LABEL]
    preds = db_lower + [UNKNOWN_LABEL]
    mat = {a: {p: 0 for p in preds} for a in actuals}

    for _, row in df_log.iterrows():
        true_label = str(row["true_label"]).lower()
        md = float(row["min_dist"])
        closest = str(row["closest"]).lower()
        pred = _decide_at_threshold(md, closest, thr)
        a = true_label if true_label in db_lower else UNKNOWN_LABEL
        p = pred.lower() if pred else UNKNOWN_LABEL
        if p not in preds:
            p = UNKNOWN_LABEL
        mat[a][p] += 1

    rows = []
    for a in actuals:
        r = {"actual": a}
        r.update({f"pred_{p}": mat[a][p] for p in preds})
        r["total"] = sum(mat[a].values())
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-cell confusion samples (TP / TN / FA / FR) for the figure
# ---------------------------------------------------------------------------
def export_confusion_samples(video_accs: List[VideoAcc], sample_dir: str,
                              n_per_cell: int = 2) -> None:
    """Walk through best/worst heaps and pull 2 representative frames per cell."""
    pools = {"TP": [], "TN": [], "FA": [], "FR": []}
    for acc in video_accs:
        for item in (acc.best_heap + acc.worst_heap):
            _, _ctr, fidx, fbgr, box, mind, closest, pred, outcome = item
            if outcome == "MIS":
                # Count mis-identification under FA visually -- still a wrong "accept".
                outcome = "FA"
            if outcome in pools:
                pools[outcome].append((mind, acc.name, fbgr, box,
                                        closest, pred, outcome, fidx))

    # Pick the most informative frames:
    pools["TP"].sort(key=lambda r:  r[0])   # smallest distance = best match
    pools["TN"].sort(key=lambda r: -r[0])   # largest distance = best reject
    pools["FA"].sort(key=lambda r:  r[0])   # smallest distance = most dangerous false accept
    pools["FR"].sort(key=lambda r:  r[0])   # smallest distance = nearest-miss reject

    out_dir = os.path.join(sample_dir, "_confusion")
    os.makedirs(out_dir, exist_ok=True)
    written = {"TP": 0, "TN": 0, "FA": 0, "FR": 0}
    for cell, rows in pools.items():
        for mind, vname, fbgr, box, closest, pred, outcome, fidx in rows[:n_per_cell]:
            img = annotate_frame(fbgr, box, mind, closest, pred, outcome)
            fname = f"{cell}_{written[cell] + 1}_{vname}_f{fidx}.jpg"
            cv2.imwrite(os.path.join(out_dir, fname), img)
            written[cell] += 1
    print(f"[OK] Confusion samples written: {written}")


# ---------------------------------------------------------------------------
# Build the Excel report
# ---------------------------------------------------------------------------
def _per_group_metrics(df_log: pd.DataFrame, db_keys: List[str],
                        thr: float) -> pd.DataFrame:
    """Compute Accuracy/FAR/FRR/IDR per group (normal/mask/spoof/ALL) at `thr`.

    FAR is defined over stranger frames of that group; FRR over registered
    frames of that group.  For groups without any stranger or any registered
    frame, the corresponding rate is NaN.
    """
    if df_log.empty:
        return pd.DataFrame()
    rows: List[Dict] = []
    groups_present = [g for g in (GROUP_NORMAL, GROUP_MASK, GROUP_SPOOF, GROUP_UNKNOWN)
                       if g in set(df_log["group"].astype(str).tolist())]
    for g in groups_present + ["ALL"]:
        sub = df_log if g == "ALL" else df_log[df_log["group"] == g]
        if sub.empty:
            continue
        m = compute_metrics_at_threshold(sub, db_keys, thr)
        rows.append({
            "group": g,
            "threshold": float(thr),
            "n_total": m["n_total"],
            "n_registered": m["n_registered"],
            "n_stranger": m["n_stranger"],
            "Accuracy(%)": m["Accuracy"] * 100,
            "FAR(%)": (m["FAR"] * 100) if not math.isnan(m["FAR"]) else float("nan"),
            "FRR(%)": (m["FRR"] * 100) if not math.isnan(m["FRR"]) else float("nan"),
            "IDR(%)": (m["IDR"] * 100) if not math.isnan(m["IDR"]) else float("nan"),
            "TP": m["TP"], "TN": m["TN"], "FA": m["FA"],
            "FR": m["FR"], "MIS": m["MIS"],
        })
    return pd.DataFrame(rows)


def _dataset_manifest_sheet(video_accs: List[VideoAcc],
                              manifest: Dict) -> pd.DataFrame:
    """One row per video describing dataset composition for the report."""
    rows = []
    for acc in video_accs:
        rows.append({
            "video": f"{acc.name}.mp4",
            "identity": acc.true_label,
            "group": acc.group,
            "role": acc.role,
            "split": acc.split,
            "frames_sampled": acc.frames,
            "frames_with_face": acc.detected,
            "detection_rate(%)": (100.0 * acc.detected / acc.frames) if acc.frames else float("nan"),
        })
    df = pd.DataFrame(rows)
    if manifest:
        df.attrs["registered"] = manifest.get("registered_identities", [])
        df.attrs["stranger"] = manifest.get("stranger_identities", [])
    return df


def build_report(video_accs: List[VideoAcc],
                  df_log_all: pd.DataFrame,
                  db_keys: List[str],
                  manifest: Dict | None = None) -> None:
    manifest = manifest or {}
    # Per-video sheet
    rows = []
    for acc in video_accs:
        if acc.frames == 0:
            continue
        # Outcome counts at default threshold
        outcomes = [
            _classify_outcome(acc.true_label, acc.is_registered, p)
            for p in acc.pred_defaults
        ]
        tp = outcomes.count("TP"); tn = outcomes.count("TN")
        fa = outcomes.count("FA"); fr = outcomes.count("FR")
        mis = outcomes.count("MIS")

        valid = [d for d in acc.min_dists if math.isfinite(d)]
        rows.append({
            "video": f"{acc.name}.mp4",
            "true_label": acc.true_label,
            "is_registered": int(acc.is_registered),
            "group": acc.group,
            "role": acc.role,
            "split": acc.split,
            "total_frames": acc.frames,
            "face_detected_frames": acc.detected,
            "detection_rate(%)": 100.0 * acc.detected / acc.frames,
            "mean_min_dist": float(np.mean(valid)) if valid else float("nan"),
            "std_min_dist":  float(np.std(valid))  if valid else float("nan"),
            "TP": tp, "TN": tn, "FA": fa, "FR": fr, "MIS": mis,
            "frame_accuracy(%)": 100.0 * (tp + tn) / acc.frames,
            "mean_latency(ms)": float(np.mean(acc.latencies)) if acc.latencies else float("nan"),
        })
    df_video = pd.DataFrame(rows)

    # Per-identity sheet (collapses all videos of one identity)
    if not df_video.empty:
        df_ident = df_video.groupby("true_label").agg(
            videos=("video", "count"),
            total_frames=("total_frames", "sum"),
            face_detected_frames=("face_detected_frames", "sum"),
            mean_min_dist=("mean_min_dist", "mean"),
            TP=("TP", "sum"), TN=("TN", "sum"),
            FA=("FA", "sum"), FR=("FR", "sum"), MIS=("MIS", "sum"),
        ).reset_index()
        df_ident["accuracy(%)"] = 100.0 * (df_ident["TP"] + df_ident["TN"]) / df_ident["total_frames"]
        df_ident["mean_latency(ms)"] = df_video.groupby("true_label")["mean_latency(ms)"].mean().values
    else:
        df_ident = pd.DataFrame()

    # Threshold sweep
    if not df_log_all.empty:
        sweep_rows = [compute_metrics_at_threshold(df_log_all, db_keys, t)
                      for t in THRESHOLD_GRID]
        df_sweep = pd.DataFrame(sweep_rows)
        opt_thr, opt_sum_err = find_optimal_threshold(df_sweep)
        m_default = compute_metrics_at_threshold(df_log_all, db_keys, DEFAULT_THR)
        m_optimal = compute_metrics_at_threshold(df_log_all, db_keys, opt_thr)

        df_cm_default = build_binary_confusion(df_log_all, db_keys, DEFAULT_THR)
        df_cm_optimal = build_binary_confusion(df_log_all, db_keys, opt_thr)

        # Full N+1 x N+1 only when the DB is small enough that the figure is readable
        n_id = len(db_keys) + 1
        if n_id <= MAX_FULL_CM_SIZE:
            df_cm_full = build_full_confusion(df_log_all, db_keys, DEFAULT_THR)
        else:
            df_cm_full = pd.DataFrame()

        df_summary = pd.DataFrame([
            {"operating_point": "default",
             "threshold": DEFAULT_THR,
             "Accuracy(%)": m_default["Accuracy"] * 100,
             "FAR(%)":      m_default["FAR"]      * 100 if not math.isnan(m_default["FAR"]) else float("nan"),
             "FRR(%)":      m_default["FRR"]      * 100 if not math.isnan(m_default["FRR"]) else float("nan"),
             "IDR(%)":      m_default["IDR"]      * 100 if not math.isnan(m_default["IDR"]) else float("nan"),
             "TP": m_default["TP"], "TN": m_default["TN"],
             "FA": m_default["FA"], "FR": m_default["FR"], "MIS": m_default["MIS"]},
            {"operating_point": "optimal",
             "threshold": opt_thr,
             "Accuracy(%)": m_optimal["Accuracy"] * 100,
             "FAR(%)":      m_optimal["FAR"]      * 100 if not math.isnan(m_optimal["FAR"]) else float("nan"),
             "FRR(%)":      m_optimal["FRR"]      * 100 if not math.isnan(m_optimal["FRR"]) else float("nan"),
             "IDR(%)":      m_optimal["IDR"]      * 100 if not math.isnan(m_optimal["IDR"]) else float("nan"),
             "TP": m_optimal["TP"], "TN": m_optimal["TN"],
             "FA": m_optimal["FA"], "FR": m_optimal["FR"], "MIS": m_optimal["MIS"]},
        ])

        df_sweep_out = df_sweep.copy()
        for col in ("Accuracy", "FAR", "FRR", "IDR"):
            df_sweep_out[col] = df_sweep_out[col] * 100
        df_sweep_out = df_sweep_out.rename(columns={
            "Accuracy": "Accuracy(%)", "FAR": "FAR(%)",
            "FRR": "FRR(%)",        "IDR": "IDR(%)",
        })
    else:
        df_sweep_out = df_cm_default = df_cm_optimal = df_cm_full = df_summary = pd.DataFrame()

    # Dataset Manifest sheet -- describes the corpus actually tested
    df_manifest = _dataset_manifest_sheet(video_accs, manifest)

    # Per-Group breakdown at default + optimal threshold
    if not df_log_all.empty:
        df_grp_default = _per_group_metrics(df_log_all, db_keys, DEFAULT_THR)
        df_grp_default.insert(0, "operating_point", "default")
        df_grp_optimal = _per_group_metrics(df_log_all, db_keys, opt_thr)
        df_grp_optimal.insert(0, "operating_point", "optimal")
        df_per_group = pd.concat([df_grp_default, df_grp_optimal], ignore_index=True)
    else:
        df_per_group = pd.DataFrame()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with pd.ExcelWriter(REPORT_XLSX, engine="openpyxl") as xw:
        df_manifest.to_excel(xw, sheet_name="Dataset Manifest", index=False)
        df_video.to_excel(xw, sheet_name="Per Video", index=False)
        df_ident.to_excel(xw, sheet_name="Per Identity", index=False)
        if not df_per_group.empty:
            df_per_group.to_excel(xw, sheet_name="Per Group", index=False)
        df_cm_default.to_excel(xw, sheet_name="Confusion @ Default", index=False)
        df_cm_optimal.to_excel(xw, sheet_name="Confusion @ Optimal", index=False)
        if not df_cm_full.empty:
            df_cm_full.to_excel(xw, sheet_name="Confusion Full @ Default", index=False)
        df_sweep_out.to_excel(xw, sheet_name="Threshold Sweep", index=False)
        df_summary.to_excel(xw, sheet_name="Summary", index=False)
        df_log_all.to_excel(xw, sheet_name="Frame Log", index=False)

    print(f"\n[OK] Report written: {REPORT_XLSX}")
    if not df_summary.empty:
        print(df_summary.to_string(index=False))
    if not df_per_group.empty:
        print("\nPer-group breakdown:")
        print(df_per_group.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate the FaceNet recognition module on a folder of test videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--videos_dir", default=VIDEO_DIR,
                   help="Folder containing the test .mp4 files")
    p.add_argument("--manifest", default="",
                   help="Path to recognition_split.json (produced by prepare_recognition_split.py). "
                        "When supplied, ground truth, role, split, and group are read from the manifest.")
    p.add_argument("--split_filter", choices=["dev", "test", ""], default="",
                   help="When set, only run videos whose manifest split == this value.")
    p.add_argument("--frame_stride", type=int, default=1,
                   help="Process every N-th frame to control runtime (default 1 = every frame).")
    p.add_argument("--results_dir", default=RESULTS_DIR,
                   help="Output folder for the Excel report and sample frames")
    return p.parse_args()


def main():
    args = _parse_args()

    global RESULTS_DIR, SAMPLE_ROOT, REPORT_XLSX, VIDEO_DIR
    VIDEO_DIR = args.videos_dir
    RESULTS_DIR = args.results_dir
    SAMPLE_ROOT = os.path.join(RESULTS_DIR, "recog_samples")
    REPORT_XLSX = os.path.join(RESULTS_DIR, "Recog_Report.xlsx")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(SAMPLE_ROOT, exist_ok=True)

    db = load_embeddings()
    if not db:
        print("[ERROR] No embeddings registered. Enroll at least one user first.")
        sys.exit(1)
    print(f"[INFO] Loaded {len(db)} registered identities: "
          f"{sorted(db.keys())}")
    db_keys_lower = {k.lower() for k in db.keys()}

    manifest = load_manifest(args.manifest)

    videos = find_videos(VIDEO_DIR)
    if not videos:
        print(f"[ERROR] no videos found in '{VIDEO_DIR}'")
        sys.exit(1)

    # Resolve metadata for each video (manifest > filename fallback)
    metas: Dict[str, Dict[str, str]] = {
        v: lookup_video_meta(manifest, v, db_keys_lower) for v in videos
    }

    # Optional split filter
    if args.split_filter:
        videos = [v for v in videos if metas[v].get("split") == args.split_filter]
        print(f"[INFO] split_filter='{args.split_filter}' keeps {len(videos)} videos.")

    print(f"[INFO] Will process {len(videos)} videos from {VIDEO_DIR}/  "
          f"(frame_stride={args.frame_stride})")
    for v in videos:
        m = metas[v]
        print(f"  - {os.path.basename(v):28s}  identity={m['true_label']:10s}  "
              f"group={m['group']:7s}  role={m['role']:11s}  split={m['split'] or '-'}")

    accs: List[VideoAcc] = []
    all_frame_dfs: List[pd.DataFrame] = []
    for v in videos:
        acc, df_v = test_one_video(v, SAMPLE_ROOT, db, metas[v],
                                    frame_stride=args.frame_stride)
        accs.append(acc)
        if not df_v.empty:
            all_frame_dfs.append(df_v)

    df_log_all = pd.concat(all_frame_dfs, ignore_index=True) if all_frame_dfs else pd.DataFrame()

    # Per-cell confusion samples for the visualizer
    export_confusion_samples(accs, SAMPLE_ROOT, n_per_cell=2)

    # Excel report
    build_report(accs, df_log_all, list(db.keys()), manifest=manifest)


if __name__ == "__main__":
    main()
