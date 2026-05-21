"""
calibrate_print_detector.py
============================

Chay print_attack_detector tren tat ca video trong test_videos/,
in ra cac metric trung binh per group (user/mask/spoof_phone/spoof_print)
de tune nguong cho HIGH_FREQ_ENERGY_TH, LBP_ENTROPY_TH, LAP_VAR_TH.

Output: bang so sanh + de xuat nguong toi uu.

Chay:
    python calibrate_print_detector.py
"""
import os
import sys
import cv2
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import yolo_model
from print_attack_detector import (
    _crop_face, _high_freq_energy, _lbp_entropy, _laplacian_variance
)


VIDEO_DIR = "test_videos"
N_FRAMES_PER_VIDEO = 20


def classify(name: str) -> str:
    """user_/mask_/spoof_..._phone/spoof_..._print"""
    n = name.lower()
    if n.startswith("user_"):
        return "user"
    if n.startswith("mask_"):
        return "mask"
    if n.startswith("spoof_"):
        if "_phone" in n:
            return "spoof_phone"
        if "_print" in n:
            return "spoof_print"
        return "spoof_other"
    return "unknown"


def sample_metrics(video_path: str, n_frames: int = N_FRAMES_PER_VIDEO):
    """Lay deu n frame, detect mat, tinh 3 metric. Tra ve list of dicts."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, min(n_frames, total)).astype(int)

    metrics = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, fr = cap.read()
        if not ret:
            continue

        # Detect face
        fr_rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        results = yolo_model(fr_rgb, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            continue
        confs = results[0].boxes.conf.cpu().numpy()
        xyxy = results[0].boxes.xyxy.cpu().numpy()
        i = int(np.argmax(confs))
        x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
        bbox = (x1, y1, x2 - x1, y2 - y1)

        face = _crop_face(fr, bbox)
        if face is None:
            continue
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        metrics.append({
            "hf":  _high_freq_energy(gray),
            "lbp": _lbp_entropy(gray),
            "lap": _laplacian_variance(gray),
        })
    cap.release()
    return metrics


def main():
    by_group = defaultdict(list)

    files = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")])
    print(f"Processing {len(files)} videos...")

    for f in files:
        g = classify(f)
        if g == "unknown":
            continue
        path = os.path.join(VIDEO_DIR, f)
        ms = sample_metrics(path)
        for m in ms:
            by_group[g].append(m)
        print(f"  {f:30s} -> {g:12s} ({len(ms)} frames)")

    print("\n" + "=" * 70)
    print("METRICS PER GROUP (mean ± std)")
    print("=" * 70)
    print(f"{'group':<14} {'n':>5} {'HF energy':>18} {'LBP entropy':>18} {'Lap var':>15}")
    print("-" * 70)

    stats = {}
    for g in ["user", "mask", "spoof_phone", "spoof_print"]:
        if g not in by_group:
            continue
        rows = by_group[g]
        n = len(rows)
        hf = np.array([r["hf"] for r in rows])
        lbp = np.array([r["lbp"] for r in rows])
        lap = np.array([r["lap"] for r in rows])
        stats[g] = {
            "n": n,
            "hf_mean":  float(hf.mean()),  "hf_std":  float(hf.std()),
            "lbp_mean": float(lbp.mean()), "lbp_std": float(lbp.std()),
            "lap_mean": float(lap.mean()), "lap_std": float(lap.std()),
        }
        print(f"{g:<14} {n:>5} "
              f"{hf.mean():>10.4f}±{hf.std():.4f}  "
              f"{lbp.mean():>10.2f}±{lbp.std():.2f}  "
              f"{lap.mean():>8.1f}±{lap.std():.1f}")

    print("\n" + "=" * 70)
    print("DE XUAT NGUONG (lay diem giua mean(spoof_print) va mean(user/mask))")
    print("=" * 70)

    if "spoof_print" in stats and "user" in stats:
        p = stats["spoof_print"]
        # Lay min cua user va mask de an toan
        live_mins = []
        for g in ["user", "mask"]:
            if g in stats:
                live_mins.append(stats[g])

        if live_mins:
            hf_live  = min(s["hf_mean"]  for s in live_mins)
            lbp_live = min(s["lbp_mean"] for s in live_mins)
            lap_live = min(s["lap_mean"] for s in live_mins)

            # Threshold = trung diem (geometric / arithmetic)
            hf_th  = (p["hf_mean"]  + hf_live)  / 2
            lbp_th = (p["lbp_mean"] + lbp_live) / 2
            lap_th = (p["lap_mean"] + lap_live) / 2

            print(f"HIGH_FREQ_ENERGY_TH = {hf_th:.4f}   "
                  f"(print mean={p['hf_mean']:.4f}, live min={hf_live:.4f})")
            print(f"LBP_ENTROPY_TH      = {lbp_th:.2f}   "
                  f"(print mean={p['lbp_mean']:.2f}, live min={lbp_live:.2f})")
            print(f"LAP_VAR_TH          = {lap_th:.1f}   "
                  f"(print mean={p['lap_mean']:.1f}, live min={lap_live:.1f})")

            print("\nCAP NHAT FILE print_attack_detector.py voi cac gia tri tren.")

            # Tinh accuracy uoc luong neu apply nguong
            print("\n" + "-" * 70)
            print("UOC LUONG ACCURACY KHI APPLY (per-frame, dung trong tap nay):")
            for g in ["user", "mask", "spoof_phone", "spoof_print"]:
                if g not in by_group:
                    continue
                tagged_print = 0
                for r in by_group[g]:
                    fails = 0
                    if r["hf"]  < hf_th:  fails += 1
                    if r["lbp"] < lbp_th: fails += 1
                    if r["lap"] < lap_th: fails += 1
                    if fails >= 2:
                        tagged_print += 1
                pct = 100.0 * tagged_print / len(by_group[g])
                expected = "reject" if "print" in g else "PASS"
                print(f"  {g:<14}: {tagged_print:4d}/{len(by_group[g]):4d} "
                      f"({pct:5.1f}%) tagged as print  ({expected})")
        else:
            print("Khong co data live de tinh nguong.")
    else:
        print("Khong co spoof_print hoac user data.")


if __name__ == "__main__":
    main()
