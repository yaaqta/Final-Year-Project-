"""
============================================================
 BAI TEST 1 (v3) -- FACE DETECTION (YOLO)
============================================================
Chay hang loat tren toan bo thu muc test_videos/.
Tu phan nhom theo ten file:
   - user_*   -> "normal"  (mat thuong)
   - mask_*   -> "masked"  (deo khau trang)
   - spoof_*  -> "spoof"   (gia mao)

LUU Y QUAN TRONG:
   Test goi truc tiep yolo_model() (giong app.py) thay vi
   detect_faces(), vi detect_faces() khong tra ve confidence.
   Khong sua doi app.py.

CHI SO DO:
   - Detection Rate (%)
   - Avg Confidence / Min Confidence
   - Avg Box Area (px^2)
   - Avg / Std Inference Time (ms)
   - FPS quy doi

XUAT KET QUA:
   - results/Detection_Report.xlsx       (5 sheet)
   - results/debug_frames/<video>/...    (3 frame mau co bbox)
   - results/charts/                     (6 bieu do PNG)
   - results/face_detection_results.json

Cach chay:
   python test_face_detection.py
============================================================
"""

import os
import cv2
import time
import json
import heapq
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app import yolo_model

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def detect_faces_with_conf(img_rgb):
    """Tra ve list dict {x1,y1,x2,y2,conf}."""
    results = yolo_model(img_rgb, device="cpu", verbose=False)
    out = []
    for res in results:
        if res.boxes is None:
            continue
        for box in res.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else None
            out.append({
                "x1": float(xyxy[0]), "y1": float(xyxy[1]),
                "x2": float(xyxy[2]), "y2": float(xyxy[3]),
                "conf": conf,
            })
    return out


def group_from_filename(name: str) -> str:
    base = os.path.basename(name).lower()
    if base.startswith("user"):
        return "normal"
    if base.startswith("mask"):
        return "masked"
    if base.startswith("spoof"):
        return "spoof"
    return "other"


def collect_videos(videos_dir: str):
    """Quet folder, dedupe (Windows khong phan biet hoa/thuong)."""
    seen = set()
    files = []
    for ext in VIDEO_EXTS:
        for f in glob(os.path.join(videos_dir, f"*{ext}")) + \
                 glob(os.path.join(videos_dir, f"*{ext.upper()}")):
            real = os.path.realpath(f).lower()
            if real not in seen:
                seen.add(real)
                files.append(f)
    return sorted(files)


def _label_scale(img_w):
    s = max(1.0, img_w / 640.0)
    font_scale = 1.2 * s
    text_thick = max(2, int(round(2.5 * s)))
    box_thick = max(2, int(round(2 * s)))
    return font_scale, text_thick, box_thick


def _put_label_with_bg(img, text, org, color_bgr, font_scale, text_thick):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thick)
    pad_x, pad_y = max(4, int(font_scale * 4)), max(4, int(font_scale * 3))
    x, y = org
    x = max(0, x)
    y = max(th + pad_y * 2, y)
    cv2.rectangle(
        img,
        (x, y - th - pad_y * 2),
        (x + tw + pad_x * 2, y),
        color_bgr,
        thickness=-1,
    )
    cv2.putText(
        img, text, (x + pad_x, y - pad_y),
        font, font_scale, (255, 255, 255), text_thick, cv2.LINE_AA,
    )


def draw_boxes(frame_bgr, boxes):
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    font_scale, text_thick, box_thick = _label_scale(w)
    color = (0, 200, 0)
    for b in boxes:
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thick)
        label = f"{b['conf']:.2f}" if b.get("conf") is not None else "face"
        _put_label_with_bg(img, label, (x1, y1 - 2), color, font_scale, text_thick)
    return img


def draw_boxes_with_status(frame_bgr, boxes):
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    font_scale, text_thick, box_thick = _label_scale(w)
    if len(boxes) == 0:
        border = max(6, int(round(w / 120)))
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), border)
        _put_label_with_bg(
            img, "MISS", (border + 4, int(font_scale * 40) + border),
            (0, 0, 255), font_scale * 1.2, text_thick + 1,
        )
        return img

    border = max(4, int(round(w / 180)))
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 165, 255), border)
    color = (0, 140, 255)
    for b in boxes:
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thick)
        label = f"{b['conf']:.2f}" if b.get("conf") is not None else "face"
        _put_label_with_bg(img, label, (x1, y1 - 2), color, font_scale, text_thick)
    return img


def test_one_video(video_path: str, debug_dir: str, sample_dir: str = None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    debug_indices = set()
    if n_total >= 3:
        debug_indices = {1, n_total // 2, max(1, n_total - 2)}

    total_frames = 0
    detected_frames = 0
    confs, areas, times_ms = [], [], []
    saved_debug = 0
    per_frame = []

    K_BEST = 3
    K_WORST = 2
    best_heap = []
    worst_heap = []
    miss_list = []

    os.makedirs(debug_dir, exist_ok=True)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        total_frames += 1

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        boxes = detect_faces_with_conf(frame_rgb)
        t_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(t_ms)

        n_box = len(boxes)
        best_conf = None
        best_area = 0
        if n_box > 0:
            detected_frames += 1
            box_areas = [(b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) for b in boxes]
            best_idx = int(np.argmax(box_areas))
            best_area = box_areas[best_idx]
            areas.append(best_area)
            if boxes[best_idx]["conf"] is not None:
                best_conf = boxes[best_idx]["conf"]
                confs.append(best_conf)

        per_frame.append({
            "video": os.path.basename(video_path),
            "group": group_from_filename(video_path),
            "frame_idx": total_frames,
            "detected": n_box > 0,
            "num_boxes": n_box,
            "best_confidence": best_conf,
            "best_box_area": best_area,
            "infer_time_ms": round(t_ms, 3),
        })

        if sample_dir is not None:
            if best_conf is not None:
                if len(best_heap) < K_BEST:
                    heapq.heappush(best_heap, (best_conf, total_frames, frame_bgr.copy(), boxes))
                else:
                    if best_conf > best_heap[0][0]:
                        heapq.heapreplace(best_heap, (best_conf, total_frames, frame_bgr.copy(), boxes))

            if n_box == 0 and len(miss_list) < K_WORST:
                miss_list.append((total_frames, frame_bgr.copy(), boxes))
            elif n_box > 0 and best_conf is not None:
                neg = -best_conf
                if len(worst_heap) < K_WORST:
                    heapq.heappush(worst_heap, (neg, total_frames, frame_bgr.copy(), boxes))
                else:
                    if neg > worst_heap[0][0]:
                        heapq.heapreplace(worst_heap, (neg, total_frames, frame_bgr.copy(), boxes))

        if total_frames in debug_indices and saved_debug < 3:
            out_img = draw_boxes(frame_bgr, boxes)
            cv2.imwrite(os.path.join(debug_dir, f"frame_{total_frames:05d}.jpg"), out_img)
            saved_debug += 1

    cap.release()

    if sample_dir is not None:
        name_no_ext = os.path.splitext(os.path.basename(video_path))[0]
        out_sample = os.path.join(sample_dir, name_no_ext)
        os.makedirs(out_sample, exist_ok=True)

        best_sorted = sorted(best_heap, key=lambda x: -x[0])
        for i, (c, idx, fr, bxs) in enumerate(best_sorted, 1):
            img = draw_boxes(fr, bxs)
            cv2.imwrite(os.path.join(out_sample, f"best_{i}.jpg"), img)

        worst_samples = []
        for idx, fr, bxs in miss_list[:K_WORST]:
            worst_samples.append((None, idx, fr, bxs))
        if len(worst_samples) < K_WORST:
            worst_sorted = sorted(worst_heap, key=lambda x: -x[0])
            for neg, idx, fr, bxs in worst_sorted:
                if len(worst_samples) >= K_WORST:
                    break
                worst_samples.append((-neg, idx, fr, bxs))
        for i, (c, idx, fr, bxs) in enumerate(worst_samples, 1):
            img = draw_boxes_with_status(fr, bxs)
            cv2.imwrite(os.path.join(out_sample, f"worst_{i}.jpg"), img)

    detection_rate = (detected_frames / total_frames * 100.0) if total_frames else 0.0
    avg_time = float(np.mean(times_ms)) if times_ms else 0.0
    fps_infer = (1000.0 / avg_time) if avg_time > 0 else 0.0

    summary = {
        "video": os.path.basename(video_path),
        "group": group_from_filename(video_path),
        "video_fps": round(fps_video, 2),
        "total_frames": total_frames,
        "detected_frames": detected_frames,
        "detection_rate(%)": round(detection_rate, 2),
        "avg_confidence": round(float(np.mean(confs)), 3) if confs else None,
        "min_confidence": round(float(np.min(confs)), 3) if confs else None,
        "avg_box_area(px2)": int(np.mean(areas)) if areas else 0,
        "avg_infer_time(ms)": round(avg_time, 2),
        "std_infer_time(ms)": round(float(np.std(times_ms)), 2) if times_ms else 0.0,
        "fps_infer": round(fps_infer, 2),
    }
    return summary, per_frame


def build_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in ["normal", "masked", "spoof"]:
        sub = df[df["group"] == g]
        if len(sub) == 0:
            continue
        rows.append({
            "Group": g,
            "So video": len(sub),
            "Avg Detection Rate (%)": round(sub["detection_rate(%)"].mean(), 2),
            "Min Detection Rate (%)": round(sub["detection_rate(%)"].min(), 2),
            "Avg Confidence": round(sub["avg_confidence"].dropna().mean(), 3)
                              if sub["avg_confidence"].notna().any() else None,
            "Avg Inference Time (ms)": round(sub["avg_infer_time(ms)"].mean(), 2),
            "Avg FPS": round(sub["fps_infer"].mean(), 2),
        })
    return pd.DataFrame(rows)


def build_comparison(group_df: pd.DataFrame) -> pd.DataFrame:
    if len(group_df) < 2:
        return pd.DataFrame()
    g = group_df.set_index("Group")
    rows = []

    pairs = [("normal", "masked"), ("normal", "spoof"), ("masked", "spoof")]
    metrics = [
        ("Avg Detection Rate (%)", "Detection Rate", "%"),
        ("Avg Confidence", "Confidence", ""),
        ("Avg Inference Time (ms)", "Time", "ms"),
        ("Avg FPS", "FPS", ""),
    ]

    for g1, g2 in pairs:
        if g1 not in g.index or g2 not in g.index:
            continue
        for col, label, unit in metrics:
            v1 = g.loc[g1, col]
            v2 = g.loc[g2, col]
            if pd.isna(v1) or pd.isna(v2):
                continue
            delta = v2 - v1
            pct = (delta / v1 * 100.0) if v1 else 0.0
            rows.append({
                "So sanh": f"{g1} vs {g2}",
                "Chi so": label,
                g1: f"{v1} {unit}".strip(),
                g2: f"{v2} {unit}".strip(),
                "Thay doi": f"{delta:+.2f} {unit}".strip(),
                "% thay doi": f"{pct:+.1f}%",
            })
    return pd.DataFrame(rows)


def build_methodology() -> pd.DataFrame:
    rows = [
        ("Detection Rate (%)", "(so frame co >=1 bounding box) / (tong so frame) * 100",
         "Do nhay - YOLO phat hien duoc bao nhieu phan tram frame co mat"),
        ("Avg Confidence", "mean(confidence cua box lon nhat moi frame)",
         "Do tu tin trung binh cua model voi ket qua tra ve"),
        ("Min Confidence", "min(confidence) qua tat ca frame co detect",
         "Truong hop xau nhat - frame ma model luong lu nhat"),
        ("Avg Box Area (px2)", "mean((x2-x1) * (y2-y1)) qua cac frame co mat",
         "Kich thuoc mat trung binh - kiem soat khoang cach quay"),
        ("Avg Inference Time (ms)", "mean(time.perf_counter() truoc/sau yolo_model()) * 1000",
         "Thoi gian xu ly trung binh 1 frame -- quyet dinh realtime hay khong"),
        ("Std Inference Time (ms)", "std(inference times)",
         "Do on dinh - std thap = thoi gian xu ly deu"),
        ("FPS Inference", "1000 / Avg Inference Time (ms)",
         "So frame xu ly duoc moi giay"),
    ]
    return pd.DataFrame(rows, columns=["Chi so", "Cong thuc", "Y nghia"])


def make_charts(df_video, df_group, df_frames, charts_dir):
    os.makedirs(charts_dir, exist_ok=True)
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    GC = {"normal": "#2E86DE", "masked": "#E67E22", "spoof": "#C0392B"}
    colors = [GC.get(g, "#7f8c8d") for g in df_video["group"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(df_video["video"], df_video["detection_rate(%)"], color=colors, edgecolor="black")
    ax.axhline(95, color="red", linestyle="--", linewidth=1, label="Target 95%")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Chart 1 - Detection Rate per video")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "01_detection_rate_per_video.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(df_video["video"], df_video["avg_infer_time(ms)"],
                  yerr=df_video["std_infer_time(ms)"], capsize=4,
                  color=colors, edgecolor="black")
    ax.axhline(40, color="red", linestyle="--", linewidth=1, label="Target 40 ms")
    ax.set_ylabel("Inference Time (ms)")
    ax.set_title("Chart 2 - Avg Inference Time per video")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "02_inference_time_per_video.png"))
    plt.close(fig)

    if df_frames["best_confidence"].notna().any():
        fig, ax = plt.subplots(figsize=(10, 5))
        for grp in ["normal", "masked", "spoof"]:
            confs = df_frames[(df_frames["group"] == grp) &
                              (df_frames["best_confidence"].notna())]["best_confidence"]
            if len(confs) > 0:
                ax.hist(confs, bins=30, alpha=0.6, label=grp.capitalize(),
                        color=GC.get(grp, "#7f8c8d"), edgecolor="black")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("So frame")
        ax.set_title("Chart 3 - Confidence histogram")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "03_confidence_histogram.png"))
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", default="test_videos")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    if not os.path.isdir(args.videos_dir):
        print(f"[ERROR] Khong tim thay thu muc: {args.videos_dir}")
        return

    files = collect_videos(args.videos_dir)
    files = [f for f in files if group_from_filename(f) in ("normal", "masked", "spoof")]
    if not files:
        print(f"[ERROR] Khong co file user_*/mask_*/spoof_* trong {args.videos_dir}")
        return

    print(f"\n[INFO] Tim thay {len(files)} video de test:")
    for f in files:
        print(f"   - {os.path.basename(f)}  ({group_from_filename(f)})")
    print()

    os.makedirs(args.out_dir, exist_ok=True)
    debug_root = os.path.join(args.out_dir, "debug_frames")
    charts_dir = os.path.join(args.out_dir, "charts")
    sample_root = os.path.join(args.out_dir, "sample_frames")
    os.makedirs(debug_root, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(sample_root, exist_ok=True)

    summaries = []
    frame_logs = []

    for i, video_path in enumerate(files, 1):
        name_no_ext = os.path.splitext(os.path.basename(video_path))[0]
        print(f"[{i}/{len(files)}] Dang xu ly: {os.path.basename(video_path)} ...")
        t0 = time.time()
        summary, frames = test_one_video(
            video_path,
            debug_dir=os.path.join(debug_root, name_no_ext),
            sample_dir=sample_root,
        )
        elapsed = time.time() - t0
        if summary is None:
            print("   [LOI] Khong mo duoc video\n")
            continue

        print(f"   Detection Rate: {summary['detection_rate(%)']}%"
              f"   |   Time/frame: {summary['avg_infer_time(ms)']} ms"
              f"   |   Conf: {summary['avg_confidence']}"
              f"   |   Xu ly mat {elapsed:.1f}s\n")
        summaries.append(summary)
        frame_logs.extend(frames)

    if not summaries:
        print("[WARN] Khong co ket qua nao.")
        return

    df_video = pd.DataFrame(summaries)
    df_group = build_group_summary(df_video)
    df_compare = build_comparison(df_group)
    df_method = build_methodology()
    df_frames = pd.DataFrame(frame_logs)

    print("[INFO] Dang ve bieu do bang matplotlib ...")
    make_charts(df_video, df_group, df_frames, charts_dir)

    out_xlsx = os.path.join(args.out_dir, "Detection_Report.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_video.to_excel(writer, sheet_name="Per_Video", index=False)
        if not df_group.empty:
            df_group.to_excel(writer, sheet_name="Group_Summary", index=False)
        if not df_compare.empty:
            df_compare.to_excel(writer, sheet_name="Comparison", index=False)
        df_method.to_excel(writer, sheet_name="Methodology", index=False)
        df_frames.to_excel(writer, sheet_name="Frame_Level_Detail", index=False)

    out_json = os.path.join(args.out_dir, "face_detection_results.json")
    payload = [{**r, "test_name": "Face Detection (YOLO)",
                "timestamp": datetime.now().isoformat(timespec="seconds")}
               for r in summaries]
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n========== DA XUAT BAO CAO ==========")
    print(f"  Excel   : {out_xlsx}    (5 sheet)")
    print(f"  Charts  : {charts_dir}/")
    print(f"  Debug   : {debug_root}/")
    print(f"  Samples : {sample_root}/")
    print(f"  JSON    : {out_json}")
    print("======================================\n")

    if not df_group.empty:
        print("BANG TONG HOP THEO NHOM:")
        print(df_group.to_string(index=False))
        print()
    if not df_compare.empty:
        print("SO SANH CAC NHOM:")
        print(df_compare.to_string(index=False))
        print()


if __name__ == "__main__":
    main()