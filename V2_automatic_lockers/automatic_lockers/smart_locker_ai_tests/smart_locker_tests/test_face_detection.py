"""
============================================================
 BÀI TEST 1 — FACE DETECTION (YOLO)
============================================================
Mục đích: Kiểm tra riêng khả năng PHÁT HIỆN khuôn mặt của YOLO.
          Không quan tâm mặt đó là của ai, chỉ đo tỷ lệ bắt được mặt.

Chỉ số đo:
  - Detection Rate (%) = số frame YOLO bắt được mặt / tổng số frame có mặt
  - Avg Inference Time (ms) — thời gian trung bình 1 lần detect
  - FPS quy đổi

Cách chạy:
  python test_face_detection.py --video test_videos/video_user.mp4 --label co_mat
  python test_face_detection.py --video test_videos/video_empty.mp4 --label khong_co_mat

Tham số:
  --video : đường dẫn video đầu vào
  --label : "co_mat" nếu video CÓ khuôn mặt, "khong_co_mat" nếu KHÔNG có
            (dùng để đo cả False Positive khi quay cảnh trống)
============================================================
"""

import os
import cv2
import time
import json
import argparse
import numpy as np
from datetime import datetime

# Tái sử dụng hàm có sẵn trong app.py
from app import detect_faces


def run_detection_test(video_path: str, label: str, conf_threshold: float = 0.5):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 0

    total_frames = 0
    detected_frames = 0      # số frame có ÍT NHẤT 1 box
    false_positive = 0       # áp dụng khi label = khong_co_mat
    inference_times_ms = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        boxes = detect_faces(frame_rgb)
        t_infer = (time.perf_counter() - t0) * 1000.0   # ms
        inference_times_ms.append(t_infer)

        has_face = len(boxes) > 0
        if has_face:
            detected_frames += 1
            if label == "khong_co_mat":
                false_positive += 1

    cap.release()

    avg_time = float(np.mean(inference_times_ms)) if inference_times_ms else 0.0
    std_time = float(np.std(inference_times_ms)) if inference_times_ms else 0.0
    fps_infer = 1000.0 / avg_time if avg_time > 0 else 0.0

    if label == "co_mat":
        detection_rate = (detected_frames / total_frames * 100.0) if total_frames else 0.0
        miss_rate = 100.0 - detection_rate
        false_positive_rate = None
    else:
        detection_rate = None
        miss_rate = None
        false_positive_rate = (false_positive / total_frames * 100.0) if total_frames else 0.0

    result = {
        "test_name": "Face Detection (YOLO)",
        "video": os.path.basename(video_path),
        "label": label,
        "video_fps": round(fps_video, 2),
        "total_frames": total_frames,
        "detected_frames": detected_frames,
        "detection_rate(%)": round(detection_rate, 2) if detection_rate is not None else "N/A",
        "miss_rate(%)": round(miss_rate, 2) if miss_rate is not None else "N/A",
        "false_positive_rate(%)": round(false_positive_rate, 2) if false_positive_rate is not None else "N/A",
        "avg_infer_time(ms)": round(avg_time, 2),
        "std_infer_time(ms)": round(std_time, 2),
        "fps_infer": round(fps_infer, 2),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    return result


def save_result(result: dict, out_dir: str = "results"):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "face_detection_results.json")

    history = []
    if os.path.isfile(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(result)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Test riêng Face Detection (YOLO)")
    parser.add_argument("--video", required=True, help="Đường dẫn file video .mp4")
    parser.add_argument("--label", required=True, choices=["co_mat", "khong_co_mat"],
                        help="Video CÓ khuôn mặt (co_mat) hay KHÔNG có (khong_co_mat)")
    args = parser.parse_args()

    print(f"\n[INFO] Đang test Face Detection trên video: {args.video}")
    result = run_detection_test(args.video, args.label)

    print("\n========== KẾT QUẢ FACE DETECTION ==========")
    for k, v in result.items():
        print(f"  {k:<28}: {v}")
    print("============================================\n")

    path = save_result(result)
    print(f"[SAVED] Kết quả đã được lưu vào: {path}\n")


if __name__ == "__main__":
    main()
