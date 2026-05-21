"""
============================================================
 BÀI TEST 3 — ANTI-SPOOFING (Silent-Face)
============================================================
Mục đích: Đánh giá khả năng chống giả mạo của mô hình Silent-Face.

Chỉ số đo (theo chuẩn ISO/IEC 30107-3):
  - APCER (Attack Presentation Classification Error Rate):
        Tỷ lệ ảnh GIẢ bị NHẦM thành THẬT (càng thấp càng tốt).
  - BPCER (Bona Fide Presentation Classification Error Rate):
        Tỷ lệ ảnh THẬT bị NHẦM thành GIẢ (càng thấp càng tốt).
  - ACER = (APCER + BPCER) / 2  -> Chỉ số tổng hợp.
  - Avg Inference Time (ms)
  - Avg Liveness Score

Cách chạy:
  # Video người thật
  python test_anti_spoofing.py --video test_videos/video_user.mp4 --label real

  # Video phát lại trên điện thoại / in ảnh
  python test_anti_spoofing.py --video test_videos/video_spoof.mp4 --label spoof

Tham số:
  --video : đường dẫn video
  --label : "real" (người thật) | "spoof" (ảnh/màn hình)
============================================================
"""

import os
import cv2
import time
import json
import argparse
import numpy as np
from datetime import datetime

# Tái sử dụng các hàm có sẵn trong app.py
from app import detect_faces, crop_face_bgr, check_liveness


def run_anti_spoofing_test(video_path: str, label: str):
    assert label in ("real", "spoof"), "label phải là 'real' hoặc 'spoof'"
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

    cap = cv2.VideoCapture(video_path)

    total_frames = 0
    yolo_detected = 0
    correct = 0
    apcer_count = 0        # spoof bị nhận thành real
    bpcer_count = 0        # real bị nhận thành spoof
    liveness_scores = []
    inference_times_ms = []

    true_is_real = (label == "real")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        boxes = detect_faces(frame_rgb)
        if len(boxes) == 0:
            continue
        yolo_detected += 1

        face_bgr = crop_face_bgr(frame_rgb, boxes)

        t0 = time.perf_counter()
        is_real, live_score = check_liveness(face_bgr)
        t_infer = (time.perf_counter() - t0) * 1000.0
        inference_times_ms.append(t_infer)

        try:
            liveness_scores.append(float(live_score))
        except Exception:
            pass

        if bool(is_real) == true_is_real:
            correct += 1
        else:
            if not true_is_real and bool(is_real):
                # spoof bị nhận thành real
                apcer_count += 1
            elif true_is_real and not bool(is_real):
                # real bị nhận thành spoof
                bpcer_count += 1

    cap.release()

    n = yolo_detected
    accuracy = (correct / n * 100.0) if n else 0.0

    # Lưu ý: APCER và BPCER chỉ có ý nghĩa trong tập tương ứng.
    # Chỉ một trong hai sẽ khác 0 trong 1 lần chạy (vì video chỉ 1 nhãn).
    apcer = (apcer_count / n * 100.0) if (n and not true_is_real) else None
    bpcer = (bpcer_count / n * 100.0) if (n and true_is_real) else None

    avg_score = float(np.mean(liveness_scores)) if liveness_scores else 0.0
    avg_time = float(np.mean(inference_times_ms)) if inference_times_ms else 0.0

    result = {
        "test_name": "Anti-Spoofing (Silent-Face)",
        "video": os.path.basename(video_path),
        "label": label,
        "total_frames": total_frames,
        "yolo_detected_frames": yolo_detected,
        "correct_predictions": correct,
        "accuracy(%)": round(accuracy, 2),
        "APCER(%)": round(apcer, 2) if apcer is not None else "N/A (chỉ tính trên video spoof)",
        "BPCER(%)": round(bpcer, 2) if bpcer is not None else "N/A (chỉ tính trên video real)",
        "avg_liveness_score": round(avg_score, 4),
        "avg_infer_time(ms)": round(avg_time, 2),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    return result


def save_result(result: dict, out_dir: str = "results"):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "anti_spoofing_results.json")

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
    parser = argparse.ArgumentParser(description="Test riêng Anti-Spoofing (Silent-Face)")
    parser.add_argument("--video", required=True, help="Đường dẫn file video .mp4")
    parser.add_argument("--label", required=True, choices=["real", "spoof"],
                        help="real = người thật | spoof = ảnh/màn hình giả mạo")
    args = parser.parse_args()

    print(f"\n[INFO] Đang test Anti-Spoofing trên video: {args.video}")
    print(f"[INFO] True label: {args.label}")

    result = run_anti_spoofing_test(args.video, args.label)

    print("\n========== KẾT QUẢ ANTI-SPOOFING ==========")
    for k, v in result.items():
        print(f"  {k:<28}: {v}")
    print("============================================\n")

    path = save_result(result)
    print(f"[SAVED] Kết quả đã được lưu vào: {path}\n")
    print("Lưu ý: Để tính ACER = (APCER + BPCER)/2, bạn cần chạy file này 2 lần")
    print("       (1 lần với video real, 1 lần với video spoof), rồi chạy generate_report.py.\n")


if __name__ == "__main__":
    main()
