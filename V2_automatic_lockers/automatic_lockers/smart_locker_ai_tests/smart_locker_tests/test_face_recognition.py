"""
============================================================
 BÀI TEST 2 — FACE RECOGNITION (FaceNet)
============================================================
Mục đích: Kiểm tra riêng khả năng SO KHỚP danh tính.
          Đầu vào là các khuôn mặt đã được YOLO crop sẵn, đẩy vào FaceNet để
          so với database -> trả về tên hoặc Unknown.

Chỉ số đo:
  - Accuracy (%)
  - FAR (False Acceptance Rate)  : nhận nhầm người lạ thành đã đăng ký
  - FRR (False Rejection Rate)   : người đã đăng ký lại bị nhận thành Unknown
  - Avg Cosine Distance          : khoảng cách trung bình tới embedding gần nhất
  - Avg Inference Time (ms)

Cách chạy:
  # Video người đã đăng ký
  python test_face_recognition.py --video test_videos/video_user.mp4 --label "Nguyen Thi Huong"

  # Video người lạ (chưa có trong DB) -> kỳ vọng kết quả = Unknown
  python test_face_recognition.py --video test_videos/video_stranger.mp4 --label "Unknown"

Tham số:
  --video : đường dẫn video
  --label : tên người THẬT SỰ trong video (hoặc "Unknown" nếu người lạ)
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
from app import (
    detect_faces,
    get_face_embedding_from_image,
    recognize_face,
)


def run_recognition_test(video_path: str, true_label: str):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

    cap = cv2.VideoCapture(video_path)

    total_frames = 0
    yolo_detected = 0          # frame YOLO bắt được mặt -> mới đem đi recognize
    correct = 0                # predicted == true_label
    false_accept = 0           # true=Unknown nhưng predicted ra 1 tên trong DB
    false_reject = 0           # true=tên thật nhưng predicted=Unknown
    cosine_distances = []
    inference_times_ms = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Bắt khuôn mặt trước (chỉ recognize trên frame có mặt)
        boxes = detect_faces(frame_rgb)
        if len(boxes) == 0:
            continue
        yolo_detected += 1

        t0 = time.perf_counter()
        emb = get_face_embedding_from_image(frame_rgb, boxes)
        name, conf = recognize_face(emb)
        t_infer = (time.perf_counter() - t0) * 1000.0
        inference_times_ms.append(t_infer)

        predicted = name if name else "Unknown"

        # conf trong app.py thường là cosine distance (càng nhỏ càng giống)
        try:
            cosine_distances.append(float(conf))
        except Exception:
            pass

        if predicted == true_label:
            correct += 1
        else:
            if true_label == "Unknown" and predicted != "Unknown":
                false_accept += 1
            elif true_label != "Unknown" and predicted == "Unknown":
                false_reject += 1
            # Trường hợp predicted ra tên KHÁC tên thật -> tính là sai (impostor)
            elif true_label != "Unknown" and predicted != "Unknown" and predicted != true_label:
                false_accept += 1

    cap.release()

    n = yolo_detected
    accuracy = (correct / n * 100.0) if n else 0.0
    far = (false_accept / n * 100.0) if n else 0.0
    frr = (false_reject / n * 100.0) if n else 0.0
    avg_dist = float(np.mean(cosine_distances)) if cosine_distances else 0.0
    avg_time = float(np.mean(inference_times_ms)) if inference_times_ms else 0.0

    result = {
        "test_name": "Face Recognition (FaceNet)",
        "video": os.path.basename(video_path),
        "true_label": true_label,
        "total_frames": total_frames,
        "yolo_detected_frames": yolo_detected,
        "correct_predictions": correct,
        "accuracy(%)": round(accuracy, 2),
        "FAR(%)": round(far, 2),
        "FRR(%)": round(frr, 2),
        "avg_cosine_distance": round(avg_dist, 4),
        "avg_infer_time(ms)": round(avg_time, 2),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    return result


def save_result(result: dict, out_dir: str = "results"):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "face_recognition_results.json")

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
    parser = argparse.ArgumentParser(description="Test riêng Face Recognition (FaceNet)")
    parser.add_argument("--video", required=True, help="Đường dẫn file video .mp4")
    parser.add_argument("--label", required=True,
                        help="Tên thật của người trong video, hoặc 'Unknown' nếu người lạ")
    args = parser.parse_args()

    print(f"\n[INFO] Đang test Face Recognition trên video: {args.video}")
    print(f"[INFO] True label: {args.label}")

    result = run_recognition_test(args.video, args.label)

    print("\n========== KẾT QUẢ FACE RECOGNITION ==========")
    for k, v in result.items():
        print(f"  {k:<28}: {v}")
    print("==============================================\n")

    path = save_result(result)
    print(f"[SAVED] Kết quả đã được lưu vào: {path}\n")


if __name__ == "__main__":
    main()
