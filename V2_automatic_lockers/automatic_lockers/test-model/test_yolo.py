import cv2
from ultralytics import YOLO
import numpy as np
import time
import torch
import os

print("=== YOLOv12 Face Detection Test on VIDEO ===")

# 1. Kiểm tra GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Load model
model_path = 'automatic_lockers/yolov12n-face.pt'
yolo_model = YOLO(model_path)
yolo_model.to(device)
print("Model loaded successfully!")

# 3. MỞ FILE VIDEO (Thay vì mở webcam)
video_path = 'test_locker_flipped.mp4' 

if not os.path.exists(video_path):
    print(f"LỖI: Không tìm thấy file {video_path}. Ân kiểm tra lại tên file nhé!")
else:
    cap = cv2.VideoCapture(video_path)
    print(f"Đang chạy test trên video: {video_path}")

print("Press 'q' to quit, 's' to save test image")

fps_counter = 0
start_time = time.time()
img_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    # Kiểm tra nếu hết video thì dừng hoặc reset
    if not ret:
        print("Hết video hoặc lỗi file!")
        break

    # LƯU Ý: Vì video quay trước đó đã lật rồi, nên mình KHÔNG dùng cv2.flip nữa
    # Nếu Ân muốn lật tiếp thì bỏ comment dòng dưới:
    # frame = cv2.flip(frame, 1)

    # YOLO detect
    results = yolo_model(frame, verbose=False, conf=0.5)

    # Vẽ boxes
    for res in results:
        if res.boxes is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            confidences = res.boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = map(int, box)
                label = f"Face {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Tính FPS
    fps_counter += 1
    elapsed = time.time() - start_time
    fps = fps_counter / elapsed if elapsed > 0 else 0

    cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("YOLOv12 Face Detection - Video Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Tạo thư mục imgs nếu chưa có
        if not os.path.exists('imgs'): os.makedirs('imgs')
        cv2.imwrite(f'imgs/test_face_detect_{img_count}.jpg', frame)
        print(f"Saved imgs/test_face_detect_{img_count}.jpg")
        img_count += 1

cap.release()
cv2.destroyAllWindows()
print("Test completed!")