import cv2
from ultralytics import YOLO
import numpy as np
import time
import torch

print("=== YOLOv12 Face Detection Test ===")

# Kiểm tra GPU
if torch.cuda.is_available():
    device = "cuda"
    print(f"CUDA available - Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("CUDA not available - Using CPU")

# Load model
model_path = 'model/yolov12n-face.pt'
model_name = 'v12n-face'
print("Loading YOLO model...")

yolo_model = YOLO(model_path)
yolo_model.to(device)

print("Model loaded successfully!")

# Mở webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # đổi thành 0 nếu dùng webcam laptop
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("Camera started!")
print("Press 'q' to quit, 's' to save test image")

fps_counter = 0
start_time = time.time()

img_count = 0

while True:

    ret, frame = cap.read()
    if not ret:
        print("Camera error!")
        break

    frame = cv2.flip(frame, 1)

    # YOLO detect
    results = yolo_model(frame, verbose=False, conf=0.5)

    # Draw boxes
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

    # FPS
    fps_counter += 1
    elapsed = time.time() - start_time
    fps = fps_counter / elapsed if elapsed > 0 else 0

    cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow(f"{model_name} Face Detection Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('s'):
        cv2.imencode('.jpg', frame)[1].tofile(f'imgs/test_face_detect_{img_count}.jpg')
        print(f"Saved imgs/test_face_detect_{img_count}.jpg")
        img_count += 1

cap.release()
cv2.destroyAllWindows()

print("Test completed!")