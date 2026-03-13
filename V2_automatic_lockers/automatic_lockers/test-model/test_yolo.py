import cv2
from ultralytics import YOLO
import numpy as np
import time

# Load model (đường dẫn từ code gốc)
model_path = 'V2_automatic_lockers/automatic_lockers/yolov12n-face.pt'
yolo_model = YOLO(model_path)

# Mở webcam (index 1 cho USB cam)
cap = cv2.VideoCapture(1)  # Thay 0 nếu webcam laptop
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("=== YOLOv12 Face Detection Test ===")
print("Nhấn 'q' để thoát, 's' để lưu ảnh test")

fps_counter = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error!")
        break
    
    frame = cv2.flip(frame, 1)  # Mirror
    
    # YOLO detect
    results = yolo_model(frame, device='cpu', verbose=False, conf=0.5)
    
    # Vẽ bounding boxes
    for res in results:
        if res.boxes is not None:
            boxes = res.boxes.xyxy.cpu().numpy()  # [x1,y1,x2,y2]
            confidences = res.boxes.conf.cpu().numpy()
            
            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                x1, y1, x2, y2 = map(int, box)
                label = f"Face {conf:.2f}"
                
                # Vẽ box xanh
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # FPS counter
    fps_counter += 1
    if fps_counter % 30 == 0:
        fps = 30 / (time.time() - start_time)
        print(f"FPS: {fps:.1f}")
        start_time = time.time()
    
    # FPS hiển thị
    cv2.putText(frame, f"FPS: {fps_counter//30:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imshow("YOLOv12 Face Detection Test", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imencode('.jpg', frame)[1].tofile('test_face_detect.jpg')
        print("Saved test_face_detect.jpg")

cap.release()
cv2.destroyAllWindows()
print("Test completed!")
