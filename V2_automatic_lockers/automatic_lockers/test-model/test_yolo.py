import cv2
import time

# 1. Load model Haar Cascade
face_cascade = cv2.CascadeClassifier("V2_automatic_lockers/automatic_lockers/haarcascade_frontalface_default.xml")

# 2. FULL HD 1920x1080 cho Logitech C720
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # Full HD width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Full HD height
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)             # Force 30 FPS

# Kiểm tra resolution thực tế
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera config: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

# FPS variables - thuần túy không skip
fps_counter = 0
fps_start_time = time.time()
fps_display = "FPS: --"

print("Full HD FPS Test | 'q'=quit | 's'=save | '1'=640p | '2'=720p | '3'=1080p")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera fail, retrying...")
        time.sleep(1)
        continue

    frame_start = time.time()
    
    # Flip mirror
    frame = cv2.flip(frame, 1)
    
    # Độ phân giải thực tế
    h_frame, w_frame = frame.shape[:2]
    frame_area = w_frame * h_frame
    resolution_display = f"{w_frame}x{h_frame}"

    # Chuyển xám + Detect MỖI FRAME (không skip)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100)  # Tăng minSize cho Full HD
    )

    # Vẽ faces + ratio
    for i, (x, y, w, h) in enumerate(faces):
        face_area = w * h
        ratio = face_area / frame_area
        ratio_percent = ratio * 100
        color = (0, 255 - i*40, 0) if i < 6 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)  # Thick line Full HD
        
        cv2.putText(frame, f"{ratio_percent:.1f}%", (x, y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # === FPS PURE CALCULATION ===
    frame_end = time.time()
    fps_counter += 1
    
    # Tính FPS mỗi giây
    current_time = time.time()
    if current_time - fps_start_time >= 1.0:
        fps = fps_counter / (current_time - fps_start_time)
        fps_display = f"FPS: {fps:.1f}"
        fps_counter = 0
        fps_start_time = current_time

    # === FULL HD INFO PANEL ===
    info_bg = frame[0:180, 0:350].copy()
    overlay = info_bg.copy()
    cv2.rectangle(overlay, (0, 0), (350, 180), (20, 20, 40), -1)
    cv2.addWeighted(overlay, 0.8, info_bg, 0.2, 0, info_bg)
    frame[0:180, 0:350] = info_bg
    
    # Info lớn Full HD
    cv2.putText(info_bg, fps_display, (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
    cv2.putText(info_bg, f"Res: {resolution_display}", (15, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 2)
    cv2.putText(info_bg, f"Faces: {len(faces)}", (15, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
    cv2.putText(info_bg, f"Pixels: {frame_area/1e6:.1f}MP", (15, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
    
    frame[0:180, 0:350] = info_bg

    cv2.imshow("Full HD Face Detection + Pure FPS Test", frame)

    # Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"fullhd_test_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved Full HD: {filename}")
    elif key == ord('1'):  # 640x480
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("Switched to 640p")
    elif key == ord('2'):  # 1280x720
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("Switched to 720p")
    elif key == ord('3'):  # 1920x1080
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        print("Switched to Full HD")

cap.release()
cv2.destroyAllWindows()
print(f"Final: FPS={fps_display} | Res={resolution_display} | Faces={len(faces)}")
