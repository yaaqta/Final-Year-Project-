import cv2

cap = cv2.VideoCapture(0)

# Cấu hình file video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('test_locker_flipped.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_width, frame_height))

recording = False 

print("--- ĐÃ CẬP NHẬT LẬT CAMERA ---")
print("Nhấn 'S' để BẮT ĐẦU quay.")
print("Nhấn 'Q' để DỪNG và LƯU.")

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret: break

    # LẬT CAMERA TẠI ĐÂY: 1 là lật theo trục dọc (trái sang phải)
    frame = cv2.flip(frame, 1)

    display_frame = frame.copy()
    if not recording:
        cv2.putText(display_frame, "MIRROR MODE - Press 'S' to Start", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display_frame, "RECORDING... Press 'Q' to Stop", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        out.write(frame) 

    cv2.imshow('Smart Locker Video Recorder', display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'): 
        recording = True
    elif key == ord('q'): 
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Done!")