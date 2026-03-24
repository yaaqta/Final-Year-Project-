import cv2
import numpy as np
import json
import time
import os
from scipy.spatial.distance import cosine
from facenet_pytorch import MTCNN, InceptionResnetV1

# Models
device = 'cpu' # Nếu có GPU Ân đổi thành 'cuda' nhé
mtcnn = MTCNN(image_size=160, margin=20, min_face_size=40, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("✅ Models OK")

DB_FILE = "face_lockers.json"
lockers = {str(i): {"assigned": None, "status": "available"} for i in range(1, 9)}
faces_db = []

def load_db():
    global faces_db, lockers
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f:
                data = json.load(f)
                faces_db = data.get("faces", [])
                lockers.update(data.get("lockers", {}))
                print(f"✅ Loaded {len(faces_db)} faces from DB")
        except: pass

def save_db():
    with open(DB_FILE, "w") as f:
        json.dump({"faces": faces_db, "lockers": lockers}, f)

load_db()

def get_embedding(frame):
    # Trích xuất khuôn mặt
    face = mtcnn(frame)
    if face is None: return None
    # Tạo vector đặc trưng (Embedding)
    emb = resnet(face.unsqueeze(0)).detach().cpu().numpy().flatten()
    return emb / np.linalg.norm(emb)

def checkin(user, locker, frame):
    global lockers
    if lockers[str(locker)]["status"] != "available":
        return False, f"L{locker} busy"
    
    emb = get_embedding(frame)
    if emb is None: return False, "No face"
    
    faces_db.append({"user": user, "locker": locker, "emb": emb.tolist()})
    lockers[str(locker)] = {"assigned": user, "status": "occupied"}
    save_db()
    return True, f"L{locker} OK"

def checkout(frame):
    emb = get_embedding(frame)
    if emb is None: return None, "No face"
    
    best = None
    min_dist = 999
    for f in faces_db:
        dist = cosine(emb, np.array(f["emb"]))
        if dist < min_dist:
            min_dist = dist
            best = f
    
    # Ngưỡng cosine distance 0.4 tương đương độ khớp ~80%
    if best and min_dist < 0.4:
        lid = best["locker"]
        conf = (1 - min_dist) * 100
        lockers[str(lid)]["assigned"] = None
        lockers[str(lid)]["status"] = "available"
        save_db()
        return best, f"L{lid} {conf:.0f}%"
    return None, "FAIL"

# --- MAIN: ĐỌC TỪ VIDEO ---
video_path = 'test_locker_flipped.mp4'
if not os.path.exists(video_path):
    print(f"Không tìm thấy file {video_path}!")
    exit()

cap = cv2.VideoCapture(video_path)

mode = "scan"
locker = 1
user = "User01"
msg = "Waiting..."

print("C=CheckIn O=CheckOut 1-8=Locker U=User Q=Quit")

while True:
    ret, frame = cap.read()
    
    # Tự động lặp lại video nếu hết (Để Ân test liên tục)
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    
    # KHÔNG lật gương nữa vì video gốc đã lật rồi
    # frame = cv2.flip(frame, 1) 
    
    # Xử lý logic
    if mode == "checkin":
        ok, txt = checkin(user, locker, frame)
        msg = txt
        mode = "scan"
    elif mode == "checkout":
        match, txt = checkout(frame)
        msg = txt
        mode = "scan"
    
    # --- UI: Vẽ trạng thái tủ ---
    h, w = frame.shape[:2]
    y_pos = h - 90
    for i in range(1, 9):
        x_pos = int(40 + (w-80)*(i-1)/8)
        data = lockers[str(i)]
        col = (0,255,0) if data["status"]=="available" else (0,0,255)
        cv2.rectangle(frame, (x_pos-15, y_pos), (x_pos+15, y_pos+30), col, -1)
        cv2.putText(frame, f"L{i}", (x_pos-10, y_pos+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        if data["assigned"]:
             cv2.putText(frame, str(data["assigned"])[:2], (x_pos-12, y_pos+45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
    
    # Bảng thông báo
    cv2.rectangle(frame, (10, 10), (250, 75), (50, 50, 60), -1)
    cv2.putText(frame, f"Mode: {mode}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, msg, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"Target L{locker} for {user}", (260, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    
    cv2.imshow("Smart Locker Identify System", frame)
    
    k = cv2.waitKey(20) & 0xFF # Tăng thời gian delay một chút cho mượt
    if k == ord('q'): break
    elif k == ord('c'): mode = "checkin"
    elif k == ord('o'): mode = "checkout"
    elif ord('1') <= k <= ord('8'): locker = int(chr(k))
    elif k == ord('u'): user = f"U{int(time.time()%100):02d}"

cap.release()
cv2.destroyAllWindows()
print("Done!")