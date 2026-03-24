import cv2
import numpy as np
import json
import time
import torch
import os
from scipy.spatial.distance import cosine
from facenet_pytorch import MTCNN, InceptionResnetV1

print("Loading FaceNet... (Tối ưu cho VIDEO)")

# TỐI ƯU CPU
torch.set_num_threads(2)
torch.backends.cudnn.benchmark = False

# Load models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=20, device=device, keep_all=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Warmup
dummy_frame = torch.zeros((1, 3, 160, 160)).to(device)
try:
    _ = mtcnn(dummy_frame)
    _ = resnet(dummy_frame)
    print(f"Models OK - Running on {device}! ⚡")
except: pass

# Data
data_file = "lockers.json"
lockers = {str(i):{"user":None,"status":"free"} for i in range(1,9)}
faces = []

def load():
    global faces, lockers
    if os.path.exists(data_file):
        try:
            with open(data_file,'r') as f:
                d = json.load(f)
                faces = d["faces"]
                lockers.update(d["lockers"])
            print(f"✅ Loaded {len(faces)} faces")
        except: print("📄 New system initiated")

def save():
    with open(data_file,'w') as f:
        json.dump({"faces":faces,"lockers":lockers}, f, indent=2)

load()

def get_face_emb(frame):
    face = mtcnn(frame)
    if face is None: return None
    emb = resnet(face.unsqueeze(0)).detach().cpu().numpy().flatten()
    return emb/np.linalg.norm(emb)

def checkin(name, lid, frame):
    if lockers[str(lid)]["status"] != "free":
        return False, f"L{lid} BUSY"
    emb = get_face_emb(frame)
    if emb is None: return False, "❌ No face detected"
    
    faces.append({"name":name,"lid":lid,"emb":emb.tolist()})
    lockers[str(lid)] = {"user":name,"status":"busy"}
    save()
    return True, f"✅ L{lid} CHECKIN OK - {name}"

def check_out(frame):
    emb = get_face_emb(frame)
    if emb is None: return None, "❌ No face detected"
    
    best = None
    mind = 1.0
    for f in faces:
        d = cosine(emb, np.array(f["emb"]))
        if d < mind and d < 0.4:
            mind = d
            best = f
    
    if best and (1-mind)>0.8:
        lid = best["lid"]
        faces[:] = [face for face in faces if face["lid"] != lid]
        lockers[str(lid)]["user"] = None
        lockers[str(lid)]["status"] = "free"
        save()
        return best, f"✅ L{lid} CHECKOUT {int((1-mind)*100)}%"
    return None, "❌ No match (<80%)"

# --- THAY ĐỔI TẠI ĐÂY: ĐỌC TỪ VIDEO ---
video_path = 'test_locker_flipped.mp4'
if not os.path.exists(video_path):
    print(f"❌ Không tìm thấy file {video_path}!")
    exit()

cap = cv2.VideoCapture(video_path)
mode = "scan"
lid = 1
name = "User01"
msg = "Waiting for command..."

print("\n" + "="*50)
print("🚀 FACE LOCKER SYSTEM v2.0 - VIDEO MODE")
print("="*50)
print("C = Checkin    O = Checkout")
print("1-8 = Locker   N = New User Name")
print("Q = Quit")
print("="*50)

while True:
    ok, frame = cap.read()
    
    # TỰ ĐỘNG LẶP VIDEO KHI HẾT
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    
    # KHÔNG DÙNG cv2.flip VÌ VIDEO ĐÃ LẬT RỒI
    
    # Logic xử lý
    if mode == "checkin":
        suc, txt = checkin(name, lid, frame)
        msg = txt
        mode = "scan"
    elif mode == "checkout":
        mat, txt = check_out(frame)
        msg = txt
        mode = "scan"
    
    # --- VẼ GIAO DIỆN (Giữ nguyên phần vẽ của Ân) ---
    h, w = frame.shape[:2]
    for i in range(1, 9):
        x = int(30 + (w-60) * (i-1) / 8)
        d = lockers[str(i)]
        col = (0, 255, 0) if d["status"] == "free" else (0, 0, 255)
        cv2.rectangle(frame, (x-15, h-90), (x+15, h-40), col, -1)
        cv2.putText(frame, f"L{i}", (x-10, h-65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        if d["user"]:
            cv2.putText(frame, d["user"][:5], (x-12, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
    
    cv2.rectangle(frame, (10, 10), (w-10, 110), (50, 50, 50), -1)
    cv2.putText(frame, f"MODE: {mode.upper()}", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, msg, (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"TARGET: L{lid} | USER: {name}", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    cv2.imshow("Smart Locker Identify System", frame)
    
    k = cv2.waitKey(20) & 0xFF # Tăng delay để video chạy đúng tốc độ
    if k == ord('q'): break
    elif k == ord('c'): mode = "checkin"
    elif k == ord('o'): mode = "checkout"
    elif ord('1') <= k <= ord('8'): 
        lid = int(chr(k))
        msg = f"Selected Locker L{lid}"
    elif k == ord('n'):
        name = f"U{int(time.time()%100):02d}"
        msg = f"New user name: {name}"

cap.release()
cv2.destroyAllWindows()
save()
print("\n✅ System closed. Data saved.")