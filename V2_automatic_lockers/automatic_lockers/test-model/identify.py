import cv2
import numpy as np
import json
import time
from scipy.spatial.distance import cosine
from facenet_pytorch import MTCNN, InceptionResnetV1

# Models
device = 'cpu'
mtcnn = MTCNN(image_size=160, margin=20, min_face_size=40, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("✅ Models OK")

# JSON file cùng thư mục
DB_FILE = "face_lockers.json"

# 8 Lockers
lockers = {str(i): {"assigned": None, "status": "available"} for i in range(1, 9)}
faces_db = []

def load_db():
    global faces_db, lockers
    try:
        with open(DB_FILE, "r") as f:
            data = json.load(f)
            faces_db = data.get("faces", [])
            lockers.update(data.get("lockers", {}))
    except:
        pass  # File mới

def save_db():
    with open(DB_FILE, "w") as f:
        json.dump({"faces": faces_db, "lockers": lockers}, f)

load_db()
print("✅ DB loaded")

def get_embedding(frame):
    face = mtcnn(frame)
    if face is None: return None
    emb = resnet(face.unsqueeze(0)).detach().cpu().numpy().flatten()
    return emb / np.linalg.norm(emb)

def checkin(user, locker, frame):
    global lockers
    if lockers[str(locker)]["status"] != "available":
        return False, f"L{locker} busy"
    
    emb = get_embedding(frame)
    if emb is None:
        return False, "No face"
    
    faces_db.append({"user": user, "locker": locker, "emb": emb.tolist()})
    lockers[str(locker)] = {"assigned": user, "status": "occupied"}
    save_db()
    return True, f"✅ L{locker} OK"

def checkout(frame):
    emb = get_embedding(frame)
    if emb is None:
        return None, "No face"
    
    best = None
    min_dist = 999
    for f in faces_db:
        dist = cosine(emb, np.array(f["emb"]))
        if dist < min_dist and dist < 0.4:  # 80%+
            min_dist = dist
            best = f
    
    if best:
        lid = best["locker"]
        conf = (1 - min_dist) * 100
        if conf > 80:
            lockers[str(lid)]["assigned"] = None
            lockers[str(lid)]["status"] = "available"
            save_db()
            return best, f"🔓 L{lid} {conf:.0f}%"
    return None, "❌ FAIL"

# MAIN
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640); cap.set(4, 480)

mode = "scan"
locker = 1
user = "User01"

print("C=CheckIn O=CheckOut 1-8=Locker U=User Q=Quit")

while True:
    ret, frame = cap.read()
    if not ret: continue
    frame = cv2.flip(frame, 1)
    
    msg = ""
    if mode == "checkin":
        ok, txt = checkin(user, locker, frame)
        msg = txt
        mode = "scan"
    elif mode == "checkout":
        match, txt = checkout(frame)
        msg = txt
        mode = "scan"
    
    # Draw 8 lockers
    h, w = frame.shape[:2]
    y = h - 90
    for i in range(1, 9):
        x = int(40 + (w-80)*(i-1)/8)
        data = lockers[str(i)]
        col = (0,255,0) if data["status"]=="available" else (0,0,255)
        cv2.rectangle(frame, (x-15,y),(x+15,y+30), col, -1)
        cv2.putText(frame, f"L{i}", (x-10,y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        if data["assigned"]:
            cv2.putText(frame, data["assigned"][:2], (x-12,y+45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
    
    # UI
    cv2.rectangle(frame, (10,10),(200,70),(50,50,60),-1)
    cv2.putText(frame, f"Mode: {mode}", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, msg[:25], (20,55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.putText(frame, f"Tủ:{locker} {user}", (220,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    
    cv2.imshow("Face Locker", frame)
    
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    elif k == ord('c'): mode = "checkin"
    elif k == ord('o'): mode = "checkout"
    elif b'1' <= bytes([k]) <= b'8': locker = k - ord('0')
    elif k == ord('u'): user = f"U{int(time.time()%100):02d}"

cap.release()
cv2.destroyAllWindows()
print("Done!")
