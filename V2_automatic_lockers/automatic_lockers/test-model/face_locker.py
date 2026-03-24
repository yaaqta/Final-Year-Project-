import cv2
import numpy as np
import json
import time
import torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cosine

print("Loading models...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# YOLO face model
yolo = YOLO("model/yolov12n-face.pt")
yolo.to(device)

# FaceNet
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

print("Models loaded")

# ===============================
# Locker data
# ===============================

data_file = "lockers.json"

lockers = {str(i):{"user":None,"status":"free"} for i in range(1,9)}
faces = []

def load():
    global faces, lockers
    try:
        with open(data_file,'r') as f:
            d = json.load(f)
            faces = d["faces"]
            lockers.update(d["lockers"])
    except:
        pass

def save():
    with open(data_file,'w') as f:
        json.dump({"faces":faces,"lockers":lockers},f)

load()

# ===============================
# Face Embedding
# ===============================

def get_embedding(face):

    face = cv2.resize(face,(160,160))

    face = torch.tensor(face).permute(2,0,1).float()/255
    face = face.unsqueeze(0).to(device)

    emb = resnet(face).detach().cpu().numpy().flatten()

    return emb/np.linalg.norm(emb)

# ===============================
# Face Detection
# ===============================

def detect_faces(frame):

    results = yolo(frame, conf=0.5, verbose=False)

    boxes = []
    confs = []

    for r in results:

        if r.boxes is None:
            continue

        b = r.boxes.xyxy.cpu().numpy()
        c = r.boxes.conf.cpu().numpy()

        boxes.extend(b)
        confs.extend(c)

    return boxes, confs

# ===============================
# Recognition
# ===============================

def recognize(face_emb):

    best = None
    mind = 1.0

    for f in faces:

        d = cosine(face_emb, np.array(f["emb"]))

        if d < mind and d < 0.4:
            mind = d
            best = f

    if best and (1-mind) > 0.8:
        return best, (1-mind)

    return None, 0

# ===============================
# Checkin
# ===============================

def checkin(name, lid, face_emb):

    if lockers[str(lid)]["status"] != "free":
        return False, f"L{lid} busy"

    faces.append({
        "name":name,
        "lid":lid,
        "emb":face_emb.tolist()
    })

    lockers[str(lid)] = {
        "user":name,
        "status":"busy"
    }

    save()

    return True, f"L{lid} OK"

# ===============================
# Checkout
# ===============================

def checkout(face_emb):

    best, score = recognize(face_emb)

    if best:

        lid = best["lid"]

        lockers[str(lid)]["user"] = None
        lockers[str(lid)]["status"] = "free"

        save()

        return best, f"L{lid} {int(score*100)}%"

    return None,"FAIL"

# ===============================
# Camera
# ===============================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3,640)
cap.set(4,480)

mode = "scan"
lid = 1
name = "User"

print("C=Checkin O=Checkout 1-8=Locker N=NewName Q=Quit")

prev_time = time.time()

while True:

    ok, frame = cap.read()

    if not ok:
        continue

    frame = cv2.flip(frame,1)

    boxes, confs = detect_faces(frame)

    face_emb = None

    for box, conf in zip(boxes,confs):

        x1,y1,x2,y2 = map(int,box)

        face = frame[y1:y2,x1:x2]

        if face.size == 0:
            continue

        face_emb = get_embedding(face)

        # draw box
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.putText(frame,
                    f"{conf:.2f}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2)

    msg = ""

    if face_emb is not None:

        if mode=="checkin":

            suc,msg = checkin(name,lid,face_emb)
            mode="scan"

        elif mode=="checkout":

            mat,msg = checkout(face_emb)
            msg = mat["name"] if mat else msg
            mode="scan"

    # ===============================
    # Locker UI
    # ===============================

    h,w = frame.shape[:2]

    for i in range(1,9):

        x = int(30+(w-60)*(i-1)/8)

        d = lockers[str(i)]

        col = (0,255,0) if d["status"]=="free" else (0,0,255)

        cv2.rectangle(frame,(x-12,h-80),(x+12,h-45),col,-1)

        cv2.putText(frame,
                    f"L{i}",
                    (x-10,h-60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255,255),
                    1)

    # ===============================
    # FPS
    # ===============================

    cur = time.time()
    fps = 1/(cur-prev_time)
    prev_time = cur

    cv2.putText(frame,
                f"FPS:{int(fps)}",
                (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,0,0),
                2)

    cv2.putText(frame,
                f"Mode:{mode}",
                (10,60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                2)

    cv2.putText(frame,
                f"L{lid} {name}",
                (10,90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,0),
                2)

    cv2.imshow("Face Locker",frame)

    k = cv2.waitKey(1)&0xFF

    if k==ord('q'):
        break

    elif k==ord('c'):
        mode="checkin"

    elif k==ord('o'):
        mode="checkout"

    elif ord('1')<=k<=ord('8'):
        lid = k-ord('0')

    elif k==ord('n'):
        name = f"U{int(time.time()%100)}"

cap.release()
cv2.destroyAllWindows()