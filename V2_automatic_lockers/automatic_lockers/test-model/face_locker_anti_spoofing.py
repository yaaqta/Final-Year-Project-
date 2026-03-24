import cv2
import numpy as np
import torch
import json
import time
import onnxruntime as ort
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cosine

print("Loading models...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ---------------------------
# YOLO FACE DETECTION
# ---------------------------

yolo = YOLO("model/yolov12n-face.pt")
yolo.to(device)

# ---------------------------
# FACENET
# ---------------------------

resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# ---------------------------
# ANTI SPOOF MODEL
# ---------------------------

spoof_session = ort.InferenceSession("model/miniFASnetV2.onnx")
spoof_input = spoof_session.get_inputs()[0].name

print("Models loaded")

# ---------------------------
# LOCKER DATABASE
# ---------------------------

data_file = "lockers.json"

lockers = {str(i): {"user": None, "status": "free"} for i in range(1,9)}
faces = []

print(spoof_session.get_inputs()[0].shape)


def load():
    global faces, lockers
    try:
        with open(data_file,"r") as f:
            d = json.load(f)
            faces = d["faces"]
            lockers.update(d["lockers"])
    except:
        pass

def save():
    with open(data_file,"w") as f:
        json.dump({"faces":faces,"lockers":lockers},f)

load()

# ---------------------------
# ANTI SPOOF FUNCTION
# ---------------------------

def anti_spoof(face):

    face = cv2.resize(face,(80,80))

    face = face.astype(np.float32)
    face = (face - 127.5) / 128.0

    face = np.transpose(face,(2,0,1))
    face = np.expand_dims(face,0)

    pred = spoof_session.run(None,{spoof_input:face})[0]

    exp = np.exp(pred)
    prob = exp / np.sum(exp, axis=1, keepdims=True)

    score = prob[0][2]   # REAL

    return score

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    face = cv2.resize(face,(80,80))

    face = face.astype(np.float32)

    face = (face - 127.5) / 128.0

    face = np.transpose(face,(2,0,1))

    face = np.expand_dims(face,0)

    pred = spoof_session.run(None,{spoof_input:face})[0]
    print("raw:", pred)

    # softmax
    exp = np.exp(pred)
    prob = exp / np.sum(exp, axis=1, keepdims=True)

    score = prob[0][2]   # REAL probability

    return score
# ---------------------------
# FACE EMBEDDING
# ---------------------------

def get_embedding(face):

    face = cv2.resize(face,(160,160))

    face = torch.tensor(face).permute(2,0,1).float()/255

    face = face.unsqueeze(0).to(device)

    emb = resnet(face).detach().cpu().numpy().flatten()

    return emb / np.linalg.norm(emb)

# ---------------------------
# RECOGNITION
# ---------------------------

def recognize(emb):

    best = None
    mind = 1.0

    for f in faces:

        d = cosine(emb,np.array(f["emb"]))

        if d < mind and d < 0.4:
            mind = d
            best = f

    if best and (1-mind) > 0.8:
        return best,(1-mind)

    return None,0

# ---------------------------
# CAMERA
# ---------------------------

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(3,640)
cap.set(4,480)

mode = "scan"
lid = 1
name = "User"

prev_time = time.time()

print("C=Checkin O=Checkout 1-8=Locker N=NewUser Q=Quit")

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.flip(frame,1)

    results = yolo(frame,conf=0.5,verbose=False)

    face_emb = None
    msg = ""

    for r in results:

        if r.boxes is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for box,conf in zip(boxes,confs):

            x1,y1,x2,y2 = map(int,box)

            h, w = frame.shape[:2]

            margin = 0.3

            bw = x2 - x1
            bh = y2 - y1

            x1 = max(0, int(x1 - bw*margin))
            y1 = max(0, int(y1 - bh*margin))
            x2 = min(w, int(x2 + bw*margin))
            y2 = min(h, int(y2 + bh*margin))

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            score = anti_spoof(face)

            real = score > 0.5

            color = (0,255,0) if real else (0,0,255)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

            cv2.putText(frame,
                        f"{conf:.2f} real:{score:.2f}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2)

            if not real:

                cv2.putText(frame,
                            "FAKE",
                            (x1,y2+20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0,0,255),
                            2)

                continue

            face_emb = get_embedding(face)

    # ---------------------------
    # LOCKER LOGIC
    # ---------------------------

    if face_emb is not None:

        if mode == "checkin":

            if lockers[str(lid)]["status"] == "free":

                faces.append({
                    "name":name,
                    "lid":lid,
                    "emb":face_emb.tolist()
                })

                lockers[str(lid)] = {"user":name,"status":"busy"}

                save()

                msg = f"L{lid} OK"

            else:

                msg = f"L{lid} busy"

            mode = "scan"

        elif mode == "checkout":

            best,score = recognize(face_emb)

            if best:

                lid2 = best["lid"]

                lockers[str(lid2)]["user"] = None
                lockers[str(lid2)]["status"] = "free"

                save()

                msg = f"L{lid2} {int(score*100)}%"

            else:

                msg = "FAIL"

            mode = "scan"

    # ---------------------------
    # LOCKER UI
    # ---------------------------

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

    # ---------------------------
    # FPS
    # ---------------------------

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
                msg,
                (10,90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2)

    cv2.imshow("Face Locker",frame)

    k = cv2.waitKey(1)&0xFF

    if k == ord("q"):
        break

    elif k == ord("c"):
        mode = "checkin"

    elif k == ord("o"):
        mode = "checkout"

    elif ord("1") <= k <= ord("8"):
        lid = k - ord("0")

    elif k == ord("n"):
        name = f"U{int(time.time()%100)}"

cap.release()
cv2.destroyAllWindows()