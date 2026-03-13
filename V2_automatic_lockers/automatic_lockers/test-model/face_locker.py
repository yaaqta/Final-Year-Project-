import cv2
import numpy as np
import json
import time
from scipy.spatial.distance import cosine
from facenet_pytorch import MTCNN, InceptionResnetV1

print("Loading FaceNet...")
mtcnn = MTCNN(image_size=160, margin=20, device='cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to('cpu')
print("Models OK")

# Data
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
    except: pass

def save():
    with open(data_file,'w') as f:
        json.dump({"faces":faces,"lockers":lockers},f)

load()

def get_face_emb(frame):
    face = mtcnn(frame)
    if face is None: return None
    emb = resnet(face.unsqueeze(0)).detach().cpu().numpy().flatten()
    return emb/np.linalg.norm(emb)

def checkin(name, lid, frame):
    if lockers[str(lid)]["status"] != "free":
        return False, f"L{lid} busy"
    emb = get_face_emb(frame)
    if emb is None:
        return False, "No face"
    faces.append({"name":name,"lid":lid,"emb":emb.tolist()})
    lockers[str(lid)] = {"user":name,"status":"busy"}
    save()
    return True, f"L{lid} OK"

def check_out(frame):
    emb = get_face_emb(frame)
    if emb is None: return None, "No face"
    best = None
    mind = 1.0
    for f in faces:
        d = cosine(emb, np.array(f["emb"]))
        if d < mind and d < 0.4:
            mind = d
            best = f
    if best and (1-mind)>0.8:
        lid = best["lid"]
        lockers[str(lid)]["user"] = None
        lockers[str(lid)]["status"] = "free"
        save()
        return best, f"L{lid} {int((1-mind)*100)}%"
    return None, "FAIL"

# Main
cap = cv2.VideoCapture(0)
cap.set(3,640);cap.set(4,480)
mode = "scan"
lid = 1
name = "User"

print("C=Checkin O=Checkout 1-8=Locker Q=Quit")

while True:
    ok, frame = cap.read()
    if not ok: continue
    frame = cv2.flip(frame,1)
    
    msg = ""
    if mode=="checkin":
        suc, txt = checkin(name, lid, frame)
        msg = txt
        mode = "scan"
    elif mode=="checkout":
        mat, txt = check_out(frame)
        msg = txt
        mode = "scan"
    
    # 8 lockers
    h,w = frame.shape[:2]
    for i in range(1,9):
        x = int(30+(w-60)*(i-1)/8)
        d = lockers[str(i)]
        col = (0,255,0) if d["status"]=="free" else (0,0,255)
        cv2.rectangle(frame,(x-12,h-80),(x+12,h-45),col,-1)
        cv2.putText(frame,f"L{i}",(x-8,h-60),0.4,1,(255,255,255),1)
    
    # UI
    cv2.rectangle(frame,(10,10),(220,70),(30,30,40),-1)
    cv2.putText(frame,f"Mode:{mode}",(20,30),0.5,1,(255,255,255),1)
    cv2.putText(frame,msg[:20],(20,50),0.5,1,(0,255,0),1)
    cv2.putText(frame,f"L{lid} {name}",(20,70),0.5,1,(255,255,0),1)
    
    cv2.imshow("Face Locker", frame)
    
    k = cv2.waitKey(1)&0xFF
    if k==ord('q'): break
    elif k==ord('c'): mode="checkin"
    elif k==ord('o'): mode="checkout"
    elif ord('1')<=k<=ord('8'): lid = k-ord('0')
    elif k==ord('n'): name = f"U{int(time.time()%100)}"

cap.release()
cv2.destroyAllWindows()
