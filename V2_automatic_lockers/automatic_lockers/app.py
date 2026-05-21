# ============================================================
#  Smart Locker System - app_v3.py
# ============================================================

from flask import Flask, render_template, request, jsonify
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from ultralytics import YOLO
from scipy.spatial.distance import cosine
import cv2, torch, numpy as np
import sqlite3, json, base64, io, os, datetime, threading, time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sys

# --- TÍCH HỢP ANTI-SPOOFING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
from anti_spoof_service import check_liveness

# --- CẤU HÌNH EMAIL ---
SENDER_EMAIL = "quach.ya90@gmail.com"
SENDER_PASSWORD = "gjxoiyqeufvwppdw"


def send_smart_locker_email(receiver, subject, body):
    if not receiver or receiver.strip() == '' or receiver == '-':
        return

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = receiver
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'html'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, receiver, msg.as_string())
        server.quit()
        print(f"[Email] Da gui thu thanh cong toi {receiver} - {subject}")
    except Exception as e:
        print(f"[Email] Loi khi gui email toi {receiver}: {e}")


# ── Serial ──────────────────────────────────────────────────
try:
    import serial
    ser = serial.Serial('COM6', 115200, timeout=1)
    print("Serial OK")
except Exception as e:
    ser = None
    print(f"Serial offline: {e}")


# ── Config ───────────────────────────────────────────────────
DATA_DIR          = os.path.join(BASE_DIR, 'data')
DB_FILE           = os.path.join(DATA_DIR, 'face.db')
MODEL_PATH        = os.path.join(BASE_DIR, 'yolov12n-face.pt')
ADMIN_PASSWORD    = 'admin123'
TOTAL_LOCKERS     = 16
DOOR_AUTO_CLOSE_S = 3
LIVENESS_THRESHOLD = 0.85   # se tinh chinh sau khi co ROC dung

app    = Flask(__name__)
device = torch.device('cpu')

print("Load AI models...")
yolo_model = YOLO(MODEL_PATH)
resnet     = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("Models ready!")

_db_cache      = {}
_db_cache_lock = threading.Lock()
_last_db_load  = 0
_CACHE_TTL     = 3


# ============================================================
#  DATABASE
# ============================================================
def get_conn():
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            username   TEXT PRIMARY KEY,
            embedding  TEXT NOT NULL,
            gmail      TEXT,
            created_at TEXT
        )
    """)
    try:
        c.execute("ALTER TABLE faces ADD COLUMN gmail TEXT")
    except Exception:
        pass

    c.execute("""
        CREATE TABLE IF NOT EXISTS lockers (
            locker_id     TEXT PRIMARY KEY,
            username      TEXT,
            gmail         TEXT,
            checkin_time  TEXT,
            checkout_time TEXT,
            status        TEXT DEFAULT 'available',
            door_status   TEXT DEFAULT 'closed',
            has_items     TEXT DEFAULT 'no'
        )
    """)
    try:
        c.execute("ALTER TABLE lockers ADD COLUMN gmail TEXT")
    except Exception:
        pass

    c.execute("""
        CREATE TABLE IF NOT EXISTS access_logs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT,
            action     TEXT,
            timestamp  TEXT,
            confidence REAL
        )
    """)

    try:
        c.execute("ALTER TABLE access_logs ADD COLUMN gmail TEXT")
    except Exception:
        pass

    try:
        c.execute("ALTER TABLE access_logs ADD COLUMN liveness_score REAL")
    except Exception:
        pass

    for i in range(1, TOTAL_LOCKERS + 1):
        c.execute("""
            INSERT OR IGNORE INTO lockers (locker_id, status, door_status, has_items)
            VALUES (?, 'available', 'closed', 'no')
        """, (str(i),))

    conn.commit()
    conn.close()

init_db()

def log_access(username, action, confidence=None, gmail=None, liveness_score=None):
    conn = get_conn()
    conn.execute(
        "INSERT INTO access_logs (username, action, timestamp, confidence, gmail, liveness_score) VALUES (?,?,?,?,?,?)",
        (username, action, datetime.datetime.now().isoformat(), confidence, gmail, liveness_score)
    )
    conn.commit()
    conn.close()


# ============================================================
#  EMBEDDING / RECOGNITION
# ============================================================
def _refresh_cache():
    global _last_db_load
    _last_db_load = 0

def load_embeddings():
    global _db_cache, _last_db_load
    now = time.time()
    with _db_cache_lock:
        if now - _last_db_load < _CACHE_TTL and _db_cache:
            return _db_cache
        conn  = get_conn()
        rows  = conn.execute("SELECT username, embedding FROM faces").fetchall()
        conn.close()
        _db_cache     = {u: np.array(json.loads(e)) for u, e in rows}
        _last_db_load = now
        return _db_cache

def get_embedding(face_img):
    if isinstance(face_img, Image.Image):
        face_img = cv2.cvtColor(np.array(face_img.convert('RGB')), cv2.COLOR_RGB2BGR)
    face_img = cv2.resize(face_img, (160, 160))
    t = torch.tensor(face_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        return resnet(t.to(device)).cpu().numpy()[0]

def save_embedding(username, embedding, gmail=None):
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO faces (username, embedding, gmail, created_at) VALUES (?,?,?,?)",
        (username, json.dumps(embedding.tolist()), gmail, datetime.datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    _refresh_cache()

def recognize_face(embedding, threshold=0.55):
    db = load_embeddings()
    min_dist, best = float('inf'), None
    for user, db_emb in db.items():
        dist = cosine(embedding, db_emb)
        if dist < min_dist:
            min_dist = dist
            best     = user if dist < threshold else None
    confidence = round(1 - min_dist, 4) if best else None
    return best, confidence

def check_overdue_lockers():
    """Luong chay ngam de kiem tra tu qua ngay luc 08:00 sang moi ngay"""
    while True:
        now = datetime.datetime.now()
        if now.hour == 8 and now.minute == 0:
            print("[System] Bat dau quet cac tu de qua dem...")
            conn = get_conn()
            c = conn.cursor()

            c.execute('''
                SELECT locker_id, username, gmail, checkin_time
                FROM lockers
                WHERE status='occupied' AND date(checkin_time) < date('now', 'localtime')
            ''')
            overdue_lockers = c.fetchall()

            for locker in overdue_lockers:
                if locker['gmail']:
                    subject = f"Thong bao: Do o Tu {locker['locker_id']} da qua ngay"
                    body = f"""
                    <h3>Chao {locker['username']},</h3>
                    <p>He thong ghi nhan ban da gui do tai <b>Tu {locker['locker_id']}</b> tu luc {locker['checkin_time']}.</p>
                    <p>Hien tai do cua ban da duoc luu tru qua ngay moi. Theo quy dinh, he thong se <b>tinh them phi luu tru qua dem</b>.</p>
                    <p>Vui long den nhan do som nhat co the.</p>
                    <br><p><i>He thong Smart Locker</i></p>
                    """
                    send_smart_locker_email(locker['gmail'], subject, body)

            conn.close()
            time.sleep(61)

        time.sleep(30)

threading.Thread(target=check_overdue_lockers, daemon=True).start()


# ============================================================
#  SERIAL RELAY (Server -> ESP32)
# ============================================================
def send_relay(locker_id, state):
    cmd = json.dumps({"locker_id": str(locker_id), "state": state}) + '\n'
    if ser:
        try:
            ser.write(cmd.encode('utf-8'))
        except Exception as e:
            print(f"Relay error: {e}")
    print(f"[Relay] Tu {locker_id}: {'OPEN' if state else 'CLOSE'}")

def open_then_auto_close(locker_id):
    send_relay(locker_id, 1)
    conn = get_conn()
    conn.execute("UPDATE lockers SET door_status='open' WHERE locker_id=?", (str(locker_id),))
    conn.commit()
    conn.close()

    def _close():
        time.sleep(DOOR_AUTO_CLOSE_S)
        send_relay(locker_id, 0)
        try:
            c2 = get_conn()
            c2.execute("UPDATE lockers SET door_status='closed' WHERE locker_id=?", (str(locker_id),))
            c2.commit()
            c2.close()
            print(f"[Auto-close] Tu {locker_id} da dong sau {DOOR_AUTO_CLOSE_S}s")
        except Exception as e:
            print(f"[Auto-close] Loi: {e}")

    threading.Thread(target=_close, daemon=True).start()

def process_esp32_data(raw: str):
    raw = raw.strip()
    if not raw.startswith('{'):
        return
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return

    conn = get_conn()
    try:
        if any(k.startswith('check_') or k.startswith('switch_') for k in data):
            updates = {}

            for key, val in data.items():
                if key.startswith('check_'):
                    lid = key[len('check_'):]
                    updates.setdefault(lid, {})['has_items'] = 'yes' if val else 'no'
                elif key.startswith('switch_'):
                    lid = key[len('switch_'):]
                    updates.setdefault(lid, {})['door_status'] = 'closed' if val else 'open'

            for lid, vals in updates.items():
                has_items   = vals.get('has_items')
                door_status = vals.get('door_status')

                if has_items is not None and door_status is not None:
                    conn.execute(
                        "UPDATE lockers SET has_items=?, door_status=? WHERE locker_id=?",
                        (has_items, door_status, lid)
                    )
                elif has_items is not None:
                    conn.execute(
                        "UPDATE lockers SET has_items=? WHERE locker_id=?",
                        (has_items, lid)
                    )
                elif door_status is not None:
                    conn.execute(
                        "UPDATE lockers SET door_status=? WHERE locker_id=?",
                        (door_status, lid)
                    )

                if has_items == 'yes':
                    locker_info = conn.execute("SELECT status, door_status FROM lockers WHERE locker_id=?", (lid,)).fetchone()

                    if locker_info and locker_info['status'] == 'available' and locker_info['door_status'] == 'closed':
                        last_log = conn.execute('''
                            SELECT username, gmail, timestamp FROM access_logs
                            WHERE username IS NOT NULL AND action='checkout'
                            ORDER BY timestamp DESC LIMIT 1
                        ''').fetchone()

                        if last_log and last_log['gmail']:
                            log_time = datetime.datetime.strptime(last_log['timestamp'], "%Y-%m-%dT%H:%M:%S.%f")
                            seconds_since_checkout = (datetime.datetime.now() - log_time).total_seconds()

                            if seconds_since_checkout < 60:
                                if not hasattr(app, 'sent_forgot_warnings'):
                                    app.sent_forgot_warnings = set()

                                cache_key = f"{lid}_{last_log['timestamp']}"
                                if cache_key not in app.sent_forgot_warnings:
                                    subject = f"Canh bao: Ban da de quen do o Tu {lid}"
                                    body = f"""
                                    <h3>Chao {last_log['username']},</h3>
                                    <p>He thong ghi nhan ban vua lay do tai <b>Tu {lid}</b>.</p>
                                    <p>Tuy nhien, cam bien phat hien <b>van con do vat ben trong tu</b> sau khi cua da dong lai.</p>
                                    <br><p><i>He thong Smart Locker</i></p>
                                    """
                                    threading.Thread(target=send_smart_locker_email, args=(last_log['gmail'], subject, body)).start()
                                    app.sent_forgot_warnings.add(cache_key)

            conn.commit()
            return

        if data.get('type') == 'bulk':
            for item in data.get('data', []):
                lid  = str(item.get('id', ''))
                door = 'open' if item.get('door') == 'open' else 'closed'
                hi   = 'yes' if item.get('sensor', 0) else 'no'
                conn.execute(
                    "UPDATE lockers SET door_status=?, has_items=? WHERE locker_id=?",
                    (door, hi, lid)
                )
            conn.commit()
            return

        if 'locker_id' in data:
            lid  = str(data['locker_id'])
            door = 'open' if data.get('door') == 'open' else 'closed'
            hi   = 'yes' if data.get('sensor', 0) else 'no'
            conn.execute(
                "UPDATE lockers SET door_status=?, has_items=? WHERE locker_id=?",
                (door, hi, lid)
            )
            conn.commit()

    except Exception as e:
        print(f"[ESP32 DB] Loi: {e}")
    finally:
        conn.close()

def _serial_reader():
    """Background thread: lien tuc doc JSON tu ESP32 qua Serial"""
    print("Serial reader thread started")
    while True:
        try:
            if ser and ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    process_esp32_data(line)
        except Exception as e:
            print(f"[Serial Reader] {e}")
        time.sleep(0.02)

_serial_thread = threading.Thread(target=_serial_reader, daemon=True)
_serial_thread.start()


# ============================================================
#  HELPER
# ============================================================
def decode_image(img_base64):
    """Decode base64 sang numpy array RGB (PIL default)."""
    _, encoded = img_base64.split(',', 1)
    return np.array(Image.open(io.BytesIO(base64.b64decode(encoded))).convert('RGB'))

def detect_faces(img_np_rgb):
    """YOLO detect, tra ve list [x1, y1, x2, y2]."""
    results = yolo_model(img_np_rgb, device='cpu', verbose=False)
    return [list(map(int, box.xyxy[0])) for res in results for box in res.boxes]

def get_face_embedding_from_image(img_np_rgb, boxes):
    """Crop sat mat de tao embedding cho FaceNet."""
    x1, y1, x2, y2 = boxes[0]
    face_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)[y1:y2, x1:x2]
    return get_embedding(face_bgr)


def run_liveness(img_np_rgb, boxes, threshold=LIVENESS_THRESHOLD):
    """Goi anti-spoof DUNG CACH:
       - Truyen FULL FRAME goc da convert BGR (cv2 default)
       - Truyen bbox [x, y, w, h] tu YOLO (KHONG crop truoc)
       MiniFASNet se tu mo rong bbox theo scale 2.7 / 4.0 trong CropImage.
    """
    if not boxes:
        return False, 0.0
    x1, y1, x2, y2 = boxes[0]
    frame_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
    bbox_xywh = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
    return check_liveness(frame_bgr, bbox_xywh, threshold=threshold)


# ============================================================
#  ROUTES
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/esp32/update', methods=['POST'])
def esp32_update():
    try:
        process_esp32_data(json.dumps(request.json))
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 400

@app.route('/list_lockers')
def list_lockers():
    conn = get_conn()
    rows = conn.execute(
        """SELECT locker_id, username, gmail, status, door_status, has_items,
                  checkin_time, checkout_time
           FROM lockers ORDER BY CAST(locker_id AS INTEGER)"""
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/analyze_hint', methods=['POST'])
def analyze_hint():
    try:
        img_np = decode_image(request.json.get('img_base64'))
        boxes  = detect_faces(img_np)
        h, w   = img_np.shape[:2]

        if len(boxes) == 0:
            return jsonify({"hint": "Khong tim thay khuon mat", "valid": False,
                            "name": None, "box": None, "img_w": w, "img_h": h})
        if len(boxes) > 1:
            return jsonify({"hint": "Chi 1 nguoi dung truoc camera", "valid": False,
                            "name": None, "box": None, "img_w": w, "img_h": h})

        x1, y1, x2, y2 = boxes[0]
        area = (x2 - x1) * (y2 - y1)

        # --- 1. ANTI-SPOOF -- truyen full frame + bbox ---
        is_real, live_score = run_liveness(img_np, boxes)

        # --- 2. KET QUA ---
        if not is_real:
            hint = f"CANH BAO GIA MAO! ({live_score:.2f})"
            valid = False
            name, conf = None, None
        else:
            emb        = get_face_embedding_from_image(img_np, boxes)
            name, conf = recognize_face(emb)

            if area < 15000:
                hint, valid = "Vui long tien lai gan hon", False
            elif area > 45000:
                hint, valid = "Vui long lui ra xa mot chut", False
            else:
                valid = True
                hint  = f"Xin chao {name}!" if name else "Vui long giu yen!"

        return jsonify({
            "hint"  : hint,
            "valid" : valid,
            "name"  : name,
            "conf"  : conf,
            "box"   : [x1, y1, x2, y2],
            "img_w" : w,
            "img_h" : h
        })
    except Exception as e:
        return jsonify({"hint": "...", "valid": False,
                        "name": None, "box": None, "img_w": 640, "img_h": 480})

@app.route('/check_face_in_db', methods=['POST'])
def check_face_in_db():
    try:
        img_np     = decode_image(request.json.get('img_base64'))
        boxes      = detect_faces(img_np)
        if len(boxes) != 1:
            return jsonify({"found": False, "error": "Can dung 1 khuon mat"})

        is_real, live_score = run_liveness(img_np, boxes)
        if not is_real:
            return jsonify({"found": False, "error": f"Phat hien gia mao khuon mat (Ti le: {live_score:.2f})"})

        emb        = get_face_embedding_from_image(img_np, boxes)
        name, conf = recognize_face(emb)
        if name:
            conn  = get_conn()
            row   = conn.execute("SELECT gmail FROM faces WHERE username=?", (name,)).fetchone()
            conn.close()
            return jsonify({"found": True, "username": name,
                            "gmail": row['gmail'] if row and row['gmail'] else '',
                            "confidence": conf})
        return jsonify({"found": False})
    except Exception as e:
        return jsonify({"found": False, "error": str(e)})

@app.route('/register_locker', methods=['POST'])
def register_locker():
    try:
        data      = request.json
        username  = data.get('username', '').strip()
        gmail     = data.get('gmail', '').strip()
        locker_id = data.get('locker_id', '').strip()
        img_b64   = data.get('img_base64', '')

        if not username or not locker_id or not img_b64:
            return jsonify({'error': 'Thieu thong tin bat buoc'}), 400

        conn = get_conn()
        row  = conn.execute("SELECT status FROM lockers WHERE locker_id=?", (locker_id,)).fetchone()
        conn.close()
        if not row:
            return jsonify({'error': f'Tu {locker_id} khong ton tai'}), 404
        if row['status'] == 'occupied':
            return jsonify({'error': f'Tu {locker_id} da co nguoi su dung'}), 400

        img_np = decode_image(img_b64)
        boxes  = detect_faces(img_np)
        if len(boxes) == 0:
            return jsonify({'error': 'Khong tim thay khuon mat'}), 400
        if len(boxes) > 1:
            return jsonify({'error': 'Chi 1 khuon mat trong anh'}), 400

        is_real, live_score = run_liveness(img_np, boxes)
        if not is_real:
            log_access(username, 'register_spoof_rejected', None, gmail, live_score)
            return jsonify({'error': f'Dang ky bi tu choi: Phat hien anh gia mao! Liveness: {live_score:.2f}'}), 403

        emb = get_face_embedding_from_image(img_np, boxes)
        save_embedding(username, emb, gmail)

        now  = datetime.datetime.now().isoformat()
        conn = get_conn()
        conn.execute(
            """UPDATE lockers
               SET username=?, gmail=?, checkin_time=?, checkout_time=NULL, status='occupied'
               WHERE locker_id=?""",
            (username, gmail, now, locker_id)
        )
        conn.commit()
        conn.close()

        log_access(username, 'checkin', 1.0, gmail, live_score)
        open_then_auto_close(locker_id)

        return jsonify({'success': True,
                        'message': f'Tu {locker_id} da mo, tu dong sau {DOOR_AUTO_CLOSE_S}s.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/face_checkout', methods=['POST'])
def face_checkout():
    try:
        data      = request.json
        locker_id = data.get('locker_id', '').strip()
        img_b64   = data.get('img_base64', '')

        if not locker_id or not img_b64:
            return jsonify({'error': 'Thieu thong tin'}), 400

        conn  = get_conn()
        row   = conn.execute("SELECT username, gmail FROM lockers WHERE locker_id=?", (locker_id,)).fetchone()
        conn.close()
        if not row or not row['username']:
            return jsonify({'error': f'Tu {locker_id} khong co nguoi dung'}), 400

        owner  = row['username']
        img_np = decode_image(img_b64)
        boxes  = detect_faces(img_np)
        if len(boxes) == 0:
            return jsonify({'error': 'Khong tim thay khuon mat'}), 400
        if len(boxes) > 1:
            return jsonify({'error': 'Chi 1 khuon mat trong anh'}), 400

        is_real, live_score = run_liveness(img_np, boxes)
        if not is_real:
            log_access(owner, 'checkout_spoof_rejected', None, row['gmail'], live_score)
            return jsonify({'error': f'Mo tu bi tu choi: Phat hien anh gia mao! Liveness: {live_score:.2f}'}), 403

        emb        = get_face_embedding_from_image(img_np, boxes)
        best, conf = recognize_face(emb)

        if not best:
            return jsonify({'error': 'Khuon mat khong co trong he thong'}), 400
        if best != owner:
            return jsonify({'error': f'Khong phai chu tu (chu: {owner})'}), 400

        now  = datetime.datetime.now().isoformat()
        conn = get_conn()
        conn.execute(
            """UPDATE lockers
               SET checkout_time=?, status='available', username=NULL, gmail=NULL
               WHERE locker_id=?""",
            (now, locker_id)
        )
        conn.commit()
        conn.close()

        log_access(owner, 'checkout', conf, row['gmail'], live_score)
        open_then_auto_close(locker_id)

        return jsonify({'success': True,
                        'message': f'Tu {locker_id} da mo, tu dong sau {DOOR_AUTO_CLOSE_S}s.',
                        'confidence': conf})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/login', methods=['POST'])
def admin_login():
    if request.json.get('password') == ADMIN_PASSWORD:
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Sai mat khau!'}), 401

@app.route('/admin/stats')
def admin_stats():
    conn     = get_conn()
    occupied = conn.execute("SELECT COUNT(*) as n FROM lockers WHERE status='occupied'").fetchone()['n']
    logs     = conn.execute("SELECT * FROM access_logs ORDER BY timestamp DESC LIMIT 100").fetchall()
    users    = conn.execute(
        "SELECT locker_id, username, gmail, checkin_time FROM lockers WHERE status='occupied'"
    ).fetchall()
    conn.close()
    return jsonify({
        'occupied': occupied, 'available': TOTAL_LOCKERS - occupied,
        'total': TOTAL_LOCKERS,
        'logs': [dict(r) for r in logs],
        'active_users': [dict(r) for r in users]
    })

@app.route('/admin/list_customers')
def admin_list_customers():
    conn = get_conn()
    rows = conn.execute(
        """SELECT f.username, f.gmail, f.created_at,
                  l.locker_id, l.status, l.checkin_time
           FROM faces f
           LEFT JOIN lockers l ON l.username = f.username
           ORDER BY f.created_at DESC"""
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/admin/control_locker', methods=['POST'])
def admin_control():
    data = request.json
    if data.get('password') != ADMIN_PASSWORD:
        return jsonify({'error': 'Xac thuc that bai'}), 401
    locker_id = data.get('locker_id')
    action    = int(data.get('action', 1))
    if not locker_id:
        return jsonify({'error': 'Thieu locker_id'}), 400
    if action == 1:
        open_then_auto_close(locker_id)
    else:
        send_relay(locker_id, 0)
        conn = get_conn()
        conn.execute("UPDATE lockers SET door_status='closed' WHERE locker_id=?", (str(locker_id),))
        conn.commit()
        conn.close()
    return jsonify({'success': True,
                    'message': f"Da {'mo' if action else 'dong'} tu {locker_id}"})

@app.route('/admin/delete_user', methods=['POST'])
def delete_user():
    data     = request.json
    if data.get('password') != ADMIN_PASSWORD:
        return jsonify({'error': 'Xac thuc that bai'}), 401
    username = data.get('username', '').strip()
    if not username:
        return jsonify({'error': 'Thieu username'}), 400
    conn = get_conn()
    conn.execute("DELETE FROM faces WHERE username=?", (username,))
    conn.execute(
        "UPDATE lockers SET username=NULL, gmail=NULL, status='available' WHERE username=?",
        (username,)
    )
    conn.commit()
    conn.close()
    _refresh_cache()
    return jsonify({'success': True, 'message': f'Da xoa {username}'})

@app.route('/admin/delete_logs', methods=['POST'])
def delete_logs():
    data = request.json
    pw = data.get('password')
    if pw != ADMIN_PASSWORD:
        return jsonify({'error': 'Sai mat khau'}), 403

    conn = get_conn()
    try:
        if data.get('delete_all'):
            conn.execute("DELETE FROM access_logs")
            conn.commit()
            return jsonify({'message': 'Da xoa toan bo lich su'})

        date = data.get('date')
        if not date:
            return jsonify({'error': 'Thieu ngay can xoa'}), 400

        conn.execute("DELETE FROM access_logs WHERE date(timestamp) = ?", (date,))
        conn.commit()
        return jsonify({'message': f'Da xoa lich su ngay {date}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

if __name__ == '__main__':
    app.run(debug=True, threaded=True, use_reloader=False)
