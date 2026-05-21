


# ============================================================
#  Smart Locker System - app.py
# ============================================================

from flask import Flask, render_template, request, jsonify
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from ultralytics import YOLO
from scipy.spatial.distance import cosine
import cv2, torch, numpy as np
import sqlite3, json, base64, io, os, datetime, threading, time

# ── Serial ──────────────────────────────────────────────────
try:
    import serial
    ser = serial.Serial('COM3', 115200, timeout=1)
    print("✅ Serial OK")
except Exception as e:
    ser = None
    print(f"⚠️  Serial offline: {e}")

# ── Config ───────────────────────────────────────────────────
DB_FILE           = 'automatic_lockers/data/face.db'
ADMIN_PASSWORD    = 'admin123'
TOTAL_LOCKERS     = 16
DOOR_AUTO_CLOSE_S = 3

app    = Flask(__name__)
device = torch.device('cpu')

print("⏳ Load AI models...")
yolo_model = YOLO('automatic_lockers/yolov12n-face.pt')
resnet     = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("✅ Models ready!")

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
    for col in ['gmail']:
        try:
            c.execute(f"ALTER TABLE faces ADD COLUMN {col} TEXT")
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

    c.execute("""
        CREATE TABLE IF NOT EXISTS access_logs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT,
            action     TEXT,
            timestamp  TEXT,
            confidence REAL
        )
    """)

    for i in range(1, TOTAL_LOCKERS + 1):
        c.execute("""
            INSERT OR IGNORE INTO lockers (locker_id, status, door_status, has_items)
            VALUES (?, 'available', 'closed', 'no')
        """, (str(i),))

    conn.commit()
    conn.close()

init_db()


def log_access(username, action, confidence=None):
    conn = get_conn()
    conn.execute(
        "INSERT INTO access_logs (username, action, timestamp, confidence) VALUES (?,?,?,?)",
        (username, action, datetime.datetime.now().isoformat(), confidence)
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

# ============================================================
#  SERIAL RELAY (Server → ESP32)
# ============================================================
def send_relay(locker_id, state):
    cmd = json.dumps({"locker_id": str(locker_id), "state": state}) + '\n'
    if ser:
        try:
            ser.write(cmd.encode('utf-8'))
        except Exception as e:
            print(f"❌ Relay error: {e}")
    print(f"[Relay] Tủ {locker_id}: {'MỞ' if state else 'ĐÓNG'}")


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
            print(f"[Auto-close] Tủ {locker_id} đã đóng sau {DOOR_AUTO_CLOSE_S}s")
        except Exception as e:
            print(f"[Auto-close] Lỗi: {e}")

    threading.Thread(target=_close, daemon=True).start()

# ============================================================
#  ESP32 SENSOR → DB
#  Format JSON từ ESP32 (mỗi dòng 1 gói):
#
#  Kiểu 1 - cập nhật 1 tủ:
#    {"locker_id":"1","door":"open","sensor":1}
#    door  : "open" | "closed"
#    sensor: 1 = có đồ trong tủ, 0 = trống
#
#  Kiểu 2 - bulk cập nhật nhiều tủ:
#    {"type":"bulk","data":[{"id":1,"door":"open","sensor":1},
#                           {"id":2,"door":"closed","sensor":0}, ...]}
# ============================================================
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
        if data.get('type') == 'bulk':
            # ── Bulk update ──────────────────────────────────
            for item in data.get('data', []):
                lid       = str(item.get('id', ''))
                door      = 'open' if item.get('door') == 'open' else 'closed'
                has_items = 'yes' if item.get('sensor', 0) else 'no'
                conn.execute(
                    "UPDATE lockers SET door_status=?, has_items=? WHERE locker_id=?",
                    (door, has_items, lid)
                )
            conn.commit()
            print(f"[ESP32 bulk] Cập nhật {len(data.get('data',[]))} tủ")

        elif 'locker_id' in data:
            # ── Single update ────────────────────────────────
            lid       = str(data['locker_id'])
            door      = 'open' if data.get('door') == 'open' else 'closed'
            has_items = 'yes' if data.get('sensor', 0) else 'no'
            conn.execute(
                "UPDATE lockers SET door_status=?, has_items=? WHERE locker_id=?",
                (door, has_items, lid)
            )
            conn.commit()
            print(f"[ESP32] Tủ {lid}: cửa={door}, đồ={has_items}")

    except Exception as e:
        print(f"[ESP32 DB] Lỗi: {e}")
    finally:
        conn.close()


def _serial_reader():
    """Background thread: liên tục đọc JSON từ ESP32 qua Serial"""
    print("✅ Serial reader thread started")
    while True:
        try:
            if ser and ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    process_esp32_data(line)
        except Exception as e:
            print(f"[Serial Reader] {e}")
        time.sleep(0.02)   # 50Hz poll


# Khởi động thread đọc Serial
_serial_thread = threading.Thread(target=_serial_reader, daemon=True)
_serial_thread.start()

# ============================================================
#  HELPER
# ============================================================
def decode_image(img_base64):
    _, encoded = img_base64.split(',', 1)
    return np.array(Image.open(io.BytesIO(base64.b64decode(encoded))).convert('RGB'))


def detect_faces(img_np_rgb):
    results = yolo_model(img_np_rgb, device='cpu', verbose=False)
    return [list(map(int, box.xyxy[0])) for res in results for box in res.boxes]


def get_face_embedding_from_image(img_np_rgb, boxes):
    x1, y1, x2, y2 = boxes[0]
    face_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)[y1:y2, x1:x2]
    return get_embedding(face_bgr)

# ============================================================
#  ROUTES
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')

# ── ESP32 HTTP endpoint (dùng khi ESP32 có WiFi) ─────────────
@app.route('/esp32/update', methods=['POST'])
def esp32_update():
    """
    ESP32 WiFi POST dữ liệu cảm biến về Flask.
    Payload giống format Serial JSON ở trên.
    """
    try:
        process_esp32_data(json.dumps(request.json))
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 400

# ── Lockers ──────────────────────────────────────────────────
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

# ── Analyze hint + BBox + Name (real-time) ───────────────────
@app.route('/analyze_hint', methods=['POST'])
def analyze_hint():
    try:
        img_np = decode_image(request.json.get('img_base64'))
        boxes  = detect_faces(img_np)
        h, w   = img_np.shape[:2]

        if len(boxes) == 0:
            return jsonify({"hint": "Không tìm thấy khuôn mặt", "valid": False,
                            "name": None, "box": None, "img_w": w, "img_h": h})
        if len(boxes) > 1:
            return jsonify({"hint": "Chỉ 1 người đứng trước camera", "valid": False,
                            "name": None, "box": None, "img_w": w, "img_h": h})

        x1, y1, x2, y2 = boxes[0]
        area = (x2 - x1) * (y2 - y1)

        emb        = get_face_embedding_from_image(img_np, boxes)
        name, conf = recognize_face(emb)

        if area < 15000:
            hint, valid = "Vui lòng tiến lại gần hơn", False
        elif area > 45000:
            hint, valid = "Vui lòng lùi ra xa một chút", False
        else:
            valid = True
            hint  = f"✅ Xin chào, {name}!" if name else "✅ Khoảng cách hợp lệ, hãy giữ yên!"

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
        return jsonify({"hint": "Đang phân tích...", "valid": False,
                        "name": None, "box": None, "img_w": 640, "img_h": 480})

# ── Check face in DB ─────────────────────────────────────────
@app.route('/check_face_in_db', methods=['POST'])
def check_face_in_db():
    try:
        img_np     = decode_image(request.json.get('img_base64'))
        boxes      = detect_faces(img_np)
        if len(boxes) != 1:
            return jsonify({"found": False, "error": "Cần đúng 1 khuôn mặt"})
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

# ── Check-in (Gửi đồ) ────────────────────────────────────────
@app.route('/register_locker', methods=['POST'])
def register_locker():
    try:
        data      = request.json
        username  = data.get('username', '').strip()
        gmail     = data.get('gmail', '').strip()
        locker_id = data.get('locker_id', '').strip()
        img_b64   = data.get('img_base64', '')

        if not username or not locker_id or not img_b64:
            return jsonify({'error': 'Thiếu thông tin bắt buộc'}), 400

        conn = get_conn()
        row  = conn.execute("SELECT status FROM lockers WHERE locker_id=?", (locker_id,)).fetchone()
        conn.close()
        if not row:
            return jsonify({'error': f'Tủ {locker_id} không tồn tại'}), 404
        if row['status'] == 'occupied':
            return jsonify({'error': f'Tủ {locker_id} đã có người sử dụng'}), 400

        img_np = decode_image(img_b64)
        boxes  = detect_faces(img_np)
        if len(boxes) == 0:
            return jsonify({'error': 'Không tìm thấy khuôn mặt'}), 400
        if len(boxes) > 1:
            return jsonify({'error': 'Chỉ 1 khuôn mặt trong ảnh'}), 400

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

        log_access(username, 'checkin', 1.0)
        open_then_auto_close(locker_id)

        return jsonify({'success': True,
                        'message': f'Tủ {locker_id} đã mở, tự đóng sau {DOOR_AUTO_CLOSE_S}s.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Check-out (Lấy đồ) ───────────────────────────────────────
@app.route('/face_checkout', methods=['POST'])
def face_checkout():
    try:
        data      = request.json
        locker_id = data.get('locker_id', '').strip()
        img_b64   = data.get('img_base64', '')

        if not locker_id or not img_b64:
            return jsonify({'error': 'Thiếu thông tin'}), 400

        conn  = get_conn()
        row   = conn.execute("SELECT username FROM lockers WHERE locker_id=?",
                             (locker_id,)).fetchone()
        conn.close()
        if not row or not row['username']:
            return jsonify({'error': f'Tủ {locker_id} không có người dùng'}), 400

        owner  = row['username']
        img_np = decode_image(img_b64)
        boxes  = detect_faces(img_np)
        if len(boxes) == 0:
            return jsonify({'error': 'Không tìm thấy khuôn mặt'}), 400
        if len(boxes) > 1:
            return jsonify({'error': 'Chỉ 1 khuôn mặt trong ảnh'}), 400

        emb        = get_face_embedding_from_image(img_np, boxes)
        best, conf = recognize_face(emb)

        if not best:
            return jsonify({'error': 'Khuôn mặt không có trong hệ thống'}), 400
        if best != owner:
            return jsonify({'error': f'Không phải chủ tủ (chủ: {owner})'}), 400

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

        log_access(owner, 'checkout', conf)
        open_then_auto_close(locker_id)

        return jsonify({'success': True,
                        'message': f'Tủ {locker_id} đã mở, tự đóng sau {DOOR_AUTO_CLOSE_S}s.',
                        'confidence': conf})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Admin ─────────────────────────────────────────────────────
@app.route('/admin/login', methods=['POST'])
def admin_login():
    if request.json.get('password') == ADMIN_PASSWORD:
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Sai mật khẩu!'}), 401


@app.route('/admin/stats')
def admin_stats():
    conn     = get_conn()
    occupied = conn.execute("SELECT COUNT(*) as n FROM lockers WHERE status='occupied'").fetchone()['n']
    logs     = conn.execute("SELECT * FROM access_logs ORDER BY timestamp DESC LIMIT 50").fetchall()
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
        return jsonify({'error': 'Xác thực thất bại'}), 401
    locker_id = data.get('locker_id')
    action    = int(data.get('action', 1))
    if not locker_id:
        return jsonify({'error': 'Thiếu locker_id'}), 400
    if action == 1:
        open_then_auto_close(locker_id)
    else:
        send_relay(locker_id, 0)
        conn = get_conn()
        conn.execute("UPDATE lockers SET door_status='closed' WHERE locker_id=?", (str(locker_id),))
        conn.commit()
        conn.close()
    return jsonify({'success': True,
                    'message': f"Đã {'mở' if action else 'đóng'} tủ {locker_id}"})


@app.route('/admin/delete_user', methods=['POST'])
def delete_user():
    data     = request.json
    if data.get('password') != ADMIN_PASSWORD:
        return jsonify({'error': 'Xác thực thất bại'}), 401
    username = data.get('username', '').strip()
    if not username:
        return jsonify({'error': 'Thiếu username'}), 400
    conn = get_conn()
    conn.execute("DELETE FROM faces WHERE username=?", (username,))
    conn.execute(
        "UPDATE lockers SET username=NULL, gmail=NULL, status='available' WHERE username=?",
        (username,)
    )
    conn.commit()
    conn.close()
    _refresh_cache()
    return jsonify({'success': True, 'message': f'Đã xóa {username}'})


if __name__ == '__main__':
    app.run(debug=True, threaded=True, use_reloader=False)