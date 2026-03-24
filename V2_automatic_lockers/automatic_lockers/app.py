from flask import Flask, render_template, Response, request, jsonify
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import cv2
import torch
import numpy as np
import sqlite3
import json
from scipy.spatial.distance import cosine
import datetime
import os
import base64
import io
import threading
import time
import serial
from ultralytics import YOLO

# Giới hạn luồng CPU cho hiệu năng tối ưu
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

app = Flask(__name__)
device = torch.device('cpu')

# Load model YOLOv12 pretrain cho face detection
yolo_model = YOLO('automatic_lockers/yolov12n-face.pt')  

# Load FaceNet pretrain cho trích xuất vector embedding
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

DB_FILE = 'automatic_lockers/data/face.db'

db_cache = {}
db_cache_lock = threading.Lock()
last_db_load = 0
DB_CACHE_DURATION = 2

streaming = True
current_frame = None
frame_lock = threading.Lock()

ser = serial.Serial('COM8', 115200, timeout=1) 

def send_relay_command(locker_id, state):
    if ser is None:
        print("Serial port not available")
        return
    cmd = {"locker_id": locker_id, "state": state}
    try:
        ser.write((json.dumps(cmd) + '\n').encode('utf-8'))
        print(f"Sent command: {cmd}")
    except Exception as e:
        print(f"Error sending serial command: {e}")

def serial_reader():
    if ser is None:
        print("Serial port not available for reading")
        return
    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:
                try:
                    data = json.loads(line)
                    # Dữ liệu từ ESP32: {"check_1": 0/1, "check_2": 0/1, "switch_1": 0/1, "switch_2": 0/1}
                    check_1 = data.get('check_1')
                    check_2 = data.get('check_2')
                    switch_1 = data.get('switch_1')
                    switch_2 = data.get('switch_2')
                    
                    conn = get_db_connection()
                    c = conn.cursor()
                    
                    # Cập nhật cho tủ 1
                    if check_1 is not None and switch_1 is not None:
                        door_status_1 = 'closed' if switch_1 == 1 else 'open'
                        has_items_1 = 'yes' if check_1 == 1 else 'no'
                        c.execute("UPDATE lockers SET door_status = ?, has_items = ? WHERE locker_id = ?",
                                 (door_status_1, has_items_1, '1'))
                        print(f"Updated locker 1: door={door_status_1}, items={has_items_1}")
                    
                    # Cập nhật cho tủ 2
                    if check_2 is not None and switch_2 is not None:
                        door_status_2 = 'closed' if switch_2 == 1 else 'open'
                        has_items_2 = 'yes' if check_2 == 1 else 'no'
                        c.execute("UPDATE lockers SET door_status = ?, has_items = ? WHERE locker_id = ?",
                                 (door_status_2, has_items_2, '2'))
                        print(f"Updated locker 2: door={door_status_2}, items={has_items_2}")
                    
                    conn.commit()
                    conn.close()
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing serial data: {e}")
                except Exception as e:
                    print(f"Unexpected error in serial reader: {e}")
        except Exception as e:
            print(f"Error reading from serial port: {e}")
        time.sleep(0.1)

# Hàm kết nối DB
def get_db_connection():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# Khởi tạo DB 
def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute("""CREATE TABLE IF NOT EXISTS faces (
        username TEXT PRIMARY KEY,
        embedding TEXT NOT NULL,
        created_at TEXT
    )""")
    
    c.execute("""CREATE TABLE IF NOT EXISTS lockers (
        locker_id TEXT PRIMARY KEY,
        username TEXT,
        checkin_time TEXT,
        checkout_time TEXT,
        status TEXT DEFAULT 'available',
        door_status TEXT DEFAULT 'closed',
        has_items TEXT DEFAULT 'no'
    )""")
    
    c.execute("""CREATE TABLE IF NOT EXISTS access_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        action TEXT,
        timestamp TEXT,
        confidence REAL
    )""")
    
    conn.commit()
    conn.close()
    print("Database initialized")

init_db()

def save_embedding_db(username, embedding):
    emb_json = json.dumps(embedding.tolist())
    conn = get_db_connection()
    c = conn.cursor()
    now = datetime.datetime.now().isoformat()
    c.execute("INSERT OR REPLACE INTO faces (username, embedding, created_at) VALUES (?, ?, ?)",
              (username, emb_json, now))
    conn.commit()
    conn.close()
    refresh_db_cache()

def load_embeddings_db():
    global db_cache, last_db_load
    current_time = time.time()
    with db_cache_lock:
        if current_time - last_db_load < DB_CACHE_DURATION and db_cache:
            return db_cache
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT username, embedding FROM faces")
        rows = c.fetchall()
        conn.close()
        db_cache = {user: np.array(json.loads(emb)) for user, emb in rows}
        last_db_load = current_time
        return db_cache

def refresh_db_cache():
    global last_db_load
    last_db_load = 0

def get_embedding(face_img):
    if isinstance(face_img, Image.Image):
        face_img = np.array(face_img.convert('RGB'))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

    face_img = cv2.resize(face_img, (160, 160))
    face_tensor = torch.unsqueeze(torch.tensor(face_img).float().permute(2, 0, 1), 0) / 255.0
    face_tensor = face_tensor.to(device)

    with torch.no_grad():
        embedding = resnet(face_tensor).cpu().numpy()[0]
    return embedding

def log_access(username, action, confidence=None):
    conn = get_db_connection()
    c = conn.cursor()
    now = datetime.datetime.now().isoformat()
    c.execute("INSERT INTO access_logs (username, action, timestamp, confidence) VALUES (?, ?, ?, ?)",
              (username, action, now, confidence))
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    global streaming, current_frame
    cap = None
    db = load_embeddings_db()
    frame_count = 0

    while True:
        if streaming:
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                    time.sleep(0.5)
                cap = cv2.VideoCapture(1)
                if not cap.isOpened():
                    print("Cannot open camera")
                    time.sleep(1)
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print("Camera opened")

            ret, frame = cap.read()
            if not ret:
                if cap is not None:
                    cap.release()
                    cap = None
                time.sleep(0.5)
                continue

            frame = cv2.flip(frame, 1)
            frame_count += 1

            if frame_count % 1 == 0:
                db = load_embeddings_db()
                try:
                    results = yolo_model(frame, device='cpu')
                    boxes = []
                    for res in results:
                        for box in res.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            boxes.append((x1, y1, x2, y2))

                    for box in boxes:
                        x1, y1, x2, y2 = box
                        face_img = frame[y1:y2, x1:x2]
                        if face_img.size == 0:
                            continue
                        embedding = get_embedding(face_img)

                        min_dist = float('inf')
                        name = "Unknown"
                        for user, db_emb in db.items():
                            dist = cosine(embedding, db_emb)
                            if dist < min_dist:
                                min_dist = dist
                                if dist < 0.55:
                                    name = user
                                else:
                                    name = "Unknown"

                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        confidence = 1 - min_dist
                        cv2.putText(frame, name, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                except Exception as e:
                    print(f"Detection error: {e}")

            with frame_lock:
                current_frame = frame.copy()

            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            if cap is not None:
                cap.release()
                cap = None
                time.sleep(0.5)
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_stream_status', methods=['POST'])
def set_stream_status():
    global streaming
    data = request.json
    streaming = bool(data.get('streaming', True))
    return jsonify({'success': True})

@app.route('/register_locker', methods=['POST'])
def register_locker():
    try:
        data = request.json
        username = data.get('username', '').strip()
        locker_id = data.get('locker_id', '').strip()
        img_base64 = data.get('img_base64')

        if not username or not locker_id or not img_base64:
            return jsonify({'error': 'Thiếu dữ liệu đăng ký'}), 400

        if len(username) < 2:
            return jsonify({'error': 'Tên người dùng phải có ít nhất 2 ký tự'}), 400

        try:
            header, encoded = img_base64.split(',', 1)
            img_bytes = base64.b64decode(encoded)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception as e:
            return jsonify({'error': 'Lỗi xử lý ảnh'}), 400

        img_np = np.array(img)
        results = yolo_model(img_np, device='cpu')

        boxes = []
        for res in results:
            for box in res.boxes:
                boxes.append(box.xyxy[0].int().tolist())

        if len(boxes) == 0:
            return jsonify({'error': 'Không tìm thấy khuôn mặt trong ảnh'}), 400
        if len(boxes) > 1:
            return jsonify({'error': 'Phát hiện nhiều khuôn mặt. Vui lòng chỉ có 1 người'}), 400

        x1, y1, x2, y2 = boxes[0]
        face_crop_np = img_np[y1:y2, x1:x2, :]
        face_crop = Image.fromarray(face_crop_np)

        embedding = get_embedding(face_crop)
        save_embedding_db(username, embedding)

        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT username FROM lockers WHERE locker_id = ?", (locker_id,))
        res = c.fetchone()
        if res and res['username'] is not None:
            conn.close()
            return jsonify({'error': f'Tủ {locker_id} đã được gán cho {res["username"]}'}), 400

        now = datetime.datetime.now().isoformat()
        c.execute("""INSERT OR REPLACE INTO lockers (locker_id, username, checkin_time, checkout_time, status, door_status, has_items)
                     VALUES (?, ?, ?, NULL, 'occupied', 'closed', 'no')""", (locker_id, username, now))
        conn.commit()
        conn.close()

        log_access(username, 'register', 1.0)
        refresh_db_cache()

        send_relay_command(locker_id, 1)
        time.sleep(5)
        send_relay_command(locker_id, 0)

        return jsonify({'success': True, 'message': f'Đăng ký thành công tủ {locker_id}'})
    
    except Exception as e:
        return jsonify({'error': f'Có lỗi xảy ra: {str(e)}'}), 500

@app.route('/list_lockers')
def list_lockers():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT locker_id, username, checkin_time, checkout_time, status, door_status, has_items FROM lockers ORDER BY locker_id")
    rows = c.fetchall()
    conn.close()

    lockers_list = []
    for row in rows:
        lockers_list.append({
            'locker_id': row['locker_id'],
            'username': row['username'],
            'checkin_time': row['checkin_time'],
            'checkout_time': row['checkout_time'],
            'status': row['status'],
            'door_status': row['door_status'],
            'has_items': row['has_items']
        })
    return jsonify(lockers_list)

@app.route('/list_customers')
def list_customers():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        SELECT username,
               CASE
                   WHEN checkout_time IS NULL THEN 'Đang sử dụng'
                   ELSE 'Đã trả tủ'
               END AS status,
               locker_id,
               checkin_time,
               checkout_time
        FROM lockers
        WHERE username IS NOT NULL
        ORDER BY checkin_time DESC
    """)
    rows = c.fetchall()
    conn.close()

    result = [{'username': row['username'], 
               'status': row['status'], 
               'locker_id': row['locker_id'],
               'checkin_time': row['checkin_time'],
               'checkout_time': row['checkout_time']}
              for row in rows]
    return jsonify(result)

@app.route('/manual_checkout', methods=['POST'])
def manual_checkout():
    data = request.json
    locker_id = data.get('locker_id', '').strip()

    if not locker_id:
        return jsonify({'error': 'Thiếu mã tủ'}), 400

    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT username FROM lockers WHERE locker_id = ? AND checkout_time IS NULL", (locker_id,))
    result = c.fetchone()

    if not result:
        conn.close()
        return jsonify({'error': 'Tủ không được sử dụng hoặc đã trả'}), 400

    username = result['username']
    now = datetime.datetime.now().isoformat()
    c.execute("""UPDATE lockers SET checkout_time = ?, status = 'available', username = NULL
                 WHERE locker_id = ?""", (now, locker_id))
    conn.commit()
    conn.close()

    log_access(username, 'manual_checkout', None)
    send_relay_command(locker_id, 1)
    time.sleep(5)
    send_relay_command(locker_id, 0)

    return jsonify({'success': True, 'message': f'Đã trả tủ {locker_id} của {username}'})

@app.route('/init_lockers', methods=['POST'])
def init_lockers():
    conn = get_db_connection()
    c = conn.cursor()

    for locker_id in ['1', '2']:
        c.execute("""INSERT OR IGNORE INTO lockers
                    (locker_id, username, checkin_time, checkout_time, status, door_status, has_items)
                    VALUES (?, NULL, NULL, NULL, 'available', 'closed', 'no')""", (locker_id,))

    conn.commit()
    conn.close()

    return jsonify({'success': True, 'message': 'Đã khởi tạo 2 tủ đồ'})

@app.route('/delete_user', methods=['POST'])
def delete_user():
    username = request.json.get('username', '').strip()
    if not username:
        return jsonify({'error': 'Vui lòng nhập tên user cần xóa'}), 400

    conn = get_db_connection()
    c = conn.cursor()

    c.execute("SELECT COUNT(*) as count FROM faces WHERE username = ?", (username,))
    if c.fetchone()['count'] == 0:
        conn.close()
        return jsonify({'error': 'Không tìm thấy user'}), 404

    c.execute("DELETE FROM faces WHERE username = ?", (username,))
    now = datetime.datetime.now().isoformat()
    c.execute("""UPDATE lockers SET username = NULL, checkout_time = ?, status = 'available'
                 WHERE username = ?""", (now, username))
    conn.commit()
    conn.close()

    log_access(username, 'delete', None)
    refresh_db_cache()

    return jsonify({'success': True, 'message': f'Đã xóa {username}'})

@app.route('/face_checkout', methods=['POST'])
def face_checkout():
    try:
        data = request.json
        img_base64 = data.get('img_base64')

        if not img_base64:
            return jsonify({'error': 'Thiếu dữ liệu ảnh'}), 400

        try:
            header, encoded = img_base64.split(',', 1)
            img_bytes = base64.b64decode(encoded)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception as e:
            return jsonify({'error': 'Lỗi xử lý ảnh'}), 400

        img_np = np.array(img)
        results = yolo_model(img_np, device='cpu')

        boxes = []
        for res in results:
            for box in res.boxes:
                boxes.append(box.xyxy[0].int().tolist())

        if len(boxes) == 0:
            return jsonify({'error': 'Không tìm thấy khuôn mặt trong ảnh'}), 400
        if len(boxes) > 1:
            return jsonify({'error': 'Phát hiện nhiều khuôn mặt. Vui lòng chỉ có 1 người'}), 400

        x1, y1, x2, y2 = boxes[0]
        face_crop_np = img_np[y1:y2, x1:x2, :]
        face_crop = Image.fromarray(face_crop_np)

        embedding = get_embedding(face_crop)

        db = load_embeddings_db()
        min_dist = float('inf')
        matched_user = None

        for user, db_emb in db.items():
            dist = cosine(embedding, db_emb)
            if dist < min_dist:
                min_dist = dist
                if dist < 0.55:
                    matched_user = user

        if not matched_user:
            return jsonify({'error': 'Không nhận diện được khuôn mặt. Vui lòng thử lại'}), 400

        conn = get_db_connection()
        c = conn.cursor()
        c.execute("""SELECT locker_id, checkout_time FROM lockers
                     WHERE username = ? AND checkout_time IS NULL""", (matched_user,))
        result = c.fetchone()

        if not result:
            conn.close()
            return jsonify({'error': f'Người dùng {matched_user} không có tủ nào đang sử dụng'}), 400

        locker_id = result['locker_id']
        now = datetime.datetime.now().isoformat()
        confidence = 1 - min_dist

        c.execute("""UPDATE lockers SET checkout_time = ?, status = 'available', username = NULL
                     WHERE locker_id = ?""", (now, locker_id))

        c.execute("DELETE FROM faces WHERE username = ?", (matched_user,))

        conn.commit()
        conn.close()

        log_access(matched_user, 'face_checkout', confidence)

        send_relay_command(locker_id, 1)
        time.sleep(5)
        send_relay_command(locker_id, 0)

        return jsonify({
            'success': True,
            'message': f'Check-out thành công cho {matched_user}',
            'locker_id': locker_id,
            'username': matched_user,
            'confidence': round(confidence * 100, 1)
        })

    except Exception as e:
        return jsonify({'error': f'Có lỗi xảy ra: {str(e)}'}), 500

@app.route('/statistics')
def statistics():
    conn = get_db_connection()
    c = conn.cursor()

    c.execute("SELECT COUNT(*) as total FROM faces")
    total_users = c.fetchone()['total']

    c.execute("SELECT COUNT(*) as occupied FROM lockers WHERE checkout_time IS NULL")
    occupied_lockers = c.fetchone()['occupied']

    c.execute("SELECT COUNT(*) as total FROM lockers")
    total_lockers = c.fetchone()['total']

    c.execute("""SELECT username, action, timestamp, confidence
                 FROM access_logs ORDER BY timestamp DESC LIMIT 10""")
    recent_logs = [dict(row) for row in c.fetchall()]

    conn.close()

    return jsonify({
        'total_users': total_users,
        'occupied_lockers': occupied_lockers,
        'available_lockers': total_lockers - occupied_lockers,
        'total_lockers': total_lockers,
        'recent_logs': recent_logs
    })

@app.route('/export_logs', methods=['POST'])
def export_logs():
    import csv
    from io import StringIO

    data = request.json
    filter_type = data.get('filter_type', 'all')
    filter_value = data.get('filter_value', '')

    conn = get_db_connection()
    c = conn.cursor()

    if filter_type == 'date' and filter_value:
        c.execute("""SELECT username, action, timestamp, confidence
                     FROM access_logs
                     WHERE DATE(timestamp) = ?
                     ORDER BY timestamp DESC""", (filter_value,))
    elif filter_type == 'hour' and filter_value:
        c.execute("""SELECT username, action, timestamp, confidence
                     FROM access_logs
                     WHERE strftime('%Y-%m-%d %H', timestamp) = ?
                     ORDER BY timestamp DESC""", (filter_value,))
    else:
        c.execute("""SELECT username, action, timestamp, confidence
                     FROM access_logs
                     ORDER BY timestamp DESC""")

    logs = c.fetchall()
    conn.close()

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Người dùng', 'Hành động', 'Thời gian', 'Độ tin cậy (%)'])

    for log in logs:
        confidence = f"{log['confidence']*100:.1f}%" if log['confidence'] else '-'
        writer.writerow([
            log['username'],
            log['action'],
            log['timestamp'],
            confidence
        ])

    csv_content = output.getvalue()
    output.close()

    return jsonify({
        'success': True,
        'csv_content': csv_content,
        'filename': f'access_logs_{filter_type}_{filter_value or "all"}.csv'
    })

if __name__ == '__main__':
    if ser is not None:
        threading.Thread(target=serial_reader, daemon=True).start()
    app.run(debug=True, threaded=True, use_reloader=False)