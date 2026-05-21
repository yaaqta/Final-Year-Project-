import serial
import json
import time
import threading

# Cấu hình cổng COM 
SERIAL_PORT = 'COM8'
BAUD_RATE = 115200

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Kết nối thành công với {SERIAL_PORT}")
except Exception as e:
    print(f"Lỗi kết nối: {e}")
    ser = None

# Biến để chặn việc in liên tục
last_data_stamped = ""

def listen_to_hardware():
    global last_data_stamped
    while True:
        if ser and ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8').strip()
                # Kiểm tra xem dữ liệu mới có khác dữ liệu cũ không
                if line and line != last_data_stamped:
                    data = json.loads(line)
                    
                    # Chỉ in khi có sự thay đổi thực sự từ cảm biến
                    print("\n" + "-"*30)
                    print("CẬP NHẬT TRẠNG THÁI MỚI:")
                    print(f"  Tủ 1: {'ĐÓNG' if data.get('switch_1')==1 else 'MỞ'} | Vật dụng: {'CÓ' if data.get('check_1')==1 else 'KHÔNG'}")
                    print(f"  Tủ 2: {'ĐÓNG' if data.get('switch_2')==1 else 'MỞ'} | Vật dụng: {'CÓ' if data.get('check_2')==1 else 'KHÔNG'}")
                    print("-"*30)
                    print("Nhập lệnh (ví dụ '1 1') hoặc 'q' để thoát: ", end="", flush=True)
                    
                    last_data_stamped = line # Lưu lại trạng thái vừa in
            except:
                pass
        time.sleep(0.1)

def send_command(locker_id, state):
    if ser:
        try:
            cmd = {"locker_id": str(locker_id), "state": state}
            ser.write((json.dumps(cmd) + '\n').encode('utf-8'))
            print(f"Đã gửi lệnh: Tủ {locker_id} -> {'MỞ' if state==1 else 'ĐÓNG'}")
        except Exception as e:
            print(f"Lỗi gửi: {e}")

# Chạy luồng đọc dữ liệu ngầm
if ser:
    t = threading.Thread(target=listen_to_hardware, daemon=True)
    t.start()

print("\nHỆ THỐNG TEST PHẦN CỨNG ĐÃ SẴN SÀNG")
print("Cú pháp: [ID] [Trạng thái] (Ví dụ: '1 1' là mở tủ 1)")

try:
    while True:
        # Nhận lệnh từ bàn phím
        cmd_input = input("\nLệnh đóng, mở tủ: ").strip().lower()
        
        if cmd_input == 'q':
            break
        
        parts = cmd_input.split()
        if len(parts) == 2:
            l_id, l_state = parts[0], int(parts[1])
            send_command(l_id, l_state)
        elif cmd_input == "":
            print("Hướng dẫn: Nhập '1 1' để mở tủ, '1 0' để đóng.")
except KeyboardInterrupt:
    pass
finally:
    if ser:
        ser.close()
    print("\nĐã ngắt kết nối")