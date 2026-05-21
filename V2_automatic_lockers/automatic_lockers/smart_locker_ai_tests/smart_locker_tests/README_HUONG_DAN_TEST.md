# Hướng dẫn chạy bộ test cho đồ án Smart Locker

Bộ test này gồm **3 file test độc lập** + **1 file tổng hợp Excel**, dùng để chứng minh hiệu quả của 3 mô hình AI trong đồ án:

| File | Mục đích | Chỉ số đo |
|---|---|---|
| `test_face_detection.py` | Test riêng **YOLO** — bắt khuôn mặt | Detection Rate, FPS, Time |
| `test_face_recognition.py` | Test riêng **FaceNet** — so khớp danh tính | Accuracy, FAR, FRR, Cosine Distance |
| `test_anti_spoofing.py` | Test riêng **Silent-Face** — chống giả mạo | APCER, BPCER, ACER (chuẩn ISO/IEC 30107-3) |
| `generate_report.py` | Tổng hợp 3 kết quả ra file **Excel** | Bảng Summary + tính ACER tự động |

---

## 1. Cấu trúc thư mục

Đặt 4 file `.py` vào **cùng cấp với `app.py`**:

```
/ThuMucDoAn
 ├── app.py
 ├── anti_spoof_service.py
 ├── yolov12n-face.pt
 ├── test_face_detection.py
 ├── test_face_recognition.py
 ├── test_anti_spoofing.py
 ├── generate_report.py
 ├── /test_videos
 │    ├── video_user.mp4        (video bạn — đã đăng ký DB)
 │    ├── video_stranger.mp4    (video người lạ — KHÔNG có trong DB)
 │    └── video_spoof.mp4       (quay điện thoại đang phát hình bạn)
 └── /results                   (tự tạo khi chạy)
```

---

## 2. Chuẩn bị 3 video test (~10 giây mỗi video, định dạng .mp4)

| Video | Cách quay |
|---|---|
| `video_user.mp4` | Bạn đứng trước camera, **mặt bạn đã có trong DB** (chính là user đã đăng ký). Quay tự nhiên ~10 giây. |
| `video_stranger.mp4` | Một người khác (chưa đăng ký) đứng trước camera. |
| `video_spoof.mp4` | Mở điện thoại, phát một video/ảnh có khuôn mặt bạn lên rồi đưa điện thoại trước webcam. |

> **Mẹo để số đẹp:** Quay trong cùng điều kiện ánh sáng với lúc đăng ký DB. Mặt chính diện, không bị che, khoảng cách ~50–80 cm.

---

## 3. Cài thêm thư viện (nếu chưa có)

```bash
pip install opencv-python numpy pandas openpyxl
```

(Các thư viện AI như `ultralytics`, `torch`, `silent_face`… đã có sẵn vì `app.py` đang dùng.)

---

## 4. Chạy lần lượt 3 bài test

### Bài 1 — Face Detection (YOLO)

```bash
# Video có khuôn mặt
python test_face_detection.py --video test_videos/video_user.mp4 --label co_mat

# (Tuỳ chọn) Video cảnh trống để đo False Positive
python test_face_detection.py --video test_videos/video_empty.mp4 --label khong_co_mat
```

### Bài 2 — Face Recognition (FaceNet)

```bash
# Video người đã đăng ký — đổi tên cho khớp username trong DB
python test_face_recognition.py --video test_videos/video_user.mp4 --label "Nguyen Thi Huong"

# Video người lạ — kỳ vọng kết quả là Unknown
python test_face_recognition.py --video test_videos/video_stranger.mp4 --label "Unknown"
```

### Bài 3 — Anti-Spoofing (Silent-Face)

```bash
# Phải chạy CẢ 2 video mới tính được ACER
python test_anti_spoofing.py --video test_videos/video_user.mp4   --label real
python test_anti_spoofing.py --video test_videos/video_spoof.mp4  --label spoof
```

---

## 5. Xuất báo cáo Excel

```bash
python generate_report.py
```

Sẽ tạo ra file: **`results/AI_Evaluation_Report.xlsx`** gồm 4 sheet:
- `Summary` — bảng tổng hợp + tính ACER + so sánh với mục tiêu
- `Face_Detection`
- `Face_Recognition`
- `Anti_Spoofing`

Đây là file bạn copy bảng vào chương "Đánh giá / Kết quả thực nghiệm" của báo cáo tốt nghiệp.

---

## 6. Mục tiêu các chỉ số (theo đồ án đề ra)

| Chỉ số | Mục tiêu | Ý nghĩa |
|---|---|---|
| Detection Rate (YOLO) | > 95% | Đi qua camera là bắt được mặt ngay |
| Accuracy (FaceNet) | > 95% | Nhận đúng người đã đăng ký |
| FAR | < 5% | Người lạ KHÔNG được nhận nhầm thành user |
| FRR | < 5% | User KHÔNG bị từ chối nhầm |
| APCER | < 5% | Ảnh giả KHÔNG được nhận thành thật |
| BPCER | < 5% | Người thật KHÔNG bị nhận nhầm thành giả |
| ACER | < 3.5% | Chỉ số tổng hợp anti-spoofing |
| Time / module | < 40 ms | Đảm bảo hệ thống đạt ≥ 15–20 FPS |

---

## 7. Một số lưu ý khi bảo vệ

- **Khi hội đồng hỏi vì sao tách 3 bài test?** Trả lời: *"Em đánh giá theo phương pháp **Modular Evaluation** — tách riêng từng module trong pipeline để xác định chính xác lỗi (nếu có) thuộc về YOLO, FaceNet hay Silent-Face, tránh hiện tượng cascading error."*
- **Vì sao có APCER/BPCER/ACER mà không dùng Accuracy bình thường?** Trả lời: *"Đây là chỉ số chuẩn ISO/IEC 30107-3 dành riêng cho bài toán Presentation Attack Detection."*
- Nếu chạy ra số chưa đạt mục tiêu → quay lại điều chỉnh:
  - Ánh sáng khi quay
  - Ngưỡng confidence của YOLO
  - Threshold cosine của FaceNet
  - Threshold liveness của Silent-Face

Chúc bảo vệ thành công.
