"""
============================================================
 TỔNG HỢP KẾT QUẢ TEST -> XUẤT FILE EXCEL
============================================================
Đọc các file JSON kết quả trong thư mục results/:
  - face_detection_results.json
  - face_recognition_results.json
  - anti_spoofing_results.json

Rồi xuất ra file:
  results/AI_Evaluation_Report.xlsx

File Excel sẽ có 4 sheet:
  1. Face_Detection
  2. Face_Recognition
  3. Anti_Spoofing
  4. Summary  (tổng hợp + tính ACER, so sánh với mục tiêu đặt ra)

Cách chạy:
  python generate_report.py
============================================================
"""

import os
import json
import pandas as pd
from datetime import datetime


RESULTS_DIR = "results"
OUT_FILE = os.path.join(RESULTS_DIR, "AI_Evaluation_Report.xlsx")

# Mục tiêu đề ra trong đồ án
TARGETS = {
    "Detection Rate (YOLO)":      "> 95%",
    "Accuracy (FaceNet)":         "> 95%",
    "FAR (FaceNet)":              "< 5%",
    "FRR (FaceNet)":              "< 5%",
    "APCER (Anti-Spoofing)":      "< 5%",
    "BPCER (Anti-Spoofing)":      "< 5%",
    "ACER (Anti-Spoofing)":       "< 3.5%",
    "Avg Inference Time/module":  "< 40 ms",
}


def load_json(filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return []


def safe_num(value):
    """Cố gắng convert sang float. Nếu không được, trả về None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except Exception:
            return None
    return None


def build_summary(det_records, rec_records, spoof_records):
    """Lấy bản chạy mới nhất của từng loại để tổng hợp."""
    rows = []

    # 1. Face Detection
    if det_records:
        latest = det_records[-1]
        rows.append({
            "Module": "Face Detection (YOLO)",
            "Chỉ số": "Detection Rate (%)",
            "Giá trị đo": latest.get("detection_rate(%)"),
            "Mục tiêu": TARGETS["Detection Rate (YOLO)"],
        })
        rows.append({
            "Module": "Face Detection (YOLO)",
            "Chỉ số": "Avg Inference Time (ms)",
            "Giá trị đo": latest.get("avg_infer_time(ms)"),
            "Mục tiêu": TARGETS["Avg Inference Time/module"],
        })
        rows.append({
            "Module": "Face Detection (YOLO)",
            "Chỉ số": "FPS Inference",
            "Giá trị đo": latest.get("fps_infer"),
            "Mục tiêu": "> 25 FPS",
        })

    # 2. Face Recognition
    if rec_records:
        latest = rec_records[-1]
        rows.append({
            "Module": "Face Recognition (FaceNet)",
            "Chỉ số": "Accuracy (%)",
            "Giá trị đo": latest.get("accuracy(%)"),
            "Mục tiêu": TARGETS["Accuracy (FaceNet)"],
        })
        rows.append({
            "Module": "Face Recognition (FaceNet)",
            "Chỉ số": "FAR (%)",
            "Giá trị đo": latest.get("FAR(%)"),
            "Mục tiêu": TARGETS["FAR (FaceNet)"],
        })
        rows.append({
            "Module": "Face Recognition (FaceNet)",
            "Chỉ số": "FRR (%)",
            "Giá trị đo": latest.get("FRR(%)"),
            "Mục tiêu": TARGETS["FRR (FaceNet)"],
        })
        rows.append({
            "Module": "Face Recognition (FaceNet)",
            "Chỉ số": "Avg Inference Time (ms)",
            "Giá trị đo": latest.get("avg_infer_time(ms)"),
            "Mục tiêu": TARGETS["Avg Inference Time/module"],
        })

    # 3. Anti-Spoofing — Lấy 1 record real và 1 record spoof gần nhất để tính ACER
    apcer_val = None
    bpcer_val = None
    avg_time_spoof = []

    for rec in spoof_records:
        if rec.get("label") == "spoof":
            v = safe_num(rec.get("APCER(%)"))
            if v is not None:
                apcer_val = v
            t = safe_num(rec.get("avg_infer_time(ms)"))
            if t is not None:
                avg_time_spoof.append(t)
        elif rec.get("label") == "real":
            v = safe_num(rec.get("BPCER(%)"))
            if v is not None:
                bpcer_val = v
            t = safe_num(rec.get("avg_infer_time(ms)"))
            if t is not None:
                avg_time_spoof.append(t)

    if apcer_val is not None or bpcer_val is not None:
        rows.append({
            "Module": "Anti-Spoofing (Silent-Face)",
            "Chỉ số": "APCER (%)",
            "Giá trị đo": apcer_val if apcer_val is not None else "Chưa test video spoof",
            "Mục tiêu": TARGETS["APCER (Anti-Spoofing)"],
        })
        rows.append({
            "Module": "Anti-Spoofing (Silent-Face)",
            "Chỉ số": "BPCER (%)",
            "Giá trị đo": bpcer_val if bpcer_val is not None else "Chưa test video real",
            "Mục tiêu": TARGETS["BPCER (Anti-Spoofing)"],
        })

        if apcer_val is not None and bpcer_val is not None:
            acer = round((apcer_val + bpcer_val) / 2.0, 2)
            rows.append({
                "Module": "Anti-Spoofing (Silent-Face)",
                "Chỉ số": "ACER (%) = (APCER+BPCER)/2",
                "Giá trị đo": acer,
                "Mục tiêu": TARGETS["ACER (Anti-Spoofing)"],
            })
        else:
            rows.append({
                "Module": "Anti-Spoofing (Silent-Face)",
                "Chỉ số": "ACER (%)",
                "Giá trị đo": "Cần đủ cả video real & spoof để tính",
                "Mục tiêu": TARGETS["ACER (Anti-Spoofing)"],
            })

        if avg_time_spoof:
            rows.append({
                "Module": "Anti-Spoofing (Silent-Face)",
                "Chỉ số": "Avg Inference Time (ms)",
                "Giá trị đo": round(sum(avg_time_spoof) / len(avg_time_spoof), 2),
                "Mục tiêu": TARGETS["Avg Inference Time/module"],
            })

    return pd.DataFrame(rows)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    det = load_json("face_detection_results.json")
    rec = load_json("face_recognition_results.json")
    spoof = load_json("anti_spoofing_results.json")

    if not (det or rec or spoof):
        print("[WARN] Chưa có kết quả test nào trong thư mục results/.")
        print("       Hãy chạy 3 file test_*.py trước rồi mới chạy generate_report.py.")
        return

    df_det = pd.DataFrame(det) if det else pd.DataFrame()
    df_rec = pd.DataFrame(rec) if rec else pd.DataFrame()
    df_spoof = pd.DataFrame(spoof) if spoof else pd.DataFrame()
    df_summary = build_summary(det, rec, spoof)

    with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
        if not df_summary.empty:
            df_summary.to_excel(writer, sheet_name="Summary", index=False)
        if not df_det.empty:
            df_det.to_excel(writer, sheet_name="Face_Detection", index=False)
        if not df_rec.empty:
            df_rec.to_excel(writer, sheet_name="Face_Recognition", index=False)
        if not df_spoof.empty:
            df_spoof.to_excel(writer, sheet_name="Anti_Spoofing", index=False)

    print(f"\n[OK] Đã xuất báo cáo: {OUT_FILE}")
    print(f"     Thời điểm: {datetime.now().isoformat(timespec='seconds')}")
    if not df_summary.empty:
        print("\n========== BẢNG TỔNG HỢP ==========")
        print(df_summary.to_string(index=False))
        print("====================================\n")


if __name__ == "__main__":
    main()
