"""
sanity_check_antispoof.py
=========================

Script kiem tra nhanh xem fix anti-spoof co dung khong, TRUOC KHI chay
full test 36 video (mat ~1 gio).

Chay tren:
  - 1 video user_*.mp4 (live, khong khau trang)
  - 1 video mask_*.mp4 (live, co khau trang)
  - 1 video spoof_*_phone.mp4 (replay attack)

Trich 30 frame deu nhau tu moi video, in mean live_score.

Tieu chi PASS:
  - user video:  mean live_score > 0.70
  - mask video:  mean live_score > 0.40  (khau trang lam giam, nhung khong nen random)
  - spoof video: mean live_score < 0.30
  - Khoang cach user vs spoof > 0.4 (model thuc su phan biet duoc)

Neu FAIL: model van loi -> can debug them.
"""
import os
import sys
import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from app import yolo_model, run_liveness

VIDEO_DIR = "test_videos"
N_FRAMES = 30


def sample_video(path: str, n: int = N_FRAMES):
    """Lay deu n frame tu video."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, min(n, total)).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, fr = cap.read()
        if ret:
            frames.append(fr)
    cap.release()
    return frames


def score_video(path: str):
    """Tra ve mean live_score + so frame detect duoc."""
    frames = sample_video(path)
    if not frames:
        print(f"  [ERROR] khong doc duoc {path}")
        return None

    scores = []
    detected = 0
    for fr_bgr in frames:
        # YOLO can RGB (PIL/Ultralytics convention) hoac BGR deu duoc,
        # nhung de khop voi pipeline app.py thi convert BGR->RGB.
        fr_rgb = cv2.cvtColor(fr_bgr, cv2.COLOR_BGR2RGB)
        results = yolo_model(fr_rgb, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            continue
        confs = results[0].boxes.conf.cpu().numpy()
        xyxy = results[0].boxes.xyxy.cpu().numpy()
        i = int(np.argmax(confs))
        x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
        boxes = [[x1, y1, x2, y2]]

        _, score = run_liveness(fr_rgb, boxes, threshold=0.5)
        scores.append(score)
        detected += 1

    if not scores:
        return None
    return float(np.mean(scores)), float(np.std(scores)), detected, len(frames)


def find_one(prefix: str):
    """Tim 1 video bat dau bang prefix trong VIDEO_DIR."""
    if not os.path.isdir(VIDEO_DIR):
        return None
    for f in sorted(os.listdir(VIDEO_DIR)):
        if f.lower().startswith(prefix):
            return os.path.join(VIDEO_DIR, f)
    return None


def main():
    user_v = find_one("user_")
    mask_v = find_one("mask_")
    spoof_v = find_one("spoof_")

    print("=" * 60)
    print("SANITY CHECK ANTI-SPOOF")
    print("=" * 60)

    results = {}
    for tag, path, expect in [
        ("USER  (live, no mask)", user_v, "> 0.70"),
        ("MASK  (live + mask)  ", mask_v, "> 0.40"),
        ("SPOOF (phone replay) ", spoof_v, "< 0.30"),
    ]:
        if path is None:
            print(f"\n[{tag}] khong tim thay video!")
            continue
        print(f"\n[{tag}] {os.path.basename(path)}  (expect {expect})")
        r = score_video(path)
        if r is None:
            print("  -> khong score duoc")
            continue
        mean, std, det, total = r
        print(f"  mean live_score = {mean:.3f}  (std={std:.3f})")
        print(f"  detected {det}/{total} frames")
        results[tag.strip()] = mean

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    user_s = results.get("USER  (live, no mask)")
    spoof_s = results.get("SPOOF (phone replay)")

    if user_s is not None and spoof_s is not None:
        gap = user_s - spoof_s
        print(f"User-Spoof gap = {gap:.3f}")
        if gap > 0.4 and user_s > 0.70 and spoof_s < 0.30:
            print("PASS: model phan biet duoc live vs spoof. An tam chay full test.")
        elif gap > 0.2:
            print("PARTIAL: phan biet duoc nhung yeu. Co the can tune threshold.")
        else:
            print("FAIL: model van random. KHONG chay full test, can debug them.")
            print("Goi y:")
            print("  1. Kiem tra thu muc resources/anti_spoof_models/ co 2 file .pth khong")
            print("  2. Kiem tra YOLO bbox co dung khong (in box ra xem)")
            print("  3. Thu chay test goc cua repo Silent-Face de chac model con tot")
    else:
        print("Khong du data de verdict.")


if __name__ == "__main__":
    main()
