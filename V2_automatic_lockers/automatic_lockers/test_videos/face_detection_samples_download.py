import argparse
import math
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'}


def group_of(name: str):
    stem = Path(name).stem.lower()
    if stem.startswith('user_'):
        return 'normal'
    if stem.startswith('mask_'):
        return 'masked'
    if stem.startswith('spoof_'):
        return 'spoof'
    return None


def pick_center_face(dets, w, h):
    if not dets:
        return None
    cx0, cy0 = w / 2.0, h / 2.0
    best, best_score = None, None
    for x1, y1, x2, y2, conf in dets:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        area = max(1.0, (x2 - x1) * (y2 - y1))
        dist = math.hypot(cx - cx0, cy - cy0)
        score = dist - 0.001 * area
        if best_score is None or score < best_score:
            best_score = score
            best = (x1, y1, x2, y2, conf)
    return best


def detect_best_frame(video_path, model, imgsz=640, conf=0.5, max_samples=24):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        total = 1
    sample_ids = sorted(set(np.linspace(0, max(total - 1, 0), num=min(max_samples, total), dtype=int).tolist()))
    best = None
    for idx in sample_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        results = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)
        boxes = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                c = float(box.conf[0].item())
                boxes.append((x1, y1, x2, y2, c))
        if not boxes:
            continue
        h, w = frame.shape[:2]
        face = pick_center_face(boxes, w, h)
        if face is None:
            continue
        x1, y1, x2, y2, c = face
        area = (x2 - x1) * (y2 - y1)
        score = c + 0.000001 * area
        if best is None or score > best['score']:
            best = {
                'frame': frame.copy(),
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'conf': c,
                'score': score,
                'frame_idx': idx,
                'video': video_path.name,
            }
    cap.release()
    return best


def draw_sample(sample, label):
    img = sample['frame'].copy()
    x1, y1, x2, y2 = sample['bbox']
    cv2.rectangle(img, (x1, y1), (x2, y2), (25, 170, 60), 3)
    txt = f"{label} | conf={sample['conf']:.3f}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    y0 = max(0, y1 - 36)
    cv2.rectangle(img, (x1, y0), (x1 + tw + 12, y0 + th + 14), (25, 170, 60), -1)
    cv2.putText(img, txt, (x1 + 6, y0 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def resize_fit(img, target_w=560, target_h=360, pad_color=(245, 245, 245)):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    x0 = (target_w - nw) // 2
    y0 = (target_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def make_triptych(samples, out_path):
    labels = [('Normal', 'normal'), ('Masked', 'masked'), ('Spoof', 'spoof')]
    cards = []
    for title_label, key in labels:
        img = draw_sample(samples[key], title_label)
        img = resize_fit(img, 560, 360)
        band = np.full((56, 560, 3), 255, dtype=np.uint8)
        cv2.putText(band, f"{title_label}: {samples[key]['video']}", (14, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (35, 35, 35), 2, cv2.LINE_AA)
        cards.append(np.vstack([img, band]))
    gap = np.full((416, 26, 3), 255, dtype=np.uint8)
    row = np.hstack([cards[0], gap, cards[1], gap, cards[2]])
    header = np.full((90, row.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, 'Representative face detection results', (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(header, 'One representative sample is shown for each evaluation group after YOLO-based face localization.',
                (22, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 90), 2, cv2.LINE_AA)
    panel = np.vstack([header, row])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


def main():
    parser = argparse.ArgumentParser(description='Create one figure of sample face detection results for the report.')
    parser.add_argument('--videos_dir', default='test_videos', help='Folder containing the evaluation videos')
    parser.add_argument('--model', required=True, help='Path to YOLO face model, e.g. yolov12n-face.pt')
    parser.add_argument('--output', default='output/face_detection_sample_results.jpg', help='Output image path')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.5)
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    model_path = Path(args.model)
    out_path = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Please pass the correct model path, for example:\n"
            "python face_detection_samples_download_v2.py --videos_dir test_videos --model yolov12n-face.pt\n"
            "or\n"
            "python face_detection_samples_download_v2.py --videos_dir . --model ../yolov12n-face.pt"
        )

    if not videos_dir.exists():
        raise FileNotFoundError(f'Video folder not found: {videos_dir}')

    model = YOLO(str(model_path))

    groups = {'normal': [], 'masked': [], 'spoof': []}
    for p in sorted(videos_dir.iterdir()):
        if p.suffix not in VIDEO_EXTS:
            continue
        g = group_of(p.name)
        if g:
            groups[g].append(p)

    for g, files in groups.items():
        if not files:
            raise RuntimeError(f'No videos found for group: {g} in {videos_dir}')

    samples = {}
    for g, files in groups.items():
        best = None
        for f in files:
            s = detect_best_frame(f, model, imgsz=args.imgsz, conf=args.conf)
            if s is None:
                continue
            if best is None or s['score'] > best['score']:
                best = s
        if best is None:
            raise RuntimeError(f'No valid detection sample found for group: {g}')
        samples[g] = best

    make_triptych(samples, out_path)
    print(f'Saved sample figure to: {out_path}')


if __name__ == '__main__':
    main()
