"""
print_attack_detector.py
========================

Tang cuong anti-spoof bang 2nd-stage check chuyen biet cho print attack.

LY DO TON TAI:
    MiniFASNet (Silent-Face-Anti-Spoofing) train chu yeu voi screen replay
    attack (CelebA-Spoof, SiW). Voi print attack chat luong cao (in mau,
    cam tay gan camera), model goc khong phan biet duoc -> APCER ~50%.

    Module nay phan tich texture/frequency cua khuon mat de phat hien
    artifact dac trung cua giay in:
      1. Suy giam tan so cao (high-freq energy)
      2. LBP entropy thap (texture qua deu)
      3. Laplacian variance thap (giay phang, da co micro-structure)

CACH SU DUNG:
    Goi sau MiniFASNet, CHI khi score nam trong vung uncertain (0.5-0.95):
        is_real, score = check_liveness(frame, bbox)
        if 0.5 < score < 0.95:
            # Khong chac chan -> chay them texture check
            is_print = detect_print_attack(frame, bbox)
            if is_print:
                is_real = False
                score = score * 0.3  # penalize score

    Voi score < 0.5 hoac > 0.95, ket qua MiniFASNet du tin cay roi.

PERFORMANCE:
    ~10-15ms tren CPU (so voi MiniFASNet ~200ms) -> overhead khong dang ke.
"""
import cv2
import numpy as np
from typing import Tuple


# Nguong (tuned tu data thuc te, co the chinh sau khi co them sample)
HIGH_FREQ_ENERGY_TH = 0.045     # < nguong nay = nghi print
LBP_ENTROPY_TH      = 4.8       # < nguong nay = texture qua deu (print)
LAP_VAR_TH          = 35.0      # < nguong nay = anh phang (print)

# So criteria can dat de coi la print (cho phep 1 fail -> tranh false alarm)
N_CRITERIA_REQUIRED = 2


def _crop_face(frame: np.ndarray, bbox: Tuple[int, int, int, int],
               size: int = 128, margin: float = 0.15) -> np.ndarray:
    """Crop face vung trung tam mat + 15% margin, resize ve size x size."""
    x, y, w, h = bbox
    cx, cy = x + w // 2, y + h // 2
    half = int(max(w, h) * (1 + margin) / 2)
    H, W = frame.shape[:2]
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(W, cx + half)
    y2 = min(H, cy + half)
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None
    face = cv2.resize(face, (size, size))
    return face


def _high_freq_energy(gray: np.ndarray) -> float:
    """Ti le nang luong tan so cao / tong nang luong (sau FFT).

    Da that co micro-texture (lo chan long, vet nho) -> energy cao.
    Giay in mat chi tiet nay khi qua may in + chup -> energy thap.
    """
    f = np.fft.fft2(gray.astype(np.float32))
    f = np.fft.fftshift(f)
    mag = np.abs(f)

    h, w = gray.shape
    cy, cx = h // 2, w // 2

    # Mask cho vung tan so cao (xa center > 1/4 kich thuoc)
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_cut = min(h, w) // 4

    high_energy = mag[dist > r_cut].sum()
    total_energy = mag.sum()
    return float(high_energy / (total_energy + 1e-9))


def _lbp_entropy(gray: np.ndarray) -> float:
    """Entropy cua histogram LBP (radius=1).

    Da that co nhieu pattern texture phong phu -> entropy cao.
    Giay in texture deu (mau chu nhat tu may in) -> entropy thap.

    Implementation don gian (khong dung skimage).
    """
    h, w = gray.shape
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
    center = gray[1:-1, 1:-1]

    # 8 neighbors theo thu tu chuan
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
               (1, 1), (1, 0), (1, -1), (0, -1)]
    for i, (dy, dx) in enumerate(offsets):
        neigh = gray[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
        lbp |= ((neigh >= center).astype(np.uint8) << i)

    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist / (hist.sum() + 1e-9)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance cua Laplacian (do blur / sharpness).

    Da that 3D co bong + sharpness tu nhien -> variance cao.
    Giay in 2D phang, blur nho do may in + cam camera gan -> variance thap.
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def detect_print_attack(frame_bgr: np.ndarray,
                        bbox: Tuple[int, int, int, int],
                        return_details: bool = False):
    """Tra ve True neu nghi van la print attack.

    Args:
        frame_bgr: full frame BGR.
        bbox: (x, y, w, h) tu face detector.
        return_details: True thi tra ve them dict cac metric.

    Returns:
        is_print: bool
        (optional) details: {high_freq, lbp_entropy, lap_var, fail_count}
    """
    face = _crop_face(frame_bgr, bbox)
    if face is None:
        if return_details:
            return False, {"error": "crop_failed"}
        return False

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    hf  = _high_freq_energy(gray)
    lbp = _lbp_entropy(gray)
    lap = _laplacian_variance(gray)

    fails = 0
    if hf  < HIGH_FREQ_ENERGY_TH: fails += 1
    if lbp < LBP_ENTROPY_TH:      fails += 1
    if lap < LAP_VAR_TH:           fails += 1

    is_print = (fails >= N_CRITERIA_REQUIRED)

    if return_details:
        return is_print, {
            "high_freq_energy": hf,
            "lbp_entropy": lbp,
            "laplacian_var": lap,
            "fails": fails,
            "n_required": N_CRITERIA_REQUIRED,
        }
    return is_print


def enhanced_liveness_check(frame_bgr: np.ndarray,
                            bbox: Tuple[int, int, int, int],
                            mini_score: float,
                            threshold: float = 0.85,
                            uncertain_lo: float = 0.50,
                            uncertain_hi: float = 0.95) -> Tuple[bool, float, str]:
    """Two-stage liveness: ket hop MiniFASNet score + texture check.

    Logic:
      1. Neu mini_score < uncertain_lo -> ro rang la spoof -> reject.
      2. Neu mini_score > uncertain_hi -> ro rang la live -> accept.
      3. Neu trong vung uncertain (0.5-0.95): chay texture check.
         - Neu nghi print attack: reject + penalize score x 0.3
         - Neu khong: dung score MiniFASNet voi threshold.

    Returns:
        (is_real, final_score, reason)
        reason: 'minifas_spoof' / 'minifas_live' / 'texture_print' / 'minifas_uncertain_pass'
    """
    if mini_score < uncertain_lo:
        return False, mini_score, "minifas_spoof"

    if mini_score >= uncertain_hi:
        return True, mini_score, "minifas_live"

    # Vung uncertain -> chay texture check
    is_print = detect_print_attack(frame_bgr, bbox)
    if is_print:
        return False, mini_score * 0.3, "texture_print"

    # Khong phai print, dung MiniFASNet voi threshold
    is_real = mini_score >= threshold
    return is_real, mini_score, "minifas_uncertain_pass"


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python print_attack_detector.py <video_path>")
        sys.exit(1)

    cap = cv2.VideoCapture(sys.argv[1])
    fr_idx = 0
    hits = 0
    samples = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fr_idx += 1
        if fr_idx % 30 != 0:    # sample 1 frame moi 30
            continue
        h, w = frame.shape[:2]
        # Dung center crop lam bbox neu khong co detector
        side = min(h, w) // 2
        bbox = (w // 2 - side // 2, h // 2 - side // 2, side, side)
        is_print, det = detect_print_attack(frame, bbox, return_details=True)
        print(f"frame {fr_idx}: HF={det['high_freq_energy']:.4f} "
              f"LBP={det['lbp_entropy']:.2f} LAP={det['laplacian_var']:.1f} "
              f"fails={det['fails']} -> print={is_print}")
        if is_print:
            hits += 1
        samples += 1

    cap.release()
    print(f"\nTotal: {hits}/{samples} frames classified as print attack")
