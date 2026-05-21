import os
import cv2
import numpy as np

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

# Lay duong dan thu muc hien tai
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "resources", "anti_spoof_models")

# Khoi tao model 1 lan duy nhat khi file duoc import de tiet kiem thoi gian
model_test = AntiSpoofPredict(0)
image_cropper = CropImage()


def check_liveness(frame_bgr, bbox, threshold=0.75):
    """
    Anti-spoof inference theo dung API cua Silent-Face-Anti-Spoofing.

    QUAN TRONG: Phai truyen FULL FRAME goc va BBOX goc tu detector,
    KHONG truyen anh da crop sat mat. Ly do: model can nhin vung nen
    xung quanh (vien dien thoai, texture man hinh, anh sang...) de
    phan biet live vs spoof. CropImage se tu mo rong bbox theo scale
    cua tung model (2.7, 4.0) truoc khi resize ve 80x80.

    Args:
        frame_bgr: numpy array, anh GOC tu camera (BGR), KHONG crop.
        bbox: list/tuple [x, y, w, h] tu detector (MTCNN/YOLO/...).
              x, y la goc tren-trai, w, h la chieu rong/cao bbox.
              KHONG dung format [x1, y1, x2, y2].
        threshold: nguong live_score de quyet dinh real.

    Returns:
        (is_real: bool, score: float)
        score la xac suat thuoc class "live" (label = 1), trong [0, 1].
    """
    # Validate bbox
    x, y, w, h = [int(v) for v in bbox]
    H, W = frame_bgr.shape[:2]
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > W or y + h > H:
        # bbox khong hop le -> coi nhu spoof de an toan
        return False, 0.0

    prediction = np.zeros((1, 3), dtype=np.float32)
    n_models = 0

    for model_name in os.listdir(MODEL_DIR):
        model_path = os.path.join(MODEL_DIR, model_name)
        if not os.path.isfile(model_path):
            continue
        if not model_name.endswith(".pth"):
            continue

        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": frame_bgr,        # FULL FRAME, khong crop truoc
            "bbox": [x, y, w, h],         # bbox GOC tu detector
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False

        img = image_cropper.crop(**param)
        prediction += model_test.predict(img, model_path)
        n_models += 1

    if n_models == 0:
        return False, 0.0

    # Trung binh hoa logits qua cac model
    prediction /= n_models

    # Silent-Face: label 0 = spoof_2D, label 1 = live, label 2 = spoof_3D
    # Lay xac suat cua class "live"
    total = float(np.sum(prediction))
    live_score = float(prediction[0][1] / total) if total > 0 else 0.0

    label = int(np.argmax(prediction))
    is_real = (label == 1 and live_score >= threshold)
    return is_real, live_score
