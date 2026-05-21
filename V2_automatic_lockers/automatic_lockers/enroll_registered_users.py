"""
enroll_registered_users.py

Reads `recognition_split.json` and enrolls every identity in
`registered_identities` into the FaceNet embeddings database used by app.py.

Embeddings are computed ONLY from `user_<name>_normal.mp4`.  Mask and spoof
videos are NEVER used for enrollment (they belong to the test set).

For each registered user, the script picks K frames where YOLO returns a
single high-confidence face, averages the FaceNet embeddings, and writes
them through the app's standard save API.

Usage:
  python enroll_registered_users.py --manifest recognition_split.json
  python enroll_registered_users.py --manifest recognition_split.json --frames_per_user 10
  python enroll_registered_users.py --manifest recognition_split.json --replace
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Lazy imports of project modules
# ----------------------------------------------------------------------


def import_app():
    """Import the project's app.py.  All heavy ML deps stay inside app."""
    try:
        import app  # noqa: WPS433
    except Exception as exc:
        print(
            "[ERROR] Could not import app.py.  Run this script from the "
            "project root that contains app.py.\n"
            f"  {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
    return app


def get_save_function(app_mod):
    """Look up the function that persists embeddings.  Falls back to a
    pickle write if no save API is exposed."""
    # save_embedding (singular) takes (username, embedding[, gmail])
    # save_embeddings (plural) takes (dict)
    # register/enroll/add_user take (username, embedding)
    for name in (
        "save_embedding",
        "save_embeddings",
        "register_user",
        "enroll_user",
        "add_user",
    ):
        if hasattr(app_mod, name):
            return name, getattr(app_mod, name)
    return None, None


# ----------------------------------------------------------------------
# Frame sampling
# ----------------------------------------------------------------------


def sample_high_quality_frames(
    video_path: Path,
    yolo_model,
    embed_fn,
    k: int,
    stride: int,
) -> List[np.ndarray]:
    """Return up to K face embeddings from frames that contain exactly one
    high-confidence face.  Frames are sampled at `stride` to skip motion blur
    and to keep enrollment fast.

    `embed_fn` is app.get_face_embedding_from_image(img_np_rgb, boxes_xyxy).
    """
    import cv2  # noqa: WPS433

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] cannot open {video_path.name}", file=sys.stderr)
        return []

    candidates: List[tuple[float, np.ndarray]] = []  # (confidence, embedding)
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                try:
                    # YOLO accepts BGR frame directly
                    results = yolo_model(frame, verbose=False)
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) == 1:
                        conf = float(boxes.conf[0].item())
                        if conf >= 0.6:
                            xyxy = boxes.xyxy[0].cpu().numpy().astype(int).tolist()
                            # embed_fn expects RGB frame + boxes list
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            emb = embed_fn(frame_rgb, [xyxy])
                            if emb is not None:
                                candidates.append((conf, emb))
                except Exception as exc:  # noqa: BLE001
                    print(f"  [WARN] frame {frame_idx} skipped: {exc}", file=sys.stderr)
            frame_idx += 1
    finally:
        cap.release()

    candidates.sort(key=lambda t: t[0], reverse=True)
    return [emb for _, emb in candidates[:k]]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="recognition_split.json")
    p.add_argument("--videos_dir", default="test_videos")
    p.add_argument("--frames_per_user", type=int, default=5)
    p.add_argument("--stride", type=int, default=5, help="Sample every N-th frame for speed")
    p.add_argument("--replace", action="store_true", help="Clear existing DB entry before enrolling")
    p.add_argument("--dry_run", action="store_true", help="Compute embeddings but do not save")
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"[ERROR] Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    registered = manifest["registered_identities"]
    if not registered:
        print("[ERROR] No registered identities in manifest", file=sys.stderr)
        sys.exit(1)

    videos_dir = Path(args.videos_dir)
    if not videos_dir.is_dir():
        print(f"[ERROR] videos_dir not found: {videos_dir}", file=sys.stderr)
        sys.exit(1)

    app_mod = import_app()
    yolo_model = getattr(app_mod, "yolo_model", None)
    embed_fn = getattr(app_mod, "get_face_embedding_from_image", None)
    load_embeddings = getattr(app_mod, "load_embeddings", None)
    if not all([yolo_model, embed_fn]):
        print(
            "[ERROR] app.py must expose yolo_model and "
            "get_face_embedding_from_image",
            file=sys.stderr,
        )
        sys.exit(1)

    save_name, save_fn = get_save_function(app_mod)
    if save_fn is None and not args.dry_run:
        print(
            "[WARN] app.py does not expose a save function "
            "(save_embeddings/register_user/...).  Run with --dry_run or "
            "add a save API.",
            file=sys.stderr,
        )

    existing = load_embeddings() if load_embeddings else {}
    print(f"\nDB currently contains: {list(existing.keys()) or '(empty)'}")
    print(f"Enrolling {len(registered)} identities, K={args.frames_per_user} frames each\n")

    summary: List[Dict] = []
    for name in registered:
        video_path = videos_dir / f"user_{name}_normal.mp4"
        if not video_path.is_file():
            print(f"  [SKIP] {name}: missing {video_path.name}")
            summary.append({"identity": name, "status": "missing_video", "frames_used": 0})
            continue
        if name in existing and not args.replace:
            print(f"  [SKIP] {name}: already in DB (use --replace to overwrite)")
            summary.append({"identity": name, "status": "already_present", "frames_used": 0})
            continue

        print(f"  [ENROLL] {name}: scanning {video_path.name} ...", flush=True)
        embs = sample_high_quality_frames(
            video_path, yolo_model, embed_fn,
            k=args.frames_per_user, stride=args.stride,
        )
        if not embs:
            print(f"    [FAIL] no usable frames in {video_path.name}")
            summary.append({"identity": name, "status": "no_face", "frames_used": 0})
            continue

        mean_emb = np.mean(np.stack(embs, axis=0), axis=0)
        if args.dry_run:
            print(f"    [DRY] computed embedding from {len(embs)} frames (not saved)")
            summary.append({"identity": name, "status": "dry_run", "frames_used": len(embs)})
            continue

        try:
            if save_name in ("save_embedding", "register_user", "enroll_user", "add_user"):
                # (username, embedding) signature
                save_fn(name, mean_emb)
            else:
                # save_embeddings(dict) style
                existing[name] = mean_emb
                save_fn(existing)
            print(f"    [OK] saved via app.{save_name}() using {len(embs)} frames")
            summary.append({"identity": name, "status": "ok", "frames_used": len(embs)})
        except Exception as exc:  # noqa: BLE001
            print(f"    [FAIL] save error: {exc}")
            summary.append({"identity": name, "status": f"save_error:{exc}", "frames_used": len(embs)})

    print("\n=== Enrollment summary ===")
    for row in summary:
        print(f"  {row['identity']:<12}  status={row['status']:<18}  frames={row['frames_used']}")
    ok_count = sum(1 for r in summary if r["status"] == "ok")
    print(f"\nEnrolled {ok_count}/{len(registered)} identities.")
    if load_embeddings:
        final_db = load_embeddings()
        print(f"DB now contains: {list(final_db.keys())}")


if __name__ == "__main__":
    main()
