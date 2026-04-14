"""
Predict attractiveness score for a single image and show the debug overlay.

Usage:
  uv run python scripts/predict.py path/to/photo.jpg
  uv run python scripts/predict.py path/to/photo.jpg --model ensemble
"""

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.registry import load_model
from scripts.process import compute_features, draw_debug_overlay, MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description="Predict attractiveness for a single image")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--model", default="xgboost",
                        choices=["xgboost", "ensemble", "mlp", "quantile", "ranker"],
                        help="Which trained model to use (default: xgboost)")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="Disable variance calibration")
    args = parser.parse_args()

    img_path = Path(args.image).resolve()
    if not img_path.exists():
        print(f"ERROR: {img_path} not found")
        sys.exit(1)

    # Load saved model (instant, no retraining)
    print(f"Loading {args.model} model...")
    model = load_model(args.model)
    ds = model.dataset_stats

    # Process input image
    print(f"Processing {img_path.name}...")
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print("ERROR: Could not read image")
        sys.exit(1)

    h, w = img_bgr.shape[:2]

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

    landmarker = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        num_faces=1,
        output_face_blendshapes=True,
    ))

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
    )
    results = landmarker.detect(mp_image)
    landmarker.close()

    if not results.face_landmarks:
        print("ERROR: No face detected in image")
        sys.exit(1)

    landmarks = results.face_landmarks[0]
    features = compute_features(landmarks, w, h)
    if features is None:
        print("ERROR: Could not compute features")
        sys.exit(1)

    # Blendshapes
    if results.face_blendshapes:
        bs = {b.category_name: b.score for b in results.face_blendshapes[0]}
        features["expr_smile"] = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
        features["expr_frown"] = (bs.get("mouthFrownLeft", 0) + bs.get("mouthFrownRight", 0)) / 2
        features["expr_jaw_open"] = bs.get("jawOpen", 0)
        features["expr_brow_up"] = bs.get("browInnerUp", 0)
        features["expr_brow_down"] = (bs.get("browDownLeft", 0) + bs.get("browDownRight", 0)) / 2
        features["expr_cheek_squint"] = (bs.get("cheekSquintLeft", 0) + bs.get("cheekSquintRight", 0)) / 2
        features["expr_eye_squint"] = (bs.get("eyeSquintLeft", 0) + bs.get("eyeSquintRight", 0)) / 2
        features["expr_eye_wide"] = (bs.get("eyeWideLeft", 0) + bs.get("eyeWideRight", 0)) / 2
        features["expr_mouth_pucker"] = bs.get("mouthPucker", 0)
    else:
        for k in ["expr_smile", "expr_frown", "expr_jaw_open", "expr_brow_up",
                   "expr_brow_down", "expr_cheek_squint", "expr_eye_squint",
                   "expr_eye_wide", "expr_mouth_pucker"]:
            features[k] = 0.0

    # Predict
    feature_cols = model.feature_cols
    X = np.array([[features[c] for c in feature_cols]])

    if args.no_calibrate:
        z = float(model.predict(X)[0])
    else:
        z = float(model.predict_calibrated(X)[0])

    me_mean, me_std = ds["mebeauty"]["mean"], ds["mebeauty"]["std"]
    sc_mean, sc_std = ds["scut"]["mean"], ds["scut"]["std"]

    print()
    print("=" * 50)
    print(f"  Model:              {args.model}")
    print(f"  Score (z):          {z:+.2f}")
    print(f"  Score (1-10 scale): {z * me_std + me_mean:.1f}")
    print(f"  Score (1-5 scale):  {z * sc_std + sc_mean:.1f}")
    print("=" * 50)

    # Save overlay
    overlay = draw_debug_overlay(img_bgr, landmarks, w, h, features)
    out_path = img_path.parent / f"{img_path.stem}_analysis.jpg"
    cv2.imwrite(str(out_path), overlay)
    print(f"\nOverlay saved to: {out_path}")


if __name__ == "__main__":
    main()
