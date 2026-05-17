"""
Web app for facial attractiveness analysis.

Usage:
  uv run python scripts/app.py
  uv run python scripts/app.py --model ensemble
  # Then open http://localhost:5001
"""

import argparse
import base64
import os
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.registry import load_model
from scripts.process import MODEL_PATH, compute_features, draw_lines_overlay

# ---------------------------------------------------------------------------
# Parse args / env
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Face analysis web app")
parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "xgboost"),
                    choices=["xgboost", "ensemble", "mlp", "quantile", "ranker"])
parser.add_argument("--port", type=int, default=5001)
args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------
# Load model + MediaPipe at startup
# ---------------------------------------------------------------------------
print(f"Loading {args.model} model...")
model = load_model(args.model)
ds = model.dataset_stats
feature_cols = model.feature_cols
me_mean, me_std = ds["mebeauty"]["mean"], ds["mebeauty"]["std"]
sc_mean, sc_std = ds["scut"]["mean"], ds["scut"]["std"]

print("Loading MediaPipe FaceLandmarker...")
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

landmarker = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    num_faces=1,
    output_face_blendshapes=True,
))

print("Ready!")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


def analyze_image(img_bgr):
    """Run the full pipeline on a BGR image. Returns (scores_dict, overlay_bgr) or raises."""
    h, w = img_bgr.shape[:2]
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
    )
    results = landmarker.detect(mp_image)

    if not results.face_landmarks:
        raise ValueError("No face detected in image")

    landmarks = results.face_landmarks[0]
    features = compute_features(landmarks, w, h)
    if features is None:
        raise ValueError("Could not compute features")

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

    X = np.array([[features[c] for c in feature_cols]])
    z = float(model.predict_calibrated(X)[0])

    scores = {
        "z": round(z, 2),
        "score_10": round(z * me_std + me_mean, 1),
        "score_5": round(z * sc_std + sc_mean, 1),
    }

    overlay = draw_lines_overlay(img_bgr, landmarks, w, h)
    return scores, overlay


@app.route("/")
def index():
    return HTML_PAGE


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({"error": "Could not decode image"}), 400

    try:
        scores, overlay = analyze_image(img_bgr)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    _, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
    overlay_b64 = base64.b64encode(buf).decode("utf-8")

    return jsonify({**scores, "overlay": overlay_b64})


# ---------------------------------------------------------------------------
# Inline HTML/CSS/JS
# ---------------------------------------------------------------------------
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Facial Attractiveness Analysis</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #ffffff; color: #1a1a1a; min-height: 100vh;
    display: flex; flex-direction: column; align-items: center;
  }
  header {
    width: 100%%; padding: 32px 24px 24px;
    border-bottom: 1px solid #eaeaea; text-align: center;
  }
  h1 { font-size: 1.4rem; font-weight: 600; letter-spacing: -0.01em; }
  .subtitle { color: #666; font-size: 0.85rem; margin-top: 6px; }

  main { width: 100%%; max-width: 1100px; padding: 32px 24px; }

  #dropzone {
    width: 100%%; min-height: 220px;
    border: 1.5px dashed #d0d0d0; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    flex-direction: column; gap: 10px;
    transition: border-color 0.15s, background 0.15s;
    cursor: pointer; padding: 40px;
    background: #fafafa;
  }
  #dropzone:hover, #dropzone.dragover {
    border-color: #1a1a1a; background: #f5f5f5;
  }
  #dropzone p { color: #666; font-size: 0.95rem; }
  #dropzone input { display: none; }

  #loading { display: none; margin: 32px 0; color: #666; text-align: center; }
  #loading.active { display: block; }

  #results { display: none; }
  #results.active {
    display: grid;
    grid-template-columns: minmax(0, 2fr) minmax(280px, 1fr);
    gap: 32px; align-items: start;
  }

  #overlay-img {
    width: 100%%; border-radius: 8px;
    border: 1px solid #eaeaea;
  }

  .scores { display: flex; flex-direction: column; gap: 12px; }
  .score-card {
    border: 1px solid #eaeaea; border-radius: 8px; padding: 18px 20px;
    background: #ffffff;
  }
  .score-card .label {
    font-size: 0.72rem; color: #888; text-transform: uppercase;
    letter-spacing: 0.06em; font-weight: 500;
  }
  .score-card .value {
    font-size: 2.2rem; font-weight: 600; margin: 4px 0 2px;
    color: #1a1a1a; letter-spacing: -0.02em;
  }
  .score-card .scale { font-size: 0.78rem; color: #888; }

  .retry { margin-top: 8px; }
  .retry button {
    background: #ffffff; color: #1a1a1a; border: 1px solid #d0d0d0;
    padding: 9px 16px; border-radius: 6px; cursor: pointer;
    font-size: 0.85rem; font-family: inherit;
    transition: border-color 0.15s, background 0.15s;
  }
  .retry button:hover { border-color: #1a1a1a; background: #fafafa; }

  #error { color: #c1272d; margin: 20px 0; display: none; text-align: center; }
  #error.active { display: block; }
</style>
</head>
<body>

<header>
  <h1>Facial Attractiveness Analysis</h1>
  <p class="subtitle">Geometric beauty markers + """ + args.model + """</p>
</header>

<main>
  <div id="dropzone" onclick="document.getElementById('fileinput').click()">
    <p>Drag &amp; drop a photo here, or click to select</p>
    <input type="file" id="fileinput" accept="image/*">
  </div>

  <div id="loading">Analyzing&hellip;</div>
  <div id="error"></div>

  <div id="results">
    <div>
      <img id="overlay-img" alt="Analysis overlay">
    </div>
    <div class="scores">
      <div class="score-card">
        <div class="label">MEBeauty Scale</div>
        <div class="value" id="val-10">&mdash;</div>
        <div class="scale">out of 10</div>
      </div>
      <div class="score-card">
        <div class="label">SCUT Scale</div>
        <div class="value" id="val-5">&mdash;</div>
        <div class="scale">out of 5</div>
      </div>
      <div class="score-card">
        <div class="label">Z-Score</div>
        <div class="value" id="val-z">&mdash;</div>
        <div class="scale">standard deviations from mean</div>
      </div>
      <div class="retry"><button onclick="reset()">Try another photo</button></div>
    </div>
  </div>
</main>

<script>
const dropzone = document.getElementById('dropzone');
const fileinput = document.getElementById('fileinput');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const errorDiv = document.getElementById('error');

['dragenter','dragover'].forEach(e =>
  dropzone.addEventListener(e, ev => { ev.preventDefault(); dropzone.classList.add('dragover'); }));
['dragleave','drop'].forEach(e =>
  dropzone.addEventListener(e, ev => { ev.preventDefault(); dropzone.classList.remove('dragover'); }));

dropzone.addEventListener('drop', ev => {
  const file = ev.dataTransfer.files[0];
  if (file) upload(file);
});

fileinput.addEventListener('change', () => {
  if (fileinput.files[0]) upload(fileinput.files[0]);
});

function reset() {
  results.classList.remove('active');
  errorDiv.classList.remove('active');
  dropzone.style.display = 'flex';
  fileinput.value = '';
}

async function upload(file) {
  dropzone.style.display = 'none';
  results.classList.remove('active');
  errorDiv.classList.remove('active');
  loading.classList.add('active');

  const form = new FormData();
  form.append('image', file);

  try {
    const resp = await fetch('/analyze', { method: 'POST', body: form });
    const data = await resp.json();

    loading.classList.remove('active');

    if (data.error) {
      errorDiv.textContent = data.error;
      errorDiv.classList.add('active');
      dropzone.style.display = 'flex';
      return;
    }

    document.getElementById('overlay-img').src = 'data:image/jpeg;base64,' + data.overlay;
    document.getElementById('val-10').textContent = data.score_10;
    document.getElementById('val-5').textContent = data.score_5;
    document.getElementById('val-z').textContent = (data.z >= 0 ? '+' : '') + data.z;
    results.classList.add('active');
  } catch (err) {
    loading.classList.remove('active');
    errorDiv.textContent = 'Request failed: ' + err.message;
    errorDiv.classList.add('active');
    dropzone.style.display = 'flex';
  }
}
</script>

</body>
</html>
"""


if __name__ == "__main__":
    print(f"\n  Open http://localhost:{args.port} in your browser\n")
    app.run(debug=False, port=args.port)
