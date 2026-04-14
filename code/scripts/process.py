"""
Extract facial beauty markers from MEBeauty dataset images using MediaPipe Face Mesh.

Outputs:
  - data/features.csv          tabular beauty markers + score + metadata
  - data/debug/*.jpg            landmark overlay images for visual verification
"""

import math
import random
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEBEAUTY_ROOT = PROJECT_ROOT / "MEBeauty-database-main"
SCUT_ROOT = PROJECT_ROOT / "SCUT-FBP5500_v2"
OUTPUT_CSV = PROJECT_ROOT / "data" / "features.csv"
DEBUG_DIR = PROJECT_ROOT / "data" / "debug"
MODEL_PATH = PROJECT_ROOT / "data" / "face_landmarker_v2_with_blendshapes.task"

NUM_DEBUG_IMAGES = 20
DEBUG_SIZE = 512  # upscale debug images to this resolution

# ---------------------------------------------------------------------------
# MediaPipe landmark indices (Face Mesh 478: 468 face + 10 iris)
# ---------------------------------------------------------------------------
# Eyes – outer / inner corners
L_EYE_OUTER = 33
L_EYE_INNER = 133
R_EYE_INNER = 362
R_EYE_OUTER = 263

# Eyes – top / bottom
L_EYE_TOP = 159
L_EYE_BOTTOM = 145
R_EYE_TOP = 386
R_EYE_BOTTOM = 374

# Eyebrows
L_BROW_INNER = 55
L_BROW_OUTER = 46
L_BROW_TOP = 105   # highest point of left brow arch
R_BROW_INNER = 285
R_BROW_OUTER = 276
R_BROW_TOP = 334   # highest point of right brow arch
L_BROW = 70
R_BROW = 300

# Iris (refine_landmarks gives 468-477)
L_IRIS_CENTER = 468
R_IRIS_CENTER = 473
L_IRIS_BOTTOM = 472
R_IRIS_BOTTOM = 477

# Nose
NOSE_TIP = 1
NOSE_BRIDGE_TOP = 6
NOSE_BOTTOM = 2
NOSE_LEFT = 129
NOSE_RIGHT = 358

# Lips
LIP_LEFT = 61
LIP_RIGHT = 291
UPPER_LIP_TOP = 13
UPPER_LIP_BOTTOM = 14
LOWER_LIP_TOP = 14
LOWER_LIP_BOTTOM = 17
CUPID_BOW_LEFT = 82
CUPID_BOW_RIGHT = 312

# Face contour
FACE_TOP = 10
FACE_BOTTOM = 152
FACE_LEFT = 234
FACE_RIGHT = 454

# Jaw & chin
JAW_LEFT = 172
JAW_RIGHT = 397
CHIN_CENTER = 152
# Jaw angle points (along the face oval for gonial angle)
JAW_ANGLE_LEFT_ABOVE = 162   # above left jaw corner
JAW_ANGLE_LEFT_BELOW = 150   # below left jaw corner
JAW_ANGLE_RIGHT_ABOVE = 389  # above right jaw corner
JAW_ANGLE_RIGHT_BELOW = 379  # below right jaw corner
# Chin width points (narrower than jaw)
CHIN_LEFT = 175
CHIN_RIGHT = 399

# Symmetry pairs
SYMMETRY_PAIRS = [
    (33, 263), (133, 362),
    (70, 300),
    (129, 358),
    (61, 291),
    (234, 454),
    (172, 397),
    (159, 386),
    (145, 374),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def lm_xy(landmarks, idx, w, h):
    """Return (x, y) pixel coordinates for a landmark index."""
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])


def dist(a, b):
    """Euclidean distance between two 2D points."""
    return np.linalg.norm(a - b)


def angle_deg(a, b):
    """Signed angle in degrees of vector a->b relative to horizontal."""
    d = b - a
    return math.degrees(math.atan2(d[1], d[0]))


def angle_at_vertex(a, vertex, b):
    """Angle in degrees at vertex point, formed by rays vertex->a and vertex->b."""
    va = a - vertex
    vb = b - vertex
    cos_angle = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def compute_head_roll(landmarks, w, h):
    """Estimate head roll from the line connecting left and right eye centers."""
    l_center = (lm_xy(landmarks, L_EYE_OUTER, w, h) + lm_xy(landmarks, L_EYE_INNER, w, h)) / 2
    r_center = (lm_xy(landmarks, R_EYE_INNER, w, h) + lm_xy(landmarks, R_EYE_OUTER, w, h)) / 2
    return angle_deg(l_center, r_center)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------
def compute_features(landmarks, w, h):
    """Compute beauty marker features from MediaPipe landmarks."""
    lm = landmarks
    head_roll = compute_head_roll(lm, w, h)

    # Key reference points & distances
    face_left = lm_xy(lm, FACE_LEFT, w, h)
    face_right = lm_xy(lm, FACE_RIGHT, w, h)
    face_top = lm_xy(lm, FACE_TOP, w, h)
    face_bottom = lm_xy(lm, FACE_BOTTOM, w, h)
    face_width = dist(face_left, face_right)
    face_height = dist(face_top, face_bottom)

    if face_width < 1 or face_height < 1:
        return None

    face_area = face_width * face_height  # bounding box approx

    # Eye corners
    l_outer = lm_xy(lm, L_EYE_OUTER, w, h)
    l_inner = lm_xy(lm, L_EYE_INNER, w, h)
    r_inner = lm_xy(lm, R_EYE_INNER, w, h)
    r_outer = lm_xy(lm, R_EYE_OUTER, w, h)

    # Eye dimensions
    l_eye_w = dist(l_outer, l_inner)
    r_eye_w = dist(r_outer, r_inner)
    avg_eye_w = (l_eye_w + r_eye_w) / 2

    l_eye_top = lm_xy(lm, L_EYE_TOP, w, h)
    l_eye_bot = lm_xy(lm, L_EYE_BOTTOM, w, h)
    r_eye_top = lm_xy(lm, R_EYE_TOP, w, h)
    r_eye_bot = lm_xy(lm, R_EYE_BOTTOM, w, h)

    l_eye_h = dist(l_eye_top, l_eye_bot)
    r_eye_h = dist(r_eye_top, r_eye_bot)
    avg_eye_h = (l_eye_h + r_eye_h) / 2

    # Nose points
    nose_left = lm_xy(lm, NOSE_LEFT, w, h)
    nose_right = lm_xy(lm, NOSE_RIGHT, w, h)
    nose_top = lm_xy(lm, NOSE_BRIDGE_TOP, w, h)
    nose_bottom = lm_xy(lm, NOSE_BOTTOM, w, h)
    nose_tip = lm_xy(lm, NOSE_TIP, w, h)

    # Lip points
    lip_left = lm_xy(lm, LIP_LEFT, w, h)
    lip_right = lm_xy(lm, LIP_RIGHT, w, h)
    upper_lip_top = lm_xy(lm, UPPER_LIP_TOP, w, h)
    upper_lip_bot = lm_xy(lm, UPPER_LIP_BOTTOM, w, h)
    lower_lip_bot = lm_xy(lm, LOWER_LIP_BOTTOM, w, h)

    # Jaw points
    jaw_left = lm_xy(lm, JAW_LEFT, w, h)
    jaw_right = lm_xy(lm, JAW_RIGHT, w, h)
    jaw_w = dist(jaw_left, jaw_right)

    # Midline
    midline_x = (face_left[0] + face_right[0]) / 2

    # ======================================================================
    # EXISTING FEATURES (refined)
    # ======================================================================

    # --- Canthal tilt (head-roll corrected) ---
    def eye_tilt(outer, inner):
        dx = inner[0] - outer[0]
        dy = inner[1] - outer[1]
        if dx < 0:
            dx, dy = -dx, -dy
        return math.degrees(math.atan2(-dy, dx))

    l_canthal_raw = eye_tilt(l_outer, l_inner)
    r_canthal_raw = eye_tilt(r_outer, r_inner)
    canthal_tilt = ((l_canthal_raw - head_roll) + (r_canthal_raw - head_roll)) / 2

    # --- Eye ratios ---
    eye_width_ratio = avg_eye_w / face_width
    eye_height_ratio = avg_eye_h / avg_eye_w if avg_eye_w > 0 else 0

    # --- Eyebrow-to-eye distance ---
    l_brow_dist = dist(lm_xy(lm, L_BROW, w, h), l_eye_top)
    r_brow_dist = dist(lm_xy(lm, R_BROW, w, h), r_eye_top)
    eyebrow_eye_dist = ((l_brow_dist + r_brow_dist) / 2) / face_height

    # --- Nose ---
    nose_width_ratio = dist(nose_left, nose_right) / face_width
    nose_length_ratio = dist(nose_top, nose_bottom) / face_height

    # --- Lips ---
    lip_width = dist(lip_left, lip_right)
    lip_width_ratio = lip_width / face_width
    upper_lip_h = dist(upper_lip_top, upper_lip_bot)
    lower_lip_h = dist(upper_lip_bot, lower_lip_bot)
    upper_lip_ratio = upper_lip_h / lower_lip_h if lower_lip_h > 0 else 0

    # --- Facial symmetry (aggregate) ---
    asym_scores = []
    for l_idx, r_idx in SYMMETRY_PAIRS:
        l_pt = lm_xy(lm, l_idx, w, h)
        r_pt = lm_xy(lm, r_idx, w, h)
        l_dist_to_mid = abs(l_pt[0] - midline_x)
        r_dist_to_mid = abs(r_pt[0] - midline_x)
        if max(l_dist_to_mid, r_dist_to_mid) > 0:
            asym_scores.append(abs(l_dist_to_mid - r_dist_to_mid) / face_width)
    facial_symmetry = np.mean(asym_scores) if asym_scores else 0

    # --- Face proportions ---
    face_length_width_ratio = face_height / face_width

    eye_center_y = (l_outer[1] + r_outer[1]) / 2
    midface_length = abs(nose_bottom[1] - eye_center_y)
    midface_ratio = midface_length / face_height

    lower_face_length = abs(face_bottom[1] - nose_bottom[1])
    lower_face_ratio = lower_face_length / face_height

    # --- Jaw and cheekbones ---
    jaw_width_ratio = jaw_w / face_width
    cheekbone_prominence = face_width / jaw_w if jaw_w > 0 else 0

    # --- Interpupillary distance ---
    try:
        l_iris = lm_xy(lm, L_IRIS_CENTER, w, h)
        r_iris = lm_xy(lm, R_IRIS_CENTER, w, h)
        interpupillary_ratio = dist(l_iris, r_iris) / face_width
    except IndexError:
        l_center = (l_outer + l_inner) / 2
        r_center = (r_outer + r_inner) / 2
        interpupillary_ratio = dist(l_center, r_center) / face_width

    # ======================================================================
    # NEW FEATURES
    # ======================================================================

    # --- Eye spacing (rule of fifths: intercanthal = 1 eye width) ---
    intercanthal = dist(l_inner, r_inner)
    eye_spacing_ratio = intercanthal / face_width

    # --- Eye area ratio (ellipse approximation: pi * a * b) ---
    l_eye_area = math.pi * (l_eye_w / 2) * (l_eye_h / 2)
    r_eye_area = math.pi * (r_eye_w / 2) * (r_eye_h / 2)
    eye_area_ratio = ((l_eye_area + r_eye_area) / 2) / face_area

    # --- Scleral show (iris bottom to lower eyelid gap / eye height) ---
    try:
        l_iris_bot = lm_xy(lm, L_IRIS_BOTTOM, w, h)
        r_iris_bot = lm_xy(lm, R_IRIS_BOTTOM, w, h)
        # Positive = white showing below iris (bad), negative = iris overlaps lid
        l_scleral = (l_eye_bot[1] - l_iris_bot[1]) / l_eye_h if l_eye_h > 0 else 0
        r_scleral = (r_eye_bot[1] - r_iris_bot[1]) / r_eye_h if r_eye_h > 0 else 0
        scleral_show = (l_scleral + r_scleral) / 2
    except IndexError:
        scleral_show = 0.0

    # --- Eye asymmetry ---
    eye_asymmetry = abs(l_eye_h - r_eye_h) / avg_eye_h if avg_eye_h > 0 else 0

    # --- Brow arch height (highest brow point above eye / face height) ---
    l_brow_top = lm_xy(lm, L_BROW_TOP, w, h)
    r_brow_top = lm_xy(lm, R_BROW_TOP, w, h)
    l_brow_arch = abs(l_eye_top[1] - l_brow_top[1])
    r_brow_arch = abs(r_eye_top[1] - r_brow_top[1])
    brow_arch_height = ((l_brow_arch + r_brow_arch) / 2) / face_height

    # --- Lip fullness (total lip height / face height) ---
    total_lip_h = dist(upper_lip_top, lower_lip_bot)
    lip_fullness_ratio = total_lip_h / face_height

    # --- Mouth width to interocular distance ratio ---
    l_eye_center = (l_outer + l_inner) / 2
    r_eye_center = (r_outer + r_inner) / 2
    interocular = dist(l_eye_center, r_eye_center)
    mouth_width_face_ratio = lip_width / interocular if interocular > 0 else 0

    # --- Cupid's bow ratio (bow width / lip width) ---
    cupid_left = lm_xy(lm, CUPID_BOW_LEFT, w, h)
    cupid_right = lm_xy(lm, CUPID_BOW_RIGHT, w, h)
    cupids_bow_ratio = dist(cupid_left, cupid_right) / lip_width if lip_width > 0 else 0

    # --- Mouth-to-chin ratio ---
    mouth_center = (lip_left + lip_right) / 2
    mouth_chin_dist = abs(face_bottom[1] - mouth_center[1])
    mouth_chin_ratio = mouth_chin_dist / lower_face_length if lower_face_length > 0 else 0

    # --- Gonial angle (jaw sharpness) ---
    jaw_above_l = lm_xy(lm, JAW_ANGLE_LEFT_ABOVE, w, h)
    jaw_below_l = lm_xy(lm, JAW_ANGLE_LEFT_BELOW, w, h)
    jaw_above_r = lm_xy(lm, JAW_ANGLE_RIGHT_ABOVE, w, h)
    jaw_below_r = lm_xy(lm, JAW_ANGLE_RIGHT_BELOW, w, h)
    l_gonial = angle_at_vertex(jaw_above_l, jaw_left, jaw_below_l)
    r_gonial = angle_at_vertex(jaw_above_r, jaw_right, jaw_below_r)
    gonial_angle = (l_gonial + r_gonial) / 2

    # --- Chin taper (chin width / jaw width) ---
    chin_l = lm_xy(lm, CHIN_LEFT, w, h)
    chin_r = lm_xy(lm, CHIN_RIGHT, w, h)
    chin_w = dist(chin_l, chin_r)
    chin_taper = chin_w / jaw_w if jaw_w > 0 else 0

    # --- Face taper (jaw width / cheekbone width, inverse of cheekbone_prominence) ---
    face_taper_ratio = jaw_w / face_width

    # --- Upper face ratio (forehead: top to brows / face height) ---
    brow_center_y = (lm_xy(lm, L_BROW, w, h)[1] + lm_xy(lm, R_BROW, w, h)[1]) / 2
    upper_face_length = abs(brow_center_y - face_top[1])
    upper_face_ratio = upper_face_length / face_height

    # --- Phi (golden ratio) deviation ---
    phi = 1.618
    phi_deviation = abs(face_length_width_ratio - phi)

    # --- Facial thirds balance (std of the three thirds — lower = more balanced) ---
    upper_third = upper_face_length / face_height
    mid_third = midface_ratio
    lower_third = lower_face_ratio
    facial_thirds_symmetry = np.std([upper_third, mid_third, lower_third])

    # --- Decomposed symmetry ---
    # Eye symmetry
    l_eye_dist_to_mid = abs((l_outer[0] + l_inner[0]) / 2 - midline_x)
    r_eye_dist_to_mid = abs((r_outer[0] + r_inner[0]) / 2 - midline_x)
    eye_symmetry = abs(l_eye_dist_to_mid - r_eye_dist_to_mid) / face_width

    # Mouth symmetry
    mouth_l_dist = abs(lip_left[0] - midline_x)
    mouth_r_dist = abs(lip_right[0] - midline_x)
    mouth_symmetry = abs(mouth_l_dist - mouth_r_dist) / face_width

    # Nose symmetry (tip deviation from midline)
    nose_symmetry = abs(nose_tip[0] - midline_x) / face_width

    return {
        # Original features
        "canthal_tilt": canthal_tilt,
        "eye_width_ratio": eye_width_ratio,
        "eye_height_ratio": eye_height_ratio,
        "eyebrow_eye_dist": eyebrow_eye_dist,
        "nose_width_ratio": nose_width_ratio,
        "nose_length_ratio": nose_length_ratio,
        "lip_width_ratio": lip_width_ratio,
        "upper_lip_ratio": upper_lip_ratio,
        "facial_symmetry": facial_symmetry,
        "face_length_width_ratio": face_length_width_ratio,
        "midface_ratio": midface_ratio,
        "lower_face_ratio": lower_face_ratio,
        "jaw_width_ratio": jaw_width_ratio,
        "cheekbone_prominence": cheekbone_prominence,
        "interpupillary_ratio": interpupillary_ratio,
        # New features
        "eye_spacing_ratio": eye_spacing_ratio,
        "eye_area_ratio": eye_area_ratio,
        "scleral_show": scleral_show,
        "eye_asymmetry": eye_asymmetry,
        "brow_arch_height": brow_arch_height,
        "lip_fullness_ratio": lip_fullness_ratio,
        "mouth_width_face_ratio": mouth_width_face_ratio,
        "cupids_bow_ratio": cupids_bow_ratio,
        "mouth_chin_ratio": mouth_chin_ratio,
        "gonial_angle": gonial_angle,
        "chin_taper": chin_taper,
        "face_taper_ratio": face_taper_ratio,
        "upper_face_ratio": upper_face_ratio,
        "phi_deviation": phi_deviation,
        "facial_thirds_symmetry": facial_thirds_symmetry,
        "eye_symmetry": eye_symmetry,
        "mouth_symmetry": mouth_symmetry,
        "nose_symmetry": nose_symmetry,
        "head_roll": head_roll,
    }


# ---------------------------------------------------------------------------
# Debug overlay drawing
# ---------------------------------------------------------------------------
# Color scheme (BGR)
C_YELLOW = (0, 255, 255)
C_CYAN = (255, 200, 0)
C_PINK = (200, 100, 255)
C_GREEN = (0, 255, 0)
C_RED = (0, 0, 255)
C_WHITE = (255, 255, 255)
C_BLUE = (255, 150, 0)
C_GRAY = (128, 128, 128)
C_MAGENTA = (255, 0, 255)
C_ORANGE = (0, 165, 255)
C_LIME = (0, 255, 128)


def draw_debug_overlay(img, landmarks, w, h, features):
    """Draw landmark overlays on an upscaled image with a legend panel."""
    # Upscale image
    scale = DEBUG_SIZE / max(w, h)
    img_up = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
    sw, sh = img_up.shape[1], img_up.shape[0]

    # Create legend panel (dark background on the right)
    legend_w = 320
    canvas = np.zeros((max(sh, 700), sw + legend_w, 3), dtype=np.uint8)
    canvas[:sh, :sw] = img_up

    def pt(idx):
        p = lm_xy(landmarks, idx, w, h) * scale
        return tuple(p.astype(int))

    def spt(a):
        """Scale a raw pixel point."""
        return tuple((a * scale).astype(int))

    # Draw all 468 landmarks as small dots
    for i in range(min(468, len(landmarks))):
        cv2.circle(canvas, pt(i), 1, (60, 60, 60), -1)

    # --- Measurement lines ---
    # Face frame (white)
    cv2.line(canvas, pt(FACE_LEFT), pt(FACE_RIGHT), C_WHITE, 1)
    cv2.line(canvas, pt(FACE_TOP), pt(FACE_BOTTOM), C_WHITE, 1)

    # Canthal tilt (yellow, thicker)
    cv2.line(canvas, pt(L_EYE_INNER), pt(L_EYE_OUTER), C_YELLOW, 2)
    cv2.line(canvas, pt(R_EYE_OUTER), pt(R_EYE_INNER), C_YELLOW, 2)

    # Eye height (cyan)
    cv2.line(canvas, pt(L_EYE_TOP), pt(L_EYE_BOTTOM), C_CYAN, 1)
    cv2.line(canvas, pt(R_EYE_TOP), pt(R_EYE_BOTTOM), C_CYAN, 1)

    # Eyebrow-eye distance (pink)
    cv2.line(canvas, pt(L_BROW), pt(L_EYE_TOP), C_PINK, 1)
    cv2.line(canvas, pt(R_BROW), pt(R_EYE_TOP), C_PINK, 1)

    # Brow arch (pink dotted)
    cv2.line(canvas, pt(L_BROW_TOP), pt(L_EYE_TOP), C_PINK, 1)
    cv2.line(canvas, pt(R_BROW_TOP), pt(R_EYE_TOP), C_PINK, 1)

    # Nose width + length (green)
    cv2.line(canvas, pt(NOSE_LEFT), pt(NOSE_RIGHT), C_GREEN, 2)
    cv2.line(canvas, pt(NOSE_BRIDGE_TOP), pt(NOSE_BOTTOM), C_GREEN, 1)

    # Lips (red)
    cv2.line(canvas, pt(LIP_LEFT), pt(LIP_RIGHT), C_RED, 2)
    cv2.line(canvas, pt(UPPER_LIP_TOP), pt(LOWER_LIP_BOTTOM), C_RED, 1)
    # Cupid's bow (orange)
    cv2.line(canvas, pt(CUPID_BOW_LEFT), pt(CUPID_BOW_RIGHT), C_ORANGE, 1)

    # Jaw (blue)
    cv2.line(canvas, pt(JAW_LEFT), pt(JAW_RIGHT), C_BLUE, 1)
    # Gonial angle lines
    cv2.line(canvas, pt(JAW_ANGLE_LEFT_ABOVE), pt(JAW_LEFT), C_BLUE, 1)
    cv2.line(canvas, pt(JAW_LEFT), pt(JAW_ANGLE_LEFT_BELOW), C_BLUE, 1)
    cv2.line(canvas, pt(JAW_ANGLE_RIGHT_ABOVE), pt(JAW_RIGHT), C_BLUE, 1)
    cv2.line(canvas, pt(JAW_RIGHT), pt(JAW_ANGLE_RIGHT_BELOW), C_BLUE, 1)

    # Chin width (lime)
    cv2.line(canvas, pt(CHIN_LEFT), pt(CHIN_RIGHT), C_LIME, 1)

    # Eye spacing (intercanthal, magenta)
    cv2.line(canvas, pt(L_EYE_INNER), pt(R_EYE_INNER), C_MAGENTA, 1)

    # Iris (magenta dots)
    try:
        cv2.circle(canvas, pt(L_IRIS_CENTER), 4, C_MAGENTA, -1)
        cv2.circle(canvas, pt(R_IRIS_CENTER), 4, C_MAGENTA, -1)
        cv2.circle(canvas, pt(L_IRIS_BOTTOM), 2, C_MAGENTA, -1)
        cv2.circle(canvas, pt(R_IRIS_BOTTOM), 2, C_MAGENTA, -1)
    except IndexError:
        pass

    # Midline (gray)
    mid_x = (pt(FACE_LEFT)[0] + pt(FACE_RIGHT)[0]) // 2
    cv2.line(canvas, (mid_x, pt(FACE_TOP)[1]), (mid_x, pt(FACE_BOTTOM)[1]), C_GRAY, 1)

    # --- Legend panel ---
    x0 = sw + 10
    y = 18
    line_h = 19

    def legend_line(label, value, color=C_WHITE, fmt=".3f"):
        nonlocal y
        text = f"{label}: {value:{fmt}}"
        cv2.putText(canvas, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        y += line_h

    def legend_header(text):
        nonlocal y
        y += 4
        cv2.putText(canvas, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1)
        y += line_h

    legend_header("-- EYES --")
    legend_line("canthal_tilt", features["canthal_tilt"], C_YELLOW)
    legend_line("eye_width_ratio", features["eye_width_ratio"], C_CYAN)
    legend_line("eye_height_ratio", features["eye_height_ratio"], C_CYAN)
    legend_line("eye_spacing", features["eye_spacing_ratio"], C_MAGENTA)
    legend_line("eye_area", features["eye_area_ratio"], C_CYAN)
    legend_line("scleral_show", features["scleral_show"], C_MAGENTA)
    legend_line("eye_asymmetry", features["eye_asymmetry"], C_CYAN)
    legend_line("brow_arch", features["brow_arch_height"], C_PINK)
    legend_line("eyebrow_eye_dist", features["eyebrow_eye_dist"], C_PINK)

    legend_header("-- NOSE --")
    legend_line("nose_width", features["nose_width_ratio"], C_GREEN)
    legend_line("nose_length", features["nose_length_ratio"], C_GREEN)
    legend_line("nose_symmetry", features["nose_symmetry"], C_GREEN)

    legend_header("-- MOUTH --")
    legend_line("lip_width", features["lip_width_ratio"], C_RED)
    legend_line("upper_lip_ratio", features["upper_lip_ratio"], C_RED)
    legend_line("lip_fullness", features["lip_fullness_ratio"], C_RED)
    legend_line("mouth_width_face", features["mouth_width_face_ratio"], C_RED)
    legend_line("cupids_bow", features["cupids_bow_ratio"], C_ORANGE)
    legend_line("mouth_chin", features["mouth_chin_ratio"], C_RED)
    legend_line("mouth_symmetry", features["mouth_symmetry"], C_RED)

    legend_header("-- JAW & FACE --")
    legend_line("jaw_width", features["jaw_width_ratio"], C_BLUE)
    legend_line("cheekbone_prom", features["cheekbone_prominence"], C_BLUE)
    legend_line("gonial_angle", features["gonial_angle"], C_BLUE, ".1f")
    legend_line("chin_taper", features["chin_taper"], C_LIME)
    legend_line("face_taper", features["face_taper_ratio"], C_BLUE)

    legend_header("-- PROPORTIONS --")
    legend_line("face_L/W", features["face_length_width_ratio"], C_WHITE)
    legend_line("upper_face", features["upper_face_ratio"], C_WHITE)
    legend_line("midface", features["midface_ratio"], C_WHITE)
    legend_line("lower_face", features["lower_face_ratio"], C_WHITE)
    legend_line("phi_deviation", features["phi_deviation"], C_WHITE)
    legend_line("thirds_balance", features["facial_thirds_symmetry"], C_WHITE)
    legend_line("symmetry", features["facial_symmetry"], C_GRAY)
    legend_line("eye_symmetry", features["eye_symmetry"], C_GRAY)
    legend_line("head_roll", features["head_roll"], C_GRAY)

    legend_header("-- EXPRESSION --")
    legend_line("smile", features.get("expr_smile", 0), C_YELLOW)
    legend_line("frown", features.get("expr_frown", 0), C_YELLOW)
    legend_line("jaw_open", features.get("expr_jaw_open", 0), C_YELLOW)
    legend_line("eye_squint", features.get("expr_eye_squint", 0), C_YELLOW)

    return canvas


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
def load_mebeauty():
    """Load MEBeauty entries: (abs_img_path, raw_score, split, dataset, gender, ethnicity)."""
    scores_dir = MEBEAUTY_ROOT / "scores"
    entries = []
    for split in ["train", "test", "val"]:
        path = scores_dir / f"{split}_2022.txt"
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(" ", 1)
                img_rel = parts[0].lstrip("./")
                score = float(parts[1])
                img_path = MEBEAUTY_ROOT / img_rel
                # Parse gender/ethnicity from path
                p = Path(img_rel).parts
                gender = p[2] if len(p) > 2 else "unknown"
                ethnicity = p[3] if len(p) > 3 else "unknown"
                entries.append((str(img_path), score, split, "mebeauty", gender, ethnicity))
    return entries


def load_scut():
    """Load SCUT-FBP5500 entries. Uses their 60/40 train/test split."""
    scut_images = SCUT_ROOT / "Images"
    split_dir = SCUT_ROOT / "train_test_files" / "split_of_60%training and 40%testing"
    entries = []

    for split, filename in [("train", "train.txt"), ("test", "test.txt")]:
        path = split_dir / filename
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(" ", 1)
                img_name = parts[0].strip()
                score = float(parts[1])
                img_path = scut_images / img_name

                # Parse gender/ethnicity from filename prefix (AF, AM, CF, CM)
                prefix = img_name[:2]
                gender_map = {"AF": "female", "AM": "male", "CF": "female", "CM": "male"}
                ethnicity_map = {"AF": "asian", "AM": "asian", "CF": "caucasian", "CM": "caucasian"}
                gender = gender_map.get(prefix, "unknown")
                ethnicity = ethnicity_map.get(prefix, "unknown")

                entries.append((str(img_path), score, split, "scut", gender, ethnicity))
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Load both datasets
    mebeauty_entries = load_mebeauty()
    scut_entries = load_scut()
    all_entries = mebeauty_entries + scut_entries

    print(f"MEBeauty entries: {len(mebeauty_entries)}")
    print(f"SCUT entries:     {len(scut_entries)}")
    print(f"Total entries:    {len(all_entries)}")

    debug_indices = set(random.sample(range(len(all_entries)), min(NUM_DEBUG_IMAGES, len(all_entries))))
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Init MediaPipe
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        num_faces=1,
        output_face_blendshapes=True,
    )
    landmarker = FaceLandmarker.create_from_options(options)

    rows = []
    skipped_missing = 0
    skipped_no_face = 0
    debug_saved = 0

    pbar = tqdm(all_entries, desc="Processing faces", unit="img")
    for i, (img_path_str, score, split, dataset, gender, ethnicity) in enumerate(pbar):
        img_path = Path(img_path_str)
        if not img_path.exists():
            skipped_missing += 1
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            skipped_missing += 1
            continue

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = landmarker.detect(mp_image)

        if not results.face_landmarks:
            skipped_no_face += 1
            pbar.set_postfix(ok=len(rows), skip=skipped_no_face + skipped_missing)
            continue

        landmarks = results.face_landmarks[0]
        features = compute_features(landmarks, w, h)
        if features is None:
            skipped_no_face += 1
            continue

        # Extract expression blendshapes
        if results.face_blendshapes:
            bs_map = {bs.category_name: bs.score for bs in results.face_blendshapes[0]}
            features["expr_smile"] = (bs_map.get("mouthSmileLeft", 0) + bs_map.get("mouthSmileRight", 0)) / 2
            features["expr_frown"] = (bs_map.get("mouthFrownLeft", 0) + bs_map.get("mouthFrownRight", 0)) / 2
            features["expr_jaw_open"] = bs_map.get("jawOpen", 0)
            features["expr_brow_up"] = bs_map.get("browInnerUp", 0)
            features["expr_brow_down"] = (bs_map.get("browDownLeft", 0) + bs_map.get("browDownRight", 0)) / 2
            features["expr_cheek_squint"] = (bs_map.get("cheekSquintLeft", 0) + bs_map.get("cheekSquintRight", 0)) / 2
            features["expr_eye_squint"] = (bs_map.get("eyeSquintLeft", 0) + bs_map.get("eyeSquintRight", 0)) / 2
            features["expr_eye_wide"] = (bs_map.get("eyeWideLeft", 0) + bs_map.get("eyeWideRight", 0)) / 2
            features["expr_mouth_pucker"] = bs_map.get("mouthPucker", 0)
        else:
            for k in ["expr_smile", "expr_frown", "expr_jaw_open", "expr_brow_up",
                       "expr_brow_down", "expr_cheek_squint", "expr_eye_squint",
                       "expr_eye_wide", "expr_mouth_pucker"]:
                features[k] = 0.0

        row = {
            "image_path": str(img_path),
            "dataset": dataset,
            "gender": gender,
            "ethnicity": ethnicity,
            "score_raw": score,
            "split": split,
            **features,
        }
        rows.append(row)

        if i in debug_indices:
            overlay = draw_debug_overlay(img_bgr, landmarks, w, h, features)
            debug_name = f"{debug_saved:02d}_{img_path.stem}.jpg"
            cv2.imwrite(str(DEBUG_DIR / debug_name), overlay)
            debug_saved += 1

        pbar.set_postfix(ok=len(rows), skip=skipped_no_face + skipped_missing)

    landmarker.close()

    df = pd.DataFrame(rows)

    # Z-score normalize scores per dataset so they're on the same scale
    print("\nScore normalization (z-score per dataset):")
    for ds in df["dataset"].unique():
        mask = df["dataset"] == ds
        raw = df.loc[mask, "score_raw"]
        print(f"  {ds}: mean={raw.mean():.2f}, std={raw.std():.2f}, n={mask.sum()}")
        df.loc[mask, "score"] = (raw - raw.mean()) / raw.std()

    print(f"  Combined normalized: mean={df['score'].mean():.3f}, std={df['score'].std():.3f}")

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDone!")
    print(f"  Processed:       {len(rows)}")
    print(f"  Skipped missing: {skipped_missing}")
    print(f"  Skipped no face: {skipped_no_face}")
    print(f"  Debug overlays:  {debug_saved} saved to {DEBUG_DIR}")
    print(f"  Output CSV:      {OUTPUT_CSV}")
    print(f"  Features:        {len([c for c in df.columns if c not in ['image_path','gender','ethnicity','score','split']])}")


if __name__ == "__main__":
    main()
