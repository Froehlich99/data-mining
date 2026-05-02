"""Download all datasets and models required to run the pipeline."""

import subprocess
import urllib.request
import zipfile
from pathlib import Path

import gdown

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

# MEBeauty — git clone pinned to a specific commit
MEBEAUTY_REPO = "https://github.com/fbplab/MEBeauty-database.git"
MEBEAUTY_COMMIT = "7b849562ee92d99d34d56afa2ebe85e5075f9b20"
MEBEAUTY_DIR = "MEBeauty-database-main"

# SCUT-FBP5500 — hosted on Google Drive (not available via git)
SCUT_GDRIVE_ID = "1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf"
SCUT_DIR = "SCUT-FBP5500_v2"

# MediaPipe face landmarker model (required by process.py)
MEDIAPIPE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MEDIAPIPE_MODEL_PATH = (
    PROJECT_ROOT / "data" / "face_landmarker_v2_with_blendshapes.task"
)


def download_mebeauty():
    dest = DATASETS_DIR / MEBEAUTY_DIR
    if dest.exists():
        print(f"Skipping {MEBEAUTY_DIR} (already exists)")
        return

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {MEBEAUTY_DIR} ...")
    subprocess.run(
        ["git", "clone", "--single-branch", MEBEAUTY_REPO, str(dest)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(dest), "checkout", MEBEAUTY_COMMIT],
        check=True,
    )
    print(f"  Pinned to {MEBEAUTY_COMMIT[:12]}")


def download_scut():
    dest = DATASETS_DIR / SCUT_DIR
    if dest.exists():
        print(f"Skipping {SCUT_DIR} (already exists)")
        return

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATASETS_DIR / "scut-fbp5500.zip"
    print(f"Downloading {SCUT_DIR} from Google Drive ...")
    gdown.download(id=SCUT_GDRIVE_ID, output=str(zip_path), quiet=False)

    print("Extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATASETS_DIR)

    # The zip may extract to a differently-named folder — rename if needed
    extracted = None
    for candidate in DATASETS_DIR.iterdir():
        if (
            candidate.is_dir()
            and "SCUT-FBP5500" in candidate.name
            and candidate.name != SCUT_DIR
        ):
            extracted = candidate
            break

    if extracted:
        extracted.rename(dest)
        print(f"  Renamed {extracted.name} -> {SCUT_DIR}")
    elif not dest.exists():
        print(f"  WARNING: expected {SCUT_DIR} after extraction but not found.")
        print(f"  Check the contents of {zip_path} and place them manually.")
        return

    zip_path.unlink()
    print(f"  Done ({SCUT_DIR})")


def download_mediapipe_model():
    if MEDIAPIPE_MODEL_PATH.exists():
        print(f"Skipping {MEDIAPIPE_MODEL_PATH.name} (already exists)")
        return

    MEDIAPIPE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MEDIAPIPE_MODEL_PATH.name} ...")
    urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, MEDIAPIPE_MODEL_PATH)
    print(f"  Saved to {MEDIAPIPE_MODEL_PATH.relative_to(PROJECT_ROOT)}")


def main():
    download_mebeauty()
    download_scut()
    download_mediapipe_model()
    print("Done.")


if __name__ == "__main__":
    main()
