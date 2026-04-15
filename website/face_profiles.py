from pathlib import Path
import re


BASE_DIR = Path(__file__).resolve().parent.parent
FACES_DIR = BASE_DIR / "faces"
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def normalize_face_profile_name(value):
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", (value or "").strip().lower()).strip("-")
    return normalized or "customer-face"


def get_face_profile_dir(profile_name):
    FACES_DIR.mkdir(exist_ok=True)
    return FACES_DIR / normalize_face_profile_name(profile_name)


def get_face_capture_temp_dir(profile_name):
    face_dir = get_face_profile_dir(profile_name)
    return face_dir.parent / f"{face_dir.name}_capture_temp"


def list_saved_face_images(profile_name):
    face_dir = get_face_profile_dir(profile_name)

    if not face_dir.exists():
        return []

    images = []

    for image_path in sorted(face_dir.iterdir()):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            continue

        if "_aug_" in image_path.stem:
            continue

        images.append(image_path.name)

    return images
