import time
import json
import shutil
import base64

import cv2
import numpy as np
from ultralytics import YOLO
from website.face_profiles import (
    BASE_DIR as PROJECT_DIR,
    get_face_capture_temp_dir,
    get_face_profile_dir,
    list_saved_face_images,
)
from face_features import (
    _get_clahe, _compute_lbp, _align_face, _prepare_face,
    _extract_face_features, augment_face, FEATURE_DIM, _GRID, _HIST_BINS,
)


BASE_DIR = PROJECT_DIR
FACES_DIR = BASE_DIR / "faces"
TRAINER_FILE = BASE_DIR / "trainer.npz"
LABEL_FILE = BASE_DIR / "face_labels.txt"
MODEL_CONFIG_FILE = BASE_DIR / "face_model.json"
MATCH_THRESHOLD = 0.82

DEFAULT_MODEL = "yolov10n"

AVAILABLE_MODELS = {
    "yolov8n": {
        "file": "yolov8n.pt",
        "label": "YOLOv8 Nano",
        "description": "Stable, widely supported, fast",
        "badge": "Stable",
    },
    "yolov10n": {
        "file": "yolov10n.pt",
        "label": "YOLOv10 Nano",
        "description": "NMS-free detection, improved efficiency",
        "badge": "Default",
    },
    "yolo11n": {
        "file": "yolo11n.pt",
        "label": "YOLO11 Nano",
        "description": "Latest generation, best accuracy",
        "badge": "Latest",
    },
}

FACES_DIR.mkdir(exist_ok=True)

_model = None
_loaded_model_name = None
_active_model_name = None
_face_detector = None
_eye_detector = None
_recognizer = None
_label_map = {}


# ── model config ──────────────────────────────────────────────────────────────

def get_active_model_name():
    global _active_model_name
    if _active_model_name is None:
        if MODEL_CONFIG_FILE.exists():
            try:
                cfg = json.loads(MODEL_CONFIG_FILE.read_text(encoding="utf-8"))
                name = cfg.get("model", DEFAULT_MODEL)
                _active_model_name = name if name in AVAILABLE_MODELS else DEFAULT_MODEL
            except Exception:
                _active_model_name = DEFAULT_MODEL
        else:
            _active_model_name = DEFAULT_MODEL
    return _active_model_name


def set_active_model_name(name):
    global _model, _loaded_model_name, _active_model_name
    if name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(AVAILABLE_MODELS)}")
    MODEL_CONFIG_FILE.write_text(json.dumps({"model": name}), encoding="utf-8")
    _active_model_name = name
    _model = None
    _loaded_model_name = None


def get_model_status():
    """Return info dict for all available models (for admin UI)."""
    active = get_active_model_name()
    status = {}
    for name, info in AVAILABLE_MODELS.items():
        path = BASE_DIR / info["file"]
        size_mb = round(path.stat().st_size / (1024 * 1024), 1) if path.exists() else None
        status[name] = {
            **info,
            "name": name,
            "active": name == active,
            "downloaded": path.exists(),
            "size_mb": size_mb,
        }
    return status


def _saved_face_data_exists():
    for person_path in FACES_DIR.iterdir():
        if person_path.is_dir() and any(image_path.is_file() for image_path in person_path.iterdir()):
            return True

    return False


def _decode_base64_frame(frame_data):
    if not frame_data or "," not in frame_data:
        raise ValueError("Invalid frame data.")

    encoded_image = frame_data.split(",", 1)[1]
    image_bytes = base64.b64decode(encoded_image)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Could not decode frame data.")

    return frame


def _load_model():
    global _model, _loaded_model_name

    model_name = get_active_model_name()
    if _model is None or _loaded_model_name != model_name:
        model_path = BASE_DIR / AVAILABLE_MODELS[model_name]["file"]
        _model = YOLO(str(model_path))
        _loaded_model_name = model_name

    return _model


def _load_face_detector():
    global _face_detector

    if _face_detector is None:
        _face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    return _face_detector


def _load_eye_detector():
    global _eye_detector

    if _eye_detector is None:
        _eye_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        )

    return _eye_detector


def _load_label_map(force_reload=False):
    global _label_map

    if _label_map and not force_reload:
        return _label_map

    _label_map = {}

    if LABEL_FILE.exists():
        with LABEL_FILE.open("r", encoding="utf-8") as label_file:
            for line in label_file:
                label, identifier = line.strip().split(",")
                _label_map[int(identifier)] = label

    return _label_map


def _current_face_person_names():
    """Return a sorted tuple of person directory names (excluding temp dirs)."""
    _SKIP_SUFFIXES = ("_temp", "_capture_temp")
    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    names = []
    for p in FACES_DIR.iterdir():
        if not p.is_dir():
            continue
        if any(p.name.endswith(s) for s in _SKIP_SUFFIXES):
            continue
        has_images = any(
            f.is_file() and f.suffix.lower() in _IMG_EXTS
            for f in p.iterdir()
        )
        if has_images:
            names.append(p.name)
    return tuple(sorted(names))


def _load_recognizer(force_reload=False):
    global _recognizer

    if _recognizer is None or force_reload:
        _load_label_map(force_reload=force_reload)

        if not TRAINER_FILE.exists() and _saved_face_data_exists():
            train_faces()

        if TRAINER_FILE.exists():
            training_data = np.load(str(TRAINER_FILE), allow_pickle=False)
            features = training_data["features"]

            # Auto-retrain when feature dimensions changed (extractor upgrade)
            if features.size > 0 and features.shape[1] != FEATURE_DIM:
                print(f"[face] Feature dim changed ({features.shape[1]} → {FEATURE_DIM}), retraining...")
                TRAINER_FILE.unlink()
                train_faces()
                return _recognizer

            # Auto-retrain when new people were added (or people removed)
            label_map = _load_label_map()
            trained_names = tuple(sorted(label_map.values()))
            current_names = _current_face_person_names()
            if trained_names != current_names:
                print(f"[face] Face roster changed {trained_names} → {current_names}, retraining...")
                TRAINER_FILE.unlink()
                train_faces()
                return _recognizer

            _recognizer = {
                "features": features,
                "labels": training_data["labels"]
            }
        else:
            _recognizer = {
                "features": np.empty((0, 0), dtype=np.float32),
                "labels": np.empty((0,), dtype=np.int32)
            }

    return _recognizer


def faces_ready():
    label_map = _load_label_map(force_reload=True)
    recognizer = _load_recognizer(force_reload=True)
    return bool(label_map) and recognizer["features"].size > 0


def _predict_face(face_img):
    recognizer = _load_recognizer()
    label_map = _load_label_map()

    if recognizer["features"].size == 0 or not label_map:
        return "Unknown", 0.0

    query_feature = _extract_face_features(face_img, _load_eye_detector())
    similarities = recognizer["features"] @ query_feature
    labels = recognizer["labels"]

    scores = {}
    for label_id in np.unique(labels):
        label_scores = similarities[labels == label_id]
        top_count = min(5, len(label_scores))
        scores[int(label_id)] = float(np.mean(np.sort(label_scores)[-top_count:]))

    if not scores:
        return "Unknown", 0.0

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_label_id, best_similarity = sorted_scores[0]

    # When multiple identities are enrolled, require a clear margin over 2nd best
    if len(sorted_scores) >= 2:
        second_similarity = sorted_scores[1][1]
        if best_similarity - second_similarity < 0.05:
            return "Unknown", max(0.0, best_similarity) * 100

    if best_label_id is not None and best_similarity >= MATCH_THRESHOLD and best_label_id in label_map:
        return label_map[best_label_id], max(0.0, best_similarity) * 100

    return "Unknown", max(0.0, best_similarity) * 100


def train_faces():
    global _recognizer

    start_time = time.time()
    features = []
    labels = []
    label_ids = {}
    current_id = 0

    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for person_path in sorted(FACES_DIR.iterdir()):
        if not person_path.is_dir():
            continue

        # Skip temporary capture directories
        if person_path.name.endswith("_temp") or person_path.name.endswith("_capture_temp"):
            continue

        if person_path.name not in label_ids:
            label_ids[person_path.name] = current_id
            current_id += 1

        label_id = label_ids[person_path.name]
        loaded = 0

        for image_path in person_path.iterdir():
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in _IMG_EXTS:
                continue

            gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            if gray is None or gray.size == 0:
                continue

            features.append(_extract_face_features(gray, _load_eye_detector()))
            labels.append(label_id)
            loaded += 1

        if loaded == 0:
            # Person directory has no usable images — remove from label map
            del label_ids[person_path.name]
            current_id -= 1

    if not features:
        if TRAINER_FILE.exists():
            TRAINER_FILE.unlink()
        if LABEL_FILE.exists():
            LABEL_FILE.unlink()
        _recognizer = {
            "features": np.empty((0, 0), dtype=np.float32),
            "labels": np.empty((0,), dtype=np.int32)
        }
        return

    feature_array = np.stack(features).astype(np.float32)
    label_array = np.array(labels, dtype=np.int32)
    np.savez_compressed(str(TRAINER_FILE), features=feature_array, labels=label_array)

    with LABEL_FILE.open("w", encoding="utf-8") as label_file:
        for label_name, label_id in label_ids.items():
            label_file.write(f"{label_name},{label_id}\n")

    _recognizer = {
        "features": feature_array,
        "labels": label_array
    }

    _load_label_map(force_reload=True)

    training_time = time.time() - start_time
    n_people = len(label_ids)
    n_samples = len(features)
    print(f"Training complete: {n_people} people, {n_samples} samples in {training_time:.2f}s")


def _detect_face(frame):
    model = _load_model()
    face_detector = _load_face_detector()
    eye_detector = _load_eye_detector()
    frame_height, frame_width = frame.shape[:2]

    for result in model(frame, verbose=False):
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            if int(cls) != 0:
                continue

            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            person = frame[y1:y2, x1:x2]

            if person.size == 0:
                continue

            gray = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))

            for fx, fy, fw, fh in faces:
                face_img = gray[fy:fy + fh, fx:fx + fw]
                eyes = eye_detector.detectMultiScale(face_img, 1.1, 4)
                cv2.rectangle(person, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

                frame[y1:y2, x1:x2] = person
                bbox = {
                    "x": float((x1 + fx) / frame_width),
                    "y": float((y1 + fy) / frame_height),
                    "w": float(fw / frame_width),
                    "h": float(fh / frame_height)
                }
                return frame, face_img, len(eyes), bbox

    return frame, None, 0, None


def _scan_frame(frame):
    annotated_frame, face_img, _, _ = _detect_face(frame)
    detected_name = "Unknown"
    confidence_text = "Unknown: N/A"

    if face_img is not None:
        detected_name, confidence = _predict_face(face_img)
        confidence_text = f"{detected_name}: {confidence:.1f}%"

    cv2.putText(
        annotated_frame,
        confidence_text,
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    return annotated_frame, detected_name


def _save_face_sample(face_dir, stage_name, sample_index, face_img):
    original_path = face_dir / f"{stage_name}_{sample_index:02d}.jpg"
    cv2.imwrite(str(original_path), face_img)

    for augment_index, augmented_face in enumerate(augment_face(face_img), start=1):
        augmented_path = face_dir / f"{stage_name}_{sample_index:02d}_aug_{augment_index:02d}.jpg"
        cv2.imwrite(str(augmented_path), augmented_face)


def _count_stage_samples(face_dir, stage_name):
    if not face_dir.exists():
        return 0

    return len([
        image_path for image_path in face_dir.iterdir()
        if image_path.is_file()
        and image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and image_path.stem.startswith(f"{stage_name}_")
        and "_aug_" not in image_path.stem
    ])


def _all_training_stages_complete(face_dir, stages):
    return all(_count_stage_samples(face_dir, stage_name) >= required for stage_name, required in stages.items())


def extract_face_from_frame_data(frame_data):
    frame = _decode_base64_frame(frame_data)
    _, face_img, eye_count, _ = _detect_face(frame)
    return face_img, eye_count


def analyze_face_frame(frame_data):
    frame = _decode_base64_frame(frame_data)
    _, face_img, eye_count, bbox = _detect_face(frame)

    if face_img is None or bbox is None:
        return {
            "ok": True,
            "detected": False,
            "message": "No face detected."
        }

    detected_name = "Unknown"
    confidence = 0.0

    if faces_ready():
        detected_name, confidence = _predict_face(face_img)

    return {
        "ok": True,
        "detected": True,
        "bbox": bbox,
        "recognized_name": detected_name,
        "confidence": confidence,
        "eye_count": eye_count,
        "message": f"{detected_name}: {confidence:.1f}%"
    }


def _fallback_face_from_frame(frame):
    """Extract a center-cropped grayscale region as fallback when detection fails."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    h, w = gray.shape[:2]
    crop_size = min(h, w)
    y_start = (h - crop_size) // 2
    x_start = (w - crop_size) // 2
    cropped = gray[y_start:y_start + crop_size, x_start:x_start + crop_size]
    return cv2.resize(cropped, (96, 96))


def capture_training_frame(profile_name, stage_name, frame_data):
    stages = {
        "front": 5,
        "left": 4,
        "right": 4,
        "blink": 3,
    }

    if stage_name not in stages:
        return {
            "ok": False,
            "message": "Unknown training step."
        }

    # Try face detection but fall back to center-crop if not detected
    face_img, eye_count = extract_face_from_frame_data(frame_data)

    if face_img is None:
        frame = _decode_base64_frame(frame_data)
        face_img = _fallback_face_from_frame(frame)

    temp_dir = get_face_capture_temp_dir(profile_name)
    temp_dir.mkdir(parents=True, exist_ok=True)

    next_index = _count_stage_samples(temp_dir, stage_name) + 1
    _save_face_sample(temp_dir, stage_name, next_index, face_img)

    completed = next_index >= stages[stage_name]
    training_complete = _all_training_stages_complete(temp_dir, stages)

    if training_complete:
        face_dir = get_face_profile_dir(profile_name)

        if face_dir.exists():
            shutil.rmtree(face_dir)

        temp_dir.replace(face_dir)
        train_faces()
        saved_count = len(list_saved_face_images(profile_name))

        return {
            "ok": True,
            "captured": next_index,
            "required": stages[stage_name],
            "stage_complete": True,
            "training_complete": True,
            "saved_count": saved_count,
            "message": f"Saved {saved_count} face samples and retrained your usual-order profile."
        }

    return {
        "ok": True,
        "captured": next_index,
        "required": stages[stage_name],
        "stage_complete": completed,
        "training_complete": False,
        "message": f"Captured {next_index} of {stages[stage_name]} for {stage_name}."
    }


def reset_training_capture(profile_name):
    temp_dir = get_face_capture_temp_dir(profile_name)

    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def recognize_face_from_frame_data(frame_data, expected_name=None):
    model_file = BASE_DIR / AVAILABLE_MODELS[get_active_model_name()]["file"]
    if not model_file.exists():
        return {
            "ok": False,
            "message": f"YOLO model not found: {model_file.name}."
        }

    if not faces_ready():
        return {
            "ok": False,
            "message": "No trained face profiles are available yet."
        }

    face_img, _ = extract_face_from_frame_data(frame_data)

    if face_img is None:
        return {
            "ok": False,
            "message": "No face detected. Position your face in the frame and try again."
        }

    detected_name, confidence = _predict_face(face_img)

    if detected_name == "Unknown":
        return {
            "ok": False,
            "message": f"No known face was recognized. Confidence: {confidence:.1f}%."
        }

    if expected_name and detected_name.strip().lower() != expected_name.strip().lower():
        return {
            "ok": False,
            "message": f"Detected {detected_name}, which does not match {expected_name}."
        }

    return {
        "ok": True,
        "recognized_name": detected_name,
        "confidence": confidence,
        "message": f"Confirmed {detected_name}."
    }


from face_camera import capture_face_training, confirm_known_face  # noqa: E402, F401


if __name__ == "__main__":
    confirmed_name, status_message = confirm_known_face()
    print(status_message)
    if confirmed_name:
        print(f"Detected user: {confirmed_name}")
