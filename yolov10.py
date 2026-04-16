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

_GRID = 4
_HIST_BINS = 64
FEATURE_DIM = _GRID * _GRID * _HIST_BINS  # 1024

_model = None
_loaded_model_name = None
_active_model_name = None
_face_detector = None
_eye_detector = None
_recognizer = None
_label_map = {}
_clahe = None


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


def _get_clahe():
    global _clahe
    if _clahe is None:
        _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _clahe


def _compute_lbp(gray):
    """Compute LBP code image using vectorised 8-neighbor circular pattern."""
    h, w = gray.shape
    p = np.pad(gray.astype(np.int32), 1, mode='edge')
    center = p[1:h + 1, 1:w + 1]
    neighbors = [
        p[0:h,     0:w],      # (-1,-1)
        p[0:h,     1:w + 1],  # (-1, 0)
        p[0:h,     2:w + 2],  # (-1,+1)
        p[1:h + 1, 2:w + 2],  # ( 0,+1)
        p[2:h + 2, 2:w + 2],  # (+1,+1)
        p[2:h + 2, 1:w + 1],  # (+1, 0)
        p[2:h + 2, 0:w],      # (+1,-1)
        p[1:h + 1, 0:w],      # ( 0,-1)
    ]
    lbp = np.zeros((h, w), dtype=np.uint8)
    for k, n in enumerate(neighbors):
        lbp += (n >= center).astype(np.uint8) * (1 << k)
    return lbp


def _align_face(gray_face):
    """Rotate face image so detected eyes are level; skip if alignment is ambiguous."""
    eye_detector = _load_eye_detector()
    h, w = gray_face.shape[:2]

    # Only search upper half of face for eyes
    upper = gray_face[:h // 2, :]
    eyes = eye_detector.detectMultiScale(upper, 1.05, 3, minSize=(8, 8))

    if len(eyes) < 2:
        return gray_face

    eyes = sorted(eyes, key=lambda e: e[0])[:2]
    (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes[0], eyes[1]
    cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
    cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2

    # Reject if eyes are at very different heights or implausible separation
    if abs(cy1 - cy2) > h * 0.15:
        return gray_face
    separation = abs(cx2 - cx1)
    if separation < w * 0.15 or separation > w * 0.70:
        return gray_face

    angle = np.degrees(np.arctan2(cy2 - cy1, cx2 - cx1))
    if abs(angle) > 20:
        return gray_face

    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray_face, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _prepare_face(face):
    resized = cv2.resize(face, (96, 96))
    aligned = _align_face(resized)
    enhanced = _get_clahe().apply(aligned)
    return enhanced


def _extract_face_features(face):
    """Extract grid-based LBP histogram feature vector from a grayscale face crop."""
    prepared = _prepare_face(face)
    lbp = _compute_lbp(prepared)

    cell_h = prepared.shape[0] // _GRID
    cell_w = prepared.shape[1] // _GRID
    features = []
    for gy in range(_GRID):
        for gx in range(_GRID):
            cell = lbp[gy * cell_h:(gy + 1) * cell_h, gx * cell_w:(gx + 1) * cell_w]
            hist = cv2.calcHist([cell], [0], None, [_HIST_BINS], [0, 256])
            hist = hist.flatten().astype(np.float32)
            s = float(hist.sum())
            if s > 0:
                hist /= s
            features.append(hist)

    feature = np.concatenate(features).astype(np.float32)
    norm = float(np.linalg.norm(feature))
    if norm > 0:
        feature /= norm
    return feature


def _predict_face(face_img):
    recognizer = _load_recognizer()
    label_map = _load_label_map()

    if recognizer["features"].size == 0 or not label_map:
        return "Unknown", 0.0

    query_feature = _extract_face_features(face_img)
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


def augment_face(face):
    augmented = []

    h, w = face.shape

    low_brightness = cv2.convertScaleAbs(face, alpha=0.7, beta=-30)
    high_brightness = cv2.convertScaleAbs(face, alpha=1.3, beta=30)
    zoom_in = cv2.resize(face[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)], (w, h))

    zoom_out = cv2.resize(face, None, fx=0.8, fy=0.8)
    zoom_out = cv2.copyMakeBorder(
        zoom_out,
        (h - zoom_out.shape[0]) // 2,
        (h - zoom_out.shape[0]) // 2,
        (w - zoom_out.shape[1]) // 2,
        (w - zoom_out.shape[1]) // 2,
        cv2.BORDER_CONSTANT
    )

    move_left = cv2.warpAffine(face, np.float32([[1, 0, -10], [0, 1, 0]]), (w, h))
    move_right = cv2.warpAffine(face, np.float32([[1, 0, 10], [0, 1, 0]]), (w, h))
    move_up = cv2.warpAffine(face, np.float32([[1, 0, 0], [0, 1, -10]]), (w, h))
    move_down = cv2.warpAffine(face, np.float32([[1, 0, 0], [0, 1, 10]]), (w, h))

    rot_r = cv2.warpAffine(face, cv2.getRotationMatrix2D((w // 2, h // 2), 10, 1), (w, h))
    rot_l = cv2.warpAffine(face, cv2.getRotationMatrix2D((w // 2, h // 2), -10, 1), (w, h))
    flip = cv2.flip(face, 1)

    augmented.extend([
        low_brightness,
        high_brightness,
        zoom_in,
        zoom_out,
        move_left,
        move_right,
        move_up,
        move_down,
        rot_r,
        rot_l,
        flip
    ])

    return augmented


def train_faces():
    global _recognizer

    start_time = time.time()
    features = []
    labels = []
    label_ids = {}
    current_id = 0

    for person_path in FACES_DIR.iterdir():
        if not person_path.is_dir():
            continue

        if person_path.name not in label_ids:
            label_ids[person_path.name] = current_id
            current_id += 1

        label_id = label_ids[person_path.name]

        for image_path in person_path.iterdir():
            gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            if gray is None:
                continue

            features.append(_extract_face_features(gray))
            labels.append(label_id)

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
    print("Training complete")
    print(f"Training time: {training_time:.2f} seconds")


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
    if not MODEL_PATH.exists():
        return {
            "ok": False,
            "message": f"YOLO model not found at {MODEL_PATH}."
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


def capture_face_training(profile_name, camera_index=0):
    if not MODEL_PATH.exists():
        return 0, f"YOLO model not found at {MODEL_PATH}."

    _load_model()
    _load_face_detector()
    _load_eye_detector()

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        return 0, "Could not access the kiosk camera."

    face_dir = get_face_profile_dir(profile_name)
    temp_dir = face_dir.parent / f"{face_dir.name}_capture_temp"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir(parents=True, exist_ok=True)

    stages = [
        {"name": "front", "label": "Look straight", "captures": 5, "blink_required": False},
        {"name": "left", "label": "Face left", "captures": 4, "blink_required": False},
        {"name": "right", "label": "Face right", "captures": 4, "blink_required": False},
        {"name": "blink", "label": "Blink", "captures": 3, "blink_required": True},
    ]

    stage_index = 0
    stage_started = False
    stage_capture_count = 0
    total_capture_count = 0
    last_capture_time = 0.0
    window_name = "Wideye Face Training"

    try:
        while stage_index < len(stages):
            ret, frame = cap.read()

            if not ret:
                shutil.rmtree(temp_dir, ignore_errors=True)
                return 0, "Could not read a frame from the kiosk camera."

            stage = stages[stage_index]
            annotated_frame, face_img, eye_count, _ = _detect_face(frame)

            cv2.putText(
                annotated_frame,
                f"Step {stage_index + 1}/{len(stages)}: {stage['label']}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            cv2.putText(
                annotated_frame,
                "Press Enter to start this step. Press ESC to cancel.",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )
            cv2.putText(
                annotated_frame,
                f"Captured {stage_capture_count}/{stage['captures']} for this step",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            if stage["blink_required"]:
                blink_text = "Blink detected" if eye_count == 0 and face_img is not None else "Blink and keep both eyes closed briefly"
                cv2.putText(
                    annotated_frame,
                    blink_text,
                    (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0) if "detected" in blink_text else (0, 191, 255),
                    2
                )

            if stage_started and face_img is not None and time.time() - last_capture_time >= 0.7:
                if not stage["blink_required"] or eye_count == 0:
                    stage_capture_count += 1
                    total_capture_count += 1
                    _save_face_sample(temp_dir, stage["name"], stage_capture_count, face_img)
                    last_capture_time = time.time()

                    if stage_capture_count >= stage["captures"]:
                        stage_index += 1
                        stage_started = False
                        stage_capture_count = 0
                        last_capture_time = 0.0

            cv2.imshow(window_name, annotated_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (10, 13):
                stage_started = True

            if key == 27:
                shutil.rmtree(temp_dir, ignore_errors=True)
                return 0, "Face training was canceled."

        if face_dir.exists():
            shutil.rmtree(face_dir)

        temp_dir.replace(face_dir)
        train_faces()
        original_count = len(list_saved_face_images(profile_name))
        return original_count, f"Saved {original_count} face samples and retrained your usual-order profile."
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def confirm_known_face(expected_name=None, camera_index=0):
    if not MODEL_PATH.exists():
        return None, f"YOLO model not found at {MODEL_PATH}."

    if not faces_ready():
        return None, "No trained face profiles are available yet."

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        return None, "Could not access the kiosk camera."

    window_name = "Wideye Usual Order"
    expected_label = expected_name.strip().lower() if expected_name else None

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                return None, "Could not read a frame from the kiosk camera."

            annotated_frame, detected_name = _scan_frame(frame)

            cv2.putText(
                annotated_frame,
                "Press Y to confirm your face or ESC to cancel",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            if expected_name:
                cv2.putText(
                    annotated_frame,
                    f"Expected profile: {expected_name}",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

            if detected_name != "Unknown":
                cv2.putText(
                    annotated_frame,
                    f"Recognized: {detected_name}",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            cv2.imshow(window_name, annotated_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("y"):
                if detected_name == "Unknown":
                    return None, "No known face was recognized to confirm."

                if expected_label and detected_name.lower() != expected_label:
                    return None, f"Detected {detected_name}, which does not match {expected_name}."

                return detected_name, f"Confirmed {detected_name}."

            if key == 27:
                return None, "Facial confirmation was canceled."
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    confirmed_name, status_message = confirm_known_face()
    print(status_message)
    if confirmed_name:
        print(f"Detected user: {confirmed_name}")
