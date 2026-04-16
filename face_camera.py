"""
face_camera.py — OpenCV window / camera-based face capture and confirmation.

Provides: capture_face_training, confirm_known_face
"""

import time
import shutil

import cv2

from website.face_profiles import (
    get_face_profile_dir,
    get_face_capture_temp_dir,
    list_saved_face_images,
)


def _model_file_exists():
    from yolov10 import BASE_DIR, AVAILABLE_MODELS, get_active_model_name
    return (BASE_DIR / AVAILABLE_MODELS[get_active_model_name()]["file"]).exists()


def _draw_text(frame, text, y, color=(255, 255, 255), scale=0.7):
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def _annotate_capture_frame(
    frame, stage, stage_index, total_stages, stage_capture_count, eye_count, face_img
):
    _draw_text(
        frame,
        f"Step {stage_index + 1}/{total_stages}: {stage['label']}",
        30,
        scale=0.8,
    )
    _draw_text(
        frame,
        "Press Enter to start this step. Press ESC to cancel.",
        65,
        scale=0.65,
    )
    _draw_text(
        frame,
        f"Captured {stage_capture_count}/{stage['captures']} for this step",
        100,
        color=(0, 255, 0),
        scale=0.65,
    )

    if stage["blink_required"]:
        blink_detected = eye_count == 0 and face_img is not None
        blink_text = (
            "Blink detected"
            if blink_detected
            else "Blink and keep both eyes closed briefly"
        )
        _draw_text(
            frame,
            blink_text,
            135,
            color=(0, 255, 0) if blink_detected else (0, 191, 255),
            scale=0.65,
        )

    return frame


def _run_capture_stage(cap, stage, stage_index, total_stages, temp_dir):
    """Run one stage's capture loop. Returns capture count or raises on ESC/error."""
    from yolov10 import _detect_face, _save_face_sample

    stage_started = False
    stage_capture_count = 0
    last_capture_time = 0.0
    window_name = "Wideye Face Training"

    while stage_capture_count < stage["captures"]:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read a frame from the kiosk camera.")

        annotated_frame, face_img, eye_count, _ = _detect_face(frame)
        _annotate_capture_frame(
            annotated_frame, stage, stage_index, total_stages,
            stage_capture_count, eye_count, face_img,
        )

        if (
            stage_started
            and face_img is not None
            and time.time() - last_capture_time >= 0.7
        ):
            if not stage["blink_required"] or eye_count == 0:
                stage_capture_count += 1
                _save_face_sample(temp_dir, stage["name"], stage_capture_count, face_img)
                last_capture_time = time.time()

        cv2.imshow(window_name, annotated_frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (10, 13):
            stage_started = True

        if key == 27:
            raise KeyboardInterrupt("Face training was canceled.")

    return stage_capture_count


def capture_face_training(profile_name, camera_index=0):
    from yolov10 import AVAILABLE_MODELS, get_active_model_name, _load_model, train_faces

    if not _model_file_exists():
        model_file = AVAILABLE_MODELS[get_active_model_name()]["file"]
        return 0, f"YOLO model not found: {model_file}."

    _load_model()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return 0, "Could not access the kiosk camera."

    face_dir = get_face_profile_dir(profile_name)
    temp_dir = get_face_capture_temp_dir(profile_name)

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    stages = [
        {"name": "front", "label": "Look straight", "captures": 5, "blink_required": False},
        {"name": "left",  "label": "Face left",     "captures": 4, "blink_required": False},
        {"name": "right", "label": "Face right",    "captures": 4, "blink_required": False},
        {"name": "blink", "label": "Blink",         "captures": 3, "blink_required": True},
    ]

    try:
        for stage_index, stage in enumerate(stages):
            _run_capture_stage(cap, stage, stage_index, len(stages), temp_dir)

        if face_dir.exists():
            shutil.rmtree(face_dir)
        temp_dir.replace(face_dir)
        train_faces()
        original_count = len(list_saved_face_images(profile_name))
        return (
            original_count,
            f"Saved {original_count} face samples and retrained your usual-order profile.",
        )
    except KeyboardInterrupt:
        return 0, "Face training was canceled."
    except RuntimeError as exc:
        return 0, str(exc)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def confirm_known_face(expected_name=None, camera_index=0):
    from yolov10 import faces_ready, _scan_frame

    if not _model_file_exists():
        from yolov10 import AVAILABLE_MODELS, get_active_model_name
        model_file = AVAILABLE_MODELS[get_active_model_name()]["file"]
        return None, f"YOLO model not found: {model_file}."

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

            _draw_text(
                annotated_frame,
                "Press Y to confirm your face or ESC to cancel",
                30,
            )

            if expected_name:
                _draw_text(
                    annotated_frame,
                    f"Expected profile: {expected_name}",
                    65,
                    scale=0.65,
                )

            if detected_name != "Unknown":
                _draw_text(
                    annotated_frame,
                    f"Recognized: {detected_name}",
                    100,
                    color=(0, 255, 0),
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
