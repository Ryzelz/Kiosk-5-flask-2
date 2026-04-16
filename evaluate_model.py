"""
evaluate_model.py  –  Face-recognition model evaluation report

Metrics
-------
1. Safety          → IoU Loss         (1 - mean IoU of Haar detections across augmented crops)
2. Performance     → Inference Time   (mean ± std ms per feature-extraction call)
3. Manufacturability → Training Time  (seconds to run train_faces() from scratch)
4. Maintainability → Maintainability Index  (Radon MI score for yolov10.py, 0-100)
5. Reliability     → Model Consistency  (recognition accuracy on held-out augmented faces)

Usage
-----
    python evaluate_model.py

Dependencies already in requirements.txt: opencv-python, numpy
Optional (auto-detected): radon  →  pip install radon
"""

import os
import sys
import time
import shutil
import statistics
import subprocess
import textwrap
from pathlib import Path

import cv2
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
FACES_DIR = ROOT / "faces"
TRAINER_FILE = ROOT / "trainer.npz"
LABEL_FILE = ROOT / "face_labels.txt"
YOLO_FILE = ROOT / "yolov10.py"

# ── helpers ───────────────────────────────────────────────────────────────────

def _section(title):
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def _result(label, value, note=""):
    padding = 38 - len(label)
    line = f"  {label}{'.' * max(1, padding)} {value}"
    if note:
        line += f"  ({note})"
    print(line)


def _rating(value, thresholds, labels, reverse=False):
    """Return a rating label.  thresholds = ascending list of break-points.
    reverse=False (lower-is-better): value <= t[0] → labels[0], ..., else labels[-1]
    reverse=True  (higher-is-better): value >= t[-1] → labels[-1], ..., else labels[0]
    """
    if not reverse:
        for threshold, label in zip(thresholds, labels[:-1]):
            if value <= threshold:
                return label
        return labels[-1]
    else:
        # Walk thresholds from highest to lowest
        for threshold, label in zip(reversed(thresholds), reversed(labels[1:])):
            if value >= threshold:
                return label
        return labels[0]


# ── 1. Safety: IoU Loss ───────────────────────────────────────────────────────

def _iou(a, b):
    """Compute IoU for two (x, y, w, h) rectangles."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx)
    iy = max(ay, by)
    iw = max(0, min(ax + aw, bx + bw) - ix)
    ih = max(0, min(ay + ah, by + bh) - iy)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _embed_in_canvas(face_gray, canvas_size=160):
    """Place a face crop centred in a larger gray canvas."""
    # Squeeze trailing channel dim if present (H, W, 1) → (H, W)
    if face_gray.ndim == 3:
        face_gray = face_gray[:, :, 0]
    canvas = np.full((canvas_size, canvas_size), 128, dtype=np.uint8)
    h, w = face_gray.shape[:2]
    y0 = (canvas_size - h) // 2
    x0 = (canvas_size - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = face_gray
    return canvas, x0, y0


def _haar_detect(img):
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = detector.detectMultiScale(img, 1.1, 4, minSize=(20, 20))
    return list(faces) if len(faces) else []


def _augment_for_iou(img):
    """Mild augmentations that should not break detection."""
    h, w = img.shape[:2]
    variants = [
        cv2.convertScaleAbs(img, alpha=0.85, beta=0),       # darker
        cv2.convertScaleAbs(img, alpha=1.15, beta=0),       # brighter
        cv2.GaussianBlur(img, (3, 3), 0),                   # slight blur
        cv2.warpAffine(img,
                       np.float32([[1, 0, 3], [0, 1, 0]]),
                       (w, h)),                              # tiny shift right
        cv2.warpAffine(img,
                       np.float32([[1, 0, -3], [0, 1, 0]]),
                       (w, h)),                              # tiny shift left
    ]
    return variants


def measure_iou_loss():
    """Average IoU loss over front-facing raw images."""
    front_images = [
        p for p in (FACES_DIR / "seb").iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and p.stem.startswith("front_")
        and "_aug_" not in p.stem
    ] if (FACES_DIR / "seb").exists() else []

    if not front_images:
        return None, "no face images found"

    all_ious = []
    for img_path in front_images:
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        canvas, _, _ = _embed_in_canvas(gray)
        ref_boxes = _haar_detect(canvas)
        if not ref_boxes:
            continue
        ref_box = ref_boxes[0]

        for aug in _augment_for_iou(canvas):
            aug_boxes = _haar_detect(aug)
            if aug_boxes:
                all_ious.append(_iou(ref_box, aug_boxes[0]))
            # no detection on augmented = IoU 0 (counts against score)
            else:
                all_ious.append(0.0)

    if not all_ious:
        return None, "Haar detected no faces in training crops"

    mean_iou = statistics.mean(all_ious)
    iou_loss = 1.0 - mean_iou
    return iou_loss, f"mean IoU {mean_iou:.4f} over {len(all_ious)} augmented pairs"


# ── 2. Performance: Inference Time ────────────────────────────────────────────

def _load_feature_extractor():
    """Import internal feature extractor without triggering Flask app."""
    sys.path.insert(0, str(ROOT))
    # Import only the functions we need, not the whole module init
    import importlib.util
    spec = importlib.util.spec_from_file_location("yolov10", YOLO_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def measure_inference_time(mod, n_runs=50):
    """Time _extract_face_features() on saved face images. mod can be yolov10 module."""
    # Find any person directory with face images
    person_dir = None
    if FACES_DIR.exists():
        for d in sorted(FACES_DIR.iterdir()):
            if d.is_dir() and not d.name.startswith("_"):
                person_dir = d
                break
    images = [
        p for p in person_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and "_aug_" not in p.stem
    ] if person_dir and person_dir.exists() else []

    if not images:
        return None, None, "no face images found"

    grays = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
             for p in images[:10] if cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) is not None]
    if not grays:
        return None, None, "could not load images"

    # _extract_face_features now requires an eye_detector parameter after refactor
    eye_detector = mod._load_eye_detector() if hasattr(mod, "_load_eye_detector") else None

    times = []
    for _ in range(n_runs):
        img = grays[_ % len(grays)]
        t0 = time.perf_counter()
        if eye_detector is not None:
            mod._extract_face_features(img, eye_detector)
        else:
            mod._extract_face_features(img)
        times.append((time.perf_counter() - t0) * 1000)

    return statistics.mean(times), statistics.stdev(times), f"{n_runs} runs"


# ── 3. Manufacturability: Training Time ───────────────────────────────────────

def measure_training_time(mod):
    """Delete trainer.npz, re-run train_faces(), measure wall time. mod can be yolov10 module."""
    backup = None
    if TRAINER_FILE.exists():
        backup = TRAINER_FILE.with_suffix(".npz.bak")
        shutil.copy2(TRAINER_FILE, backup)
    label_backup = None
    if LABEL_FILE.exists():
        label_backup = LABEL_FILE.with_suffix(".txt.bak")
        shutil.copy2(LABEL_FILE, label_backup)

    if TRAINER_FILE.exists():
        TRAINER_FILE.unlink()

    t0 = time.perf_counter()
    mod.train_faces()
    elapsed = time.perf_counter() - t0

    # Restore backups so nothing is lost
    if backup and backup.exists():
        shutil.move(str(backup), str(TRAINER_FILE))
    if label_backup and label_backup.exists():
        shutil.move(str(label_backup), str(LABEL_FILE))

    return elapsed


# ── 4. Maintainability: Maintainability Index ─────────────────────────────────

def measure_maintainability():
    """Score yolov10.py, face_features.py, and face_camera.py; return weighted average."""
    target_files = [
        ROOT / "yolov10.py",
        ROOT / "face_features.py",
        ROOT / "face_camera.py",
    ]
    existing = [f for f in target_files if f.exists()]
    if not existing:
        return 0.0, "no source files found"

    scores = []

    try:
        import radon.metrics as rm
        for f in existing:
            source = f.read_text(encoding="utf-8")
            scores.append(rm.mi_visit(source, multi=True))
        avg = sum(scores) / len(scores)
        names = ", ".join(f.name for f in existing)
        return round(avg, 2), f"radon MI avg over {names}"
    except ImportError:
        pass

    # Fallback: radon subprocess for each file
    try:
        for f in existing:
            result = subprocess.run(
                [sys.executable, "-m", "radon", "mi", "-s", str(f)],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "(" in line and ")" in line:
                        score_str = line.split("(")[-1].rstrip(")")
                        try:
                            scores.append(float(score_str))
                        except ValueError:
                            pass
        if scores:
            avg = sum(scores) / len(scores)
            return round(avg, 2), "radon MI (subprocess)"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Final fallback: AST approximation across all files
    import ast
    for f in existing:
        source = f.read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        total_lines = len(source.splitlines())
        avg_func_len = total_lines / max(len(functions), 1)
        mi_approx = max(0, min(100, 171 - 5.2 * np.log(max(1, avg_func_len))
                               - 0.23 * len(functions) - 16.2 * np.log(max(1, total_lines))))
        scores.append(mi_approx)
    avg = sum(scores) / len(scores)
    return round(float(avg), 2), "estimated MI (radon not installed)"


# ── 5. Reliability: Model Consistency ────────────────────────────────────────

def measure_reliability(mod):
    """Predict on held-out augmented faces for all enrolled persons, report accuracy."""
    if not FACES_DIR.exists():
        return None, None, "no faces directory"

    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    _SKIP_SUFFIXES = ("_temp", "_capture_temp")

    # Collect (expected_name, image_path) for augmented images across all persons
    test_items = []
    for person_path in sorted(FACES_DIR.iterdir()):
        if not person_path.is_dir():
            continue
        if any(person_path.name.endswith(s) for s in _SKIP_SUFFIXES):
            continue
        aug_images = [
            p for p in person_path.iterdir()
            if p.is_file()
            and p.suffix.lower() in _IMG_EXTS
            and "_aug_" in p.stem
        ]
        for img_path in aug_images:
            test_items.append((person_path.name.lower(), img_path))

    if not test_items:
        return None, None, "no augmented images found"

    if not mod.faces_ready():
        return None, None, "model not trained"

    correct = 0
    confidences = []

    for expected_name, img_path in test_items:
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        name, conf = mod._predict_face(gray)
        confidences.append(conf)
        if name.lower() == expected_name:
            correct += 1

    total = len(test_items)
    accuracy = correct / total if total else 0.0
    conf_std = statistics.stdev(confidences) if len(confidences) > 1 else 0.0

    return accuracy, conf_std, f"{correct}/{total} aug images correctly recognised"


# ── Report ────────────────────────────────────────────────────────────────────

def run_evaluation(model_name=None):
    """
    Run all 5 metrics and return a structured dict suitable for JSON serialisation.
    If model_name is given, temporarily activates it for YOLO-specific tests.
    Pipeline metrics (feature extraction, training, maintainability) are model-independent.
    """
    import yolov10 as face_module

    # Temporarily switch model if requested
    original_model = face_module.get_active_model_name()
    switched = False
    if model_name and model_name != original_model:
        try:
            face_module.set_active_model_name(model_name)
            switched = True
        except ValueError as e:
            return {"ok": False, "error": str(e)}

    try:
        results = {}

        # 1. Safety – IoU Loss
        iou_loss, iou_note = measure_iou_loss()
        if iou_loss is not None:
            rating = _rating(iou_loss, [0.10, 0.25, 0.40], ["Excellent", "Good", "Fair", "Poor"])
            score = max(0, round((1 - iou_loss) * 100, 1))
        else:
            rating = "N/A"
            score = None
        results["safety"] = {
            "title": "Safety",
            "subtitle": "IoU Loss",
            "value": round(iou_loss, 4) if iou_loss is not None else None,
            "display": f"{iou_loss:.4f}" if iou_loss is not None else "N/A",
            "score": score,
            "rating": rating,
            "note": iou_note,
            "description": "Detection stability across brightness/blur augmentations. Lower = safer.",
            "thresholds": "Excellent <0.10 · Good <0.25 · Fair <0.40",
        }

        # 2. Performance – Inference Time (feature extraction pipeline)
        mean_t, std_t, perf_note = measure_inference_time(face_module, n_runs=50)
        if mean_t is not None:
            rating = _rating(mean_t, [5, 15, 40], ["Excellent", "Good", "Fair", "Poor"])
            score = max(0, round(max(0, 100 - mean_t * 2), 1))
        else:
            rating = "N/A"
            score = None
        results["performance"] = {
            "title": "Performance",
            "subtitle": "Inference Time",
            "value": round(mean_t, 3) if mean_t is not None else None,
            "display": f"{mean_t:.3f} ms ±{std_t:.3f}" if mean_t is not None else "N/A",
            "score": score,
            "rating": rating,
            "note": perf_note,
            "description": "Mean feature-extraction latency per face crop.",
            "thresholds": "Excellent <5ms · Good <15ms · Fair <40ms",
        }

        # 3. Manufacturability – Training Time
        elapsed = measure_training_time(face_module)
        rating = _rating(elapsed, [5, 20, 60], ["Excellent", "Good", "Fair", "Poor"])
        score = max(0, round(max(0, 100 - elapsed * 1.5), 1))
        results["manufacturability"] = {
            "title": "Manufacturability",
            "subtitle": "Training Time",
            "value": round(elapsed, 2),
            "display": f"{elapsed:.2f} s",
            "score": score,
            "rating": rating,
            "note": "full retrain from saved face images",
            "description": "Time to retrain the recognition model from scratch.",
            "thresholds": "Excellent <5s · Good <20s · Fair <60s",
        }

        # 4. Maintainability – MI Score
        mi, mi_note = measure_maintainability()
        rating = _rating(mi, [65, 85, 95], ["Poor", "Fair", "Good", "Excellent"], reverse=True)
        score = round(mi, 1)
        results["maintainability"] = {
            "title": "Maintainability",
            "subtitle": "Maintainability Index",
            "value": round(mi, 2),
            "display": f"{mi:.2f} / 100",
            "score": score,
            "rating": rating,
            "note": mi_note,
            "description": "Code quality score across yolov10.py, face_features.py, face_camera.py.",
            "thresholds": "Excellent ≥95 · Good ≥85 · Fair ≥65",
        }

        # 5. Reliability – Recognition Accuracy
        accuracy, conf_std, rel_note = measure_reliability(face_module)
        if accuracy is not None:
            rating = _rating(accuracy, [0.70, 0.85, 0.95],
                             ["Poor", "Fair", "Good", "Excellent"], reverse=True)
            score = round(accuracy * 100, 1)
        else:
            rating = "N/A"
            score = None
        results["reliability"] = {
            "title": "Reliability",
            "subtitle": "Model Consistency",
            "value": round(accuracy * 100, 1) if accuracy is not None else None,
            "display": f"{accuracy * 100:.1f}%" if accuracy is not None else "N/A",
            "score": score,
            "rating": rating,
            "note": rel_note,
            "description": "Recognition accuracy on held-out augmented face crops.",
            "thresholds": "Excellent ≥95% · Good ≥85% · Fair ≥70%",
        }

        active_model = model_name or original_model
        return {
            "ok": True,
            "model_name": active_model,
            "metrics": results,
        }

    finally:
        if switched:
            face_module.set_active_model_name(original_model)


def main():
    print("\n" + "═" * 60)
    print("  FACE RECOGNITION MODEL EVALUATION REPORT")
    print("═" * 60)

    # Load module once
    print("\n  Loading model module…", end="", flush=True)
    mod = _load_feature_extractor()
    print(" done")

    # ── 1. Safety ────────────────────────────────────────────────────────────
    _section("1. SAFETY  ·  IoU Loss")
    print("  Measures detection stability: Haar face detection is run on")
    print("  each training crop and its augmented variants. IoU Loss = 1 - mean_IoU.")
    print()
    iou_loss, note = measure_iou_loss()
    if iou_loss is None:
        _result("IoU Loss", "N/A", note)
        _result("Rating", "N/A")
    else:
        rating = _rating(iou_loss, [0.10, 0.25, 0.40], ["Excellent", "Good", "Fair", "Poor"])
        _result("IoU Loss", f"{iou_loss:.4f}", note)
        _result("Rating", rating,
                "Excellent<0.10  Good<0.25  Fair<0.40  Poor≥0.40")

    # ── 2. Performance ────────────────────────────────────────────────────────
    _section("2. PERFORMANCE  ·  Inference Time")
    print("  Measures feature-extraction latency per face crop.")
    print()
    mean_t, std_t, note = measure_inference_time(mod)
    if mean_t is None:
        _result("Inference Time", "N/A", note)
    else:
        rating = _rating(mean_t, [5, 15, 40], ["Excellent", "Good", "Fair", "Poor"])
        _result("Mean Inference Time", f"{mean_t:.3f} ms", note)
        _result("Std Dev", f"{std_t:.3f} ms")
        _result("Rating", rating, "Excellent<5ms  Good<15ms  Fair<40ms  Poor≥40ms")

    # ── 3. Manufacturability ─────────────────────────────────────────────────
    _section("3. MANUFACTURABILITY  ·  Training Time")
    print("  Time to retrain the full model from saved face images.")
    print()
    elapsed = measure_training_time(mod)
    rating = _rating(elapsed, [5, 20, 60], ["Excellent", "Good", "Fair", "Poor"])
    _result("Training Time", f"{elapsed:.2f} s")
    _result("Rating", rating, "Excellent<5s  Good<20s  Fair<60s  Poor≥60s")

    # ── 4. Maintainability ────────────────────────────────────────────────────
    _section("4. MAINTAINABILITY  ·  Maintainability Index")
    print("  Radon MI score for yolov10.py (0–100, higher = more maintainable).")
    print()
    mi, mi_note = measure_maintainability()
    rating = _rating(mi, [65, 85, 95], ["Poor", "Fair", "Good", "Excellent"], reverse=True)
    _result("Maintainability Index", f"{mi:.2f}", mi_note)
    _result("Rating", rating, "Excellent≥95  Good≥85  Fair≥65  Poor<65")

    # ── 5. Reliability ────────────────────────────────────────────────────────
    _section("5. RELIABILITY  ·  Model Consistency")
    print("  Recognition accuracy on augmented held-out face crops and")
    print("  confidence score stability (lower std = more consistent).")
    print()
    accuracy, conf_std, note = measure_reliability(mod)
    if accuracy is None:
        _result("Recognition Accuracy", "N/A", note)
        _result("Confidence Std Dev", "N/A")
    else:
        acc_rating = _rating(accuracy, [0.70, 0.85, 0.95],
                             ["Poor", "Fair", "Good", "Excellent"], reverse=True)
        _result("Recognition Accuracy", f"{accuracy * 100:.1f}%", note)
        _result("Confidence Std Dev", f"{conf_std:.2f}%")
        _result("Rating", acc_rating,
                "Excellent≥95%  Good≥85%  Fair≥70%  Poor<70%")

    # ── Summary ───────────────────────────────────────────────────────────────
    _section("SUMMARY")
    rows = [
        ("Safety (IoU Loss)",
         f"{iou_loss:.4f}" if iou_loss is not None else "N/A"),
        ("Performance (Inference Time)",
         f"{mean_t:.3f} ms" if mean_t is not None else "N/A"),
        ("Manufacturability (Training Time)",
         f"{elapsed:.2f} s"),
        ("Maintainability (MI Score)",
         f"{mi:.2f}/100"),
        ("Reliability (Recognition Accuracy)",
         f"{accuracy * 100:.1f}%" if accuracy is not None else "N/A"),
    ]
    for label, val in rows:
        padding = 40 - len(label)
        print(f"  {label}{'.' * max(1, padding)} {val}")

    print("\n" + "═" * 60 + "\n")


if __name__ == "__main__":
    main()
