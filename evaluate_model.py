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
    """Return a rating label.  thresholds = ascending list of break-points."""
    for threshold, label in zip(thresholds, labels[:-1]):
        if (value <= threshold) if not reverse else (value >= threshold):
            return label
    return labels[-1]


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
    """Time _predict_face() on saved face images."""
    person_dir = FACES_DIR / "seb"
    images = [
        p for p in person_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and "_aug_" not in p.stem
    ] if person_dir.exists() else []

    if not images:
        return None, None, "no face images found"

    grays = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
             for p in images[:10] if cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) is not None]
    if not grays:
        return None, None, "could not load images"

    times = []
    for _ in range(n_runs):
        img = grays[_ % len(grays)]
        t0 = time.perf_counter()
        mod._extract_face_features(img)
        times.append((time.perf_counter() - t0) * 1000)

    return statistics.mean(times), statistics.stdev(times), f"{n_runs} runs"


# ── 3. Manufacturability: Training Time ───────────────────────────────────────

def measure_training_time(mod):
    """Delete trainer.npz, re-run train_faces(), measure wall time."""
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
    """Use radon MI if available, else fallback to simple LOC-based metric."""
    try:
        import radon.metrics as rm
        source = YOLO_FILE.read_text(encoding="utf-8")
        mi_score = rm.mi_visit(source, multi=True)
        return round(mi_score, 2), "radon MI (0-100, >65 = A)"
    except ImportError:
        pass

    # Fallback: run `radon mi -s yolov10.py` as subprocess
    try:
        result = subprocess.run(
            [sys.executable, "-m", "radon", "mi", "-s", str(YOLO_FILE)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            # parse "yolov10.py - A (82.34)"
            for line in result.stdout.splitlines():
                if "(" in line and ")" in line:
                    score_str = line.split("(")[-1].rstrip(")")
                    try:
                        return float(score_str), "radon MI (0-100, >65 = A)"
                    except ValueError:
                        pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Final fallback: count cyclomatic complexity via ast
    import ast
    source = YOLO_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    total_lines = len(source.splitlines())
    # Halstead-inspired approximation (not true MI, but directional)
    avg_func_len = total_lines / max(len(functions), 1)
    mi_approx = max(0, min(100, 171 - 5.2 * np.log(max(1, avg_func_len))
                           - 0.23 * len(functions) - 16.2 * np.log(max(1, total_lines))))
    return round(float(mi_approx), 2), "estimated MI (radon not installed)"


# ── 5. Reliability: Model Consistency ────────────────────────────────────────

def measure_reliability(mod):
    """Predict on held-out augmented faces, report recognition accuracy."""
    person_dir = FACES_DIR / "seb"
    if not person_dir.exists():
        return None, None, "no face data"

    # Use aug images as held-out test set (not used during training)
    aug_images = [
        p for p in person_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and "_aug_" in p.stem
    ]

    if not aug_images:
        return None, None, "no augmented images"

    if not mod.faces_ready():
        return None, None, "model not trained"

    correct = 0
    confidences = []

    for img_path in aug_images:
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        name, conf = mod._predict_face(gray)
        confidences.append(conf)
        if name.lower() == "seb":
            correct += 1

    total = len(aug_images)
    accuracy = correct / total if total else 0.0
    conf_std = statistics.stdev(confidences) if len(confidences) > 1 else 0.0

    return accuracy, conf_std, f"{correct}/{total} aug images correctly recognised"


# ── Report ────────────────────────────────────────────────────────────────────

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
