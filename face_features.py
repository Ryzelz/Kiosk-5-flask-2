"""
face_features.py — LBP feature extraction pipeline for face recognition.

Provides: FEATURE_DIM, _GRID, _HIST_BINS,
          _get_clahe, _compute_lbp, _align_face, _prepare_face,
          _extract_face_features, augment_face
"""

import cv2
import numpy as np

_GRID = 4
_HIST_BINS = 64
FEATURE_DIM = _GRID * _GRID * _HIST_BINS  # 1024

_clahe = None


def _get_clahe():
    global _clahe
    if _clahe is None:
        _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _clahe


def _compute_lbp(gray):
    """Compute LBP code image using vectorised 8-neighbor circular pattern."""
    h, w = gray.shape
    p = np.pad(gray.astype(np.int32), 1, mode="edge")
    center = p[1:h + 1, 1:w + 1]
    neighbors = [
        p[0:h,     0:w],       # (-1,-1)
        p[0:h,     1:w + 1],   # (-1, 0)
        p[0:h,     2:w + 2],   # (-1,+1)
        p[1:h + 1, 2:w + 2],   # ( 0,+1)
        p[2:h + 2, 2:w + 2],   # (+1,+1)
        p[2:h + 2, 1:w + 1],   # (+1, 0)
        p[2:h + 2, 0:w],       # (+1,-1)
        p[1:h + 1, 0:w],       # ( 0,-1)
    ]
    lbp = np.zeros((h, w), dtype=np.uint8)
    for k, n in enumerate(neighbors):
        lbp += (n >= center).astype(np.uint8) * (1 << k)
    return lbp


def _align_face(gray_face, eye_detector):
    """Rotate face image so detected eyes are level; skip if alignment is ambiguous."""
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
    return cv2.warpAffine(
        gray_face, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _prepare_face(face, eye_detector):
    resized = cv2.resize(face, (96, 96))
    aligned = _align_face(resized, eye_detector)
    enhanced = _get_clahe().apply(aligned)
    return enhanced


def _extract_face_features(face, eye_detector):
    """Extract grid-based LBP histogram feature vector from a grayscale face crop."""
    prepared = _prepare_face(face, eye_detector)
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


def augment_face(face):
    h, w = face.shape

    low_brightness = cv2.convertScaleAbs(face, alpha=0.7, beta=-30)
    high_brightness = cv2.convertScaleAbs(face, alpha=1.3, beta=30)
    zoom_in = cv2.resize(
        face[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)], (w, h)
    )

    zoom_out = cv2.resize(face, None, fx=0.8, fy=0.8)
    pad_h = h - zoom_out.shape[0]
    pad_w = w - zoom_out.shape[1]
    zoom_out = cv2.copyMakeBorder(
        zoom_out,
        pad_h // 2, pad_h - pad_h // 2,
        pad_w // 2, pad_w - pad_w // 2,
        cv2.BORDER_CONSTANT,
    )

    move_left  = cv2.warpAffine(face, np.float32([[1, 0, -10], [0, 1,   0]]), (w, h))
    move_right = cv2.warpAffine(face, np.float32([[1, 0,  10], [0, 1,   0]]), (w, h))
    move_up    = cv2.warpAffine(face, np.float32([[1, 0,   0], [0, 1, -10]]), (w, h))
    move_down  = cv2.warpAffine(face, np.float32([[1, 0,   0], [0, 1,  10]]), (w, h))

    rot_r = cv2.warpAffine(face, cv2.getRotationMatrix2D((w // 2, h // 2),  10, 1), (w, h))
    rot_l = cv2.warpAffine(face, cv2.getRotationMatrix2D((w // 2, h // 2), -10, 1), (w, h))
    flip  = cv2.flip(face, 1)

    return [
        low_brightness, high_brightness,
        zoom_in, zoom_out,
        move_left, move_right, move_up, move_down,
        rot_r, rot_l, flip,
    ]
