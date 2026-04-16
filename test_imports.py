#!/usr/bin/env python
"""Test script to verify yolov10 module imports."""

import sys
import traceback

print("=" * 70)
print("Test 1: Import yolov10 and list public API")
print("=" * 70)

try:
    sys.path.insert(0, '.')
    import yolov10
    public_api = [x for x in dir(yolov10) if not x.startswith('__')]
    print(f"OK: {public_api}")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()

print("\n" + "=" * 70)
print("Test 2: Import specific public API functions")
print("=" * 70)

try:
    from yolov10 import (
        capture_face_training,
        confirm_known_face,
        train_faces,
        faces_ready,
        AVAILABLE_MODELS,
        FEATURE_DIM,
        augment_face
    )
    print("All public API imports OK")
    print(f"  - capture_face_training: {capture_face_training}")
    print(f"  - confirm_known_face: {confirm_known_face}")
    print(f"  - train_faces: {train_faces}")
    print(f"  - faces_ready: {faces_ready}")
    print(f"  - AVAILABLE_MODELS: {list(AVAILABLE_MODELS.keys())}")
    print(f"  - FEATURE_DIM: {FEATURE_DIM}")
    print(f"  - augment_face: {augment_face}")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()

print("\n" + "=" * 70)
print("Test 3: Run pytest on tests/ directory")
print("=" * 70)

import subprocess
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-x", "-q"],
    capture_output=False,
    text=True
)
sys.exit(result.returncode)
