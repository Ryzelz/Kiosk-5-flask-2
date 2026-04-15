#!/usr/bin/env python
import py_compile
import sys
import os

os.chdir(r'C:\Users\ryzel\PycharmProjects\Kiosk-5-flask')

files = [
    'yolov10.py', 'main.py', 'wsgi.py',
    'website/__init__.py', 'website/views.py', 'website/test.py',
    'website/face_profiles.py', 'website/models.py', 'website/admin.py',
    'website/forms.py', 'website/auth.py', 'website/paymongo.py'
]

errors = []
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f'✓ {f}')
    except py_compile.PyCompileError as e:
        errors.append(str(e))
        print(f'✗ {f}')

if errors:
    print('\n=== SYNTAX ERRORS FOUND ===')
    for err in errors:
        print(err)
    sys.exit(1)
else:
    print('\n✓ All files passed syntax validation.')
    sys.exit(0)
