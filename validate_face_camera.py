#!/usr/bin/env python
import re
import sys

# Validate HTML template
html_file = r'website\templates\face_camera.html'
try:
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Check for Jinja2 block balance
    jinja_opens = len(re.findall(r'{%', html_content))
    jinja_closes = len(re.findall(r'%}', html_content))
    
    if jinja_opens == jinja_closes:
        print(f'✓ face_camera.html')
    else:
        print(f'✗ face_camera.html: Jinja2 blocks unbalanced ({jinja_opens} opens, {jinja_closes} closes)')
        sys.exit(1)
except Exception as e:
    print(f'✗ face_camera.html: {e}')
    sys.exit(1)

# Validate JavaScript
js_file = r'website\static\js\faceCamera.js'
try:
    with open(js_file, 'r', encoding='utf-8') as f:
        js_content = f.read()
    
    # Count brackets
    open_braces = js_content.count('{')
    close_braces = js_content.count('}')
    open_parens = js_content.count('(')
    close_parens = js_content.count(')')
    open_brackets = js_content.count('[')
    close_brackets = js_content.count(']')
    
    if (open_braces == close_braces and open_parens == close_parens and open_brackets == close_brackets):
        print(f'✓ faceCamera.js')
    else:
        print(f'✗ faceCamera.js: Bracket mismatch - {{ }}: {open_braces}/{close_braces}, ( ): {open_parens}/{close_parens}, [ ]: {open_brackets}/{close_brackets}')
        sys.exit(1)
except Exception as e:
    print(f'✗ faceCamera.js: {e}')
    sys.exit(1)

print('\n✓ All face camera UI files are syntactically valid.')
sys.exit(0)
