@echo off
setlocal

cd /d "%~dp0"

if "%APP_HOST%"=="" set "APP_HOST=0.0.0.0"
if "%APP_PORT%"=="" set "APP_PORT=5000"
if "%APP_DEBUG%"=="" set "APP_DEBUG=1"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_CMD=.venv\Scripts\python.exe"
) else (
    set "PYTHON_CMD=python"
)

echo Starting Wideye local host...
echo Host: %APP_HOST%
echo Port: %APP_PORT%
echo Debug: %APP_DEBUG%

"%PYTHON_CMD%" main.py
