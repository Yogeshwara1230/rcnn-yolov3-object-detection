@echo off
cd /d "%~dp0"
if "%1"=="" (
  python src\live_detect.py --model yolo
) else (
  python src\live_detect.py --model %1
)
pause
