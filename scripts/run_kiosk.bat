@echo off
cd /d "%~dp0"
cd ..
echo Starting Kiosk Mode...
call .venv\Scripts\activate
python kiosk.py
pause
