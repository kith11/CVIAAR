@echo off
cd /d "%~dp0"
cd ..
echo Starting CVI Attendance System in Production Mode (Waitress)...
call .venv\Scripts\activate

:: Start Ngrok Tunnel in a separate window
start "Public Share Tunnel" cmd /c "python scripts\share_app.py & pause"

:: Start Main Server
python wsgi.py
pause
