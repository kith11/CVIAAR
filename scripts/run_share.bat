@echo off
cd /d "%~dp0"
echo Starting Public Share Tunnel...
call .venv\Scripts\activate
python scripts\share_app.py
pause
