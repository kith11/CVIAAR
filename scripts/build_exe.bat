@echo off
cd /d "%~dp0"
cd ..
echo Building Attendance Kiosk Executable...
call .venv\Scripts\activate
pyinstaller --noconfirm --clean AttendanceKiosk.spec
echo.
echo Build Complete.
echo You can find the executable in dist\AttendanceKiosk\AttendanceKiosk.exe
pause
