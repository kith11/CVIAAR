@echo off
echo Setting up Auto-Start for Attendance Kiosk...

set "TARGET_SCRIPT=%~dp0run_kiosk.bat"
set "WORK_DIR=%~dp0"
set "SHORTCUT_PATH=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\AttendanceKiosk.lnk"

echo Target: %TARGET_SCRIPT%
echo Working Dir: %WORK_DIR%
echo Shortcut: %SHORTCUT_PATH%

powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%SHORTCUT_PATH%'); $Shortcut.TargetPath = '%TARGET_SCRIPT%'; $Shortcut.WorkingDirectory = '%WORK_DIR%'; $Shortcut.WindowStyle = 7; $Shortcut.Save()"

if exist "%SHORTCUT_PATH%" (
    echo.
    echo SUCCESS! The Attendance Kiosk will now start automatically when you log in.
    echo You can find the shortcut in your Startup folder.
) else (
    echo.
    echo FAILED to create shortcut. Please try running as Administrator.
)

pause
