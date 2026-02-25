@echo off
set /p commitMsg="Enter commit message (default: Update attendance system security and UI): "
if "%commitMsg%"=="" set commitMsg="Update attendance system security and UI"

powershell.exe -ExecutionPolicy Bypass -File .\deploy.ps1 -CommitMessage "%commitMsg%"
pause
