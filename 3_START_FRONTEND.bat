@echo off
cd /d "%~dp0frontend"
echo Starting website on http://localhost:5173
call npm run dev -- --open
pause
