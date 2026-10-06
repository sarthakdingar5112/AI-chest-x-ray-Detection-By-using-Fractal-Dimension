@echo off
REM ===== Run this ONCE (needs internet, ~15-20 min first time) =====
echo [1/3] Creating Python environment...
cd /d "%~dp0backend"
py -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo [2/3] Installing backend packages (includes PyTorch + TorchXRayVision)...
pip install -r requirements.txt
echo [3/3] Installing frontend packages...
cd /d "%~dp0frontend"
call npm install
echo.
echo DONE. Next: double-click 2_START_BACKEND.bat, then 3_START_FRONTEND.bat
pause
