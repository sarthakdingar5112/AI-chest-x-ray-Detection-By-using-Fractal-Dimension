@echo off
cd /d "%~dp0backend"
call .venv\Scripts\activate.bat
echo Starting FractalScan API on http://localhost:8000  (first start downloads AI weights ~300 MB)
uvicorn app.main:app --port 8000
pause
