@echo off
cd /d "%~dp0"
echo Directory: %CD%
echo Open http://127.0.0.1:8000/docs
python -m uvicorn src.tumor_app.api:app --reload --host 127.0.0.1 --port 8000
pause
