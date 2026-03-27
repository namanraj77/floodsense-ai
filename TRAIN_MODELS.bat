@echo off
chcp 65001 > nul
echo ================================================
echo  FloodSense AI - Model Training
echo ================================================
echo.

cd /d "%~dp0"

echo [1/2] Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

echo.
echo [2/2] Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo [3/3] Training models (may take 5-10 minutes)...
python -X utf8 backend\train_models.py
if errorlevel 1 (
    echo ERROR: Model training failed!
    pause
    exit /b 1
)

echo.
echo ================================================
echo  Models trained! Run START_APP.bat to launch.
echo ================================================
pause
