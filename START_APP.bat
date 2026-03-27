@echo off
chcp 65001 > nul
echo ================================================
echo  FloodSense AI - Starting Application
echo ================================================
echo.

cd /d "%~dp0"

:: Check if models exist
if not exist "models\random_forest.pkl" (
    echo ERROR: Models not trained yet!
    echo Please run TRAIN_MODELS.bat first.
    pause
    exit /b 1
)

echo Starting Flask backend server...
echo Backend will run on: http://localhost:5000
echo.
echo Opening app in browser in 3 seconds...
timeout /t 3 > nul

:: Open browser
start "" "http://localhost:5000"

:: Start the Flask server
python -X utf8 backend\app.py

pause
