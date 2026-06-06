@echo off
title Smart A-Roll v4.0
cd /d "%~dp0"

:: Find Python
set PYTHON=
where python >nul 2>nul
if %errorlevel%==0 (
    set PYTHON=python
) else if exist "C:\Users\%USERNAME%\.conda\envs\smart_aroll\python.exe" (
    set "PYTHON=C:\Users\%USERNAME%\.conda\envs\smart_aroll\python.exe"
)

if "%PYTHON%"=="" (
    echo ERROR: Python not found. Please install Python 3.10+ or create conda env 'smart_aroll'.
    pause
    exit /b 1
)

echo ==================================================
echo   Smart A-Roll v4.0
echo ==================================================
echo.

echo [1/3] Cleaning up...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8888 ^| findstr LISTENING') do taskkill /F /PID %%a 2>nul
timeout /t 1 /nobreak >nul

echo [2/3] Checking environment...
if not exist "%PYTHON%" (
    echo ERROR: Python not found: %PYTHON%
    pause
    exit /b 1
)
"%PYTHON%" -c "import flask" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    "%PYTHON%" -m pip install faster-whisper flask edge-tts librosa numpy pydub --quiet
)

echo [3/3] Starting server...
echo.
echo   WebUI: http://127.0.0.1:8888
echo   Press CTRL+C to stop
echo.

start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8888"
"%PYTHON%" api\app.py
pause
