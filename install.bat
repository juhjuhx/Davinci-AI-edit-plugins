@echo off
set PYTHON=
where python >nul 2>nul
if %errorlevel%==0 (set PYTHON=python) else (set "PYTHON=C:\Users\%USERNAME%\.conda\envs\smart_aroll\python.exe")
echo ==================================================
echo   Smart A-Roll v4.0 - Install Dependencies
echo ==================================================
echo.
"%PYTHON%" -m pip install faster-whisper flask edge-tts librosa numpy pydub
echo.
echo Done! Run launch.bat to start.
pause
