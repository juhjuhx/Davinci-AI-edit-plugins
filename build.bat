@echo off
title Smart A-Roll - Build
cd /d "%~dp0"
set PYTHON=
where python >nul 2>nul
if %errorlevel%==0 (set PYTHON=python) else (set "PYTHON=C:\Users\%USERNAME%\.conda\envs\smart_aroll\python.exe")

echo ==================================================
echo   Smart A-Roll v4.0 - Build
echo ==================================================
echo.

echo [1/6] Installing PyInstaller...
"%PYTHON%" -m pip install pyinstaller --quiet

echo [2/6] Searching for FFmpeg...
set FFMPEG_FOUND=
for %%P in (
    "C:\ffmpeg\bin\ffmpeg.exe"
    "C:\Program Files\ffmpeg\bin\ffmpeg.exe"
) do (
    if exist %%P (
        set FFMPEG_FOUND=%%~dpP
        echo       Found: %%P
    )
)
where ffmpeg >nul 2>nul
if %errorlevel%==0 (
    for /f "tokens=*" %%i in ('where ffmpeg') do (
        if not defined FFMPEG_FOUND (
            set FFMPEG_FOUND=%%~dpi
            echo       Found in PATH: %%i
        )
    )
)
if not defined FFMPEG_FOUND (
    echo.
    echo       FFmpeg not found automatically.
    echo       Download: https://github.com/BtbN/FFmpeg-Builds/releases
    echo.
    echo       Enter path to folder containing ffmpeg.exe:
    set /p FFMPEG_FOUND="       Path: "
)
if not defined FFMPEG_FOUND (
    echo       Skipping FFmpeg bundling.
) else if not exist "%FFMPEG_FOUND%\ffmpeg.exe" (
    echo       WARNING: ffmpeg.exe not found at %FFMPEG_FOUND%
    echo       Continuing without FFmpeg.
    set FFMPEG_FOUND=
) else (
    echo       Using: %FFMPEG_FOUND%
)

echo [3/6] Building .exe...
"%PYTHON%" -m PyInstaller --name=SmartARoll --onefile --noconfirm --clean --add-data="core;core" --add-data="workers;workers" --add-data="ui;ui" --add-data="config.json;." --hidden-import=flask --hidden-import=faster_whisper --hidden-import=edge_tts --hidden-import=librosa --hidden-import=numpy --hidden-import=pydub --hidden-import=core.config --hidden-import=core.models --hidden-import=core.analyzer --hidden-import=core.ffmpeg_gpu --hidden-import=core.edl --hidden-import=core.llm --hidden-import=core.tts --hidden-import=core.enhance --hidden-import=core.resolve --hidden-import=workers.analyze_worker api\app.py

if errorlevel 1 (
    echo       BUILD FAILED
    pause
    exit /b 1
)

echo [4/6] Packaging...
if exist dist\SmartARoll_v4.0 rmdir /s /q dist\SmartARoll_v4.0
mkdir dist\SmartARoll_v4.0
copy dist\SmartARoll.exe dist\SmartARoll_v4.0\
copy config.json dist\SmartARoll_v4.0\
copy apply_to_resolve.py dist\SmartARoll_v4.0\
mkdir dist\SmartARoll_v4.0\results
mkdir dist\SmartARoll_v4.0\logs
if defined FFMPEG_FOUND (
    mkdir dist\SmartARoll_v4.0\ffmpeg
    copy "%FFMPEG_FOUND%\ffmpeg.exe" dist\SmartARoll_v4.0\ffmpeg\
    copy "%FFMPEG_FOUND%\ffprobe.exe" dist\SmartARoll_v4.0\ffmpeg\
)
(
echo @echo off
echo title Smart A-Roll v4.0
echo cd /d "%%~dp0"
echo echo ==================================================
echo echo   Smart A-Roll v4.0
echo echo ==================================================
echo echo.
echo if not exist "qwen2.5-0.5b-instruct-q4_k_m.gguf" ^(
echo     echo [INFO] LLM model not found. LLM features disabled.
echo     echo Download: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF
echo     echo File: qwen2.5-0.5b-instruct-q4_k_m.gguf
echo     echo Place in this folder, then restart.
echo     echo.
echo ^)
echo start "" cmd /c "timeout /t 2 /nobreak ^>nul ^&^& start http://127.0.0.1:8888"
echo SmartARoll.exe
echo pause
) > dist\SmartARoll_v4.0\launch.bat

echo [5/6] Creating zip...
powershell -Command "Compress-Archive -Path 'dist\SmartARoll_v4.0\*' -DestinationPath 'dist\SmartARoll_v4.0.zip' -Force"

echo [6/6] Creating installer...
where makensis >nul 2>nul
if %errorlevel%==0 (
    makensis /DVERSION=4.0 /DOUTDIR=dist installer.nsi
    echo       Installer created: dist\SmartARoll_v4.0_Setup.exe
) else (
    echo       NSIS not found. Skipping installer.
    echo       Download NSIS: https://nsis.sourceforge.io/Download
)

echo.
echo ==================================================
echo   Build complete!
if exist dist\SmartARoll_v4.0\SmartARoll.exe echo   .exe:  dist\SmartARoll_v4.0\SmartARoll.exe
if exist dist\SmartARoll_v4.0.zip echo   .zip:  dist\SmartARoll_v4.0.zip
if exist dist\SmartARoll_v4.0_Setup.exe echo   Setup: dist\SmartARoll_v4.0_Setup.exe
echo ==================================================
pause
