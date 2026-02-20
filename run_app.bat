@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM Ensure we always have a persistent window even when launched by Explorer.
if "%RUN_APP_INNER%"=="" (
    set "RUN_APP_INNER=1"
    rem cmd uses double-quotes, not backslash escaping
    start "Alzheimer RAS Agent" cmd /k ""%~f0" %*"
    exit /b 0
)

chcp 65001 >nul

set "PY=.venv\Scripts\python.exe"
set "DATASET=data\processed\articles_clean.parquet"
set "INDEX_DIR=data\index\faiss"
set "INDEX_FAISS=%INDEX_DIR%\index.faiss"
set "INDEX_META=%INDEX_DIR%\index.pkl"

if not exist "%PY%" (
    echo [ERROR] Virtual environment not found: %PY%
    echo Run once (in this folder^):
    echo   python -m venv .venv
    echo   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    echo   .\.venv\Scripts\python.exe -m pip install -e .
    echo.
    pause
    exit /b 1
)

if not exist "data\processed" mkdir "data\processed" >nul 2>nul
if not exist "%INDEX_DIR%" mkdir "%INDEX_DIR%" >nul 2>nul

if not exist "%DATASET%" (
    echo Dataset not found. Building dataset...
    "%PY%" scripts\build_dataset.py
    if errorlevel 1 (
        echo.
        echo [ERROR] Dataset build failed.
        pause
        exit /b 1
    )
)

if not exist "%INDEX_FAISS%" (
    echo FAISS index not found. Building index...
    "%PY%" scripts\build_index.py
    if errorlevel 1 (
        echo.
        echo [ERROR] Index build failed.
        pause
        exit /b 1
    )
)

if not exist "%INDEX_META%" (
    echo [ERROR] Index metadata missing: %INDEX_META%
    echo Rebuild index: scripts\build_index.py
    echo.
    pause
    exit /b 1
)

echo.
echo Starting app...
echo (Tip: type exit to quit)
echo.
"%PY%" desktop_app.py %*

echo.
echo App finished. Press any key to close this window...
pause >nul
