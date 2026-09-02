@echo off
cd /d "%~dp0"

echo Checking for Python...
where python >nul 2>nul
if errorlevel 1 (
    echo Python not found - installing via winget...
    winget install --id Python.Python.3.12 -e --source winget --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo.
        echo Automatic install failed. Please install Python 3.12 manually from python.org,
        echo making sure to check "Add python.exe to PATH" during setup, then run this file again.
        pause
        exit /b 1
    )
    echo.
    echo Python was just installed. Close this window and run setup_install.bat again
    echo so this script can find it on PATH.
    pause
    exit /b 0
)

echo Python found. Installing required packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Package install failed - check the messages above.
    pause
    exit /b 1
)

if not exist ".env" (
    echo OPENAI_API_KEY=paste-your-key-here> .env
    echo.
    echo Created a blank .env file. Open the app and use the Settings tab to paste your API key,
    echo or edit .env directly.
)

echo.
echo Setup complete. Run start_app.bat to launch the app.
pause
