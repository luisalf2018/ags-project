@echo off
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH.
    echo Run setup_install.bat first if you have not already.
    pause
    exit /b 1
)
python -m streamlit run app\app.py
pause
