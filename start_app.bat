@echo off
cd /d "%~dp0"
set "PY_VERSION_TEXT="
for /f "delims=" %%v in ('python --version 2^>^&1') do set "PY_VERSION_TEXT=%%v"
echo %PY_VERSION_TEXT% | findstr /c:"was not found" >nul
if not errorlevel 1 set "PY_VERSION_TEXT="

if "%PY_VERSION_TEXT%"=="" (
    echo Python was not found on PATH.
    echo Run setup_install.bat first if you have not already.
    pause
    exit /b 1
)
python -m streamlit run app\app.py
pause
