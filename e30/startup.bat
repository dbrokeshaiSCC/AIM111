@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -m venv .venv
    if errorlevel 1 goto :error
)
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :error
.venv\Scripts\python.exe -m streamlit run fantasyanalytics.py
exit /b %errorlevel%
:error
echo Setup failed. Review the error messages above.
pause
exit /b 1
