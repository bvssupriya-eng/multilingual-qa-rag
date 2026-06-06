@echo off
echo Fixing virtual environment...
echo.

REM Deactivate if active
call qa_env\Scripts\deactivate.bat 2>nul

REM Wait a moment
timeout /t 2 /nobreak >nul

REM Remove old environment
echo Removing old environment...
rmdir /s /q qa_env

REM Create fresh environment
echo Creating fresh virtual environment...
python -m venv qa_env

REM Activate new environment
echo Activating environment...
call qa_env\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Done! Environment is ready.
echo Run: streamlit run app.py
pause
