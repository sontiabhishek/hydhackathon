@echo off
REM This script automates the setup of the Python environment for the summarizer app.
REM It creates a virtual environment and installs all necessary dependencies.

echo --- Setting up Python virtual environment ---

REM Check if python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo X Error: python is not installed or not in your PATH. Please install Python and try again.
    exit /b 1
)

REM Create virtual environment
echo 1. Creating virtual environment in '.venv'...
python -m venv .venv
if %errorlevel% neq 0 (
    echo X Failed to create virtual environment.
    exit /b 1
)
echo V Virtual environment created successfully.

REM Install dependencies using the pip from the new virtual environment
echo 2. Installing dependencies from requirements.txt...
.venv\Scripts\pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo X Failed to install dependencies.
    exit /b 1
)
echo V Dependencies installed successfully.

echo.
echo --- Setup Complete ---
echo To activate the virtual environment for your current session, run:
echo .venv\Scripts\activate
echo.
echo Then, to run the application, use the launcher script:
echo python launch.py
echo.