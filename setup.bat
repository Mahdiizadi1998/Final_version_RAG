@echo off
REM ═══════════════════════════════════════════════════════════════════
REM Advanced Multi-Modal RAG System - Windows Setup Script
REM ═══════════════════════════════════════════════════════════════════

echo ════════════════════════════════════════════════════════════════
echo   Advanced Multi-Modal RAG System - Windows Setup
echo ════════════════════════════════════════════════════════════════
echo.

REM ═══════════════════════════════════════════════════════════════════
REM STEP 1: Check Python Installation
REM ═══════════════════════════════════════════════════════════════════

echo [Step 1/7] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.8+ from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% found
echo.

REM ═══════════════════════════════════════════════════════════════════
REM STEP 2: Check Ollama Installation
REM ═══════════════════════════════════════════════════════════════════

echo [Step 2/7] Checking Ollama installation...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama is not installed!
    echo.
    echo Please install Ollama manually:
    echo 1. Go to: https://ollama.com/download/windows
    echo 2. Download and install Ollama
    echo 3. Restart this script after installation
    echo.
    pause
    exit /b 1
) else (
    echo [OK] Ollama is installed
)
echo.

REM ═══════════════════════════════════════════════════════════════════
REM STEP 3: Pull AI Models
REM ═══════════════════════════════════════════════════════════════════

echo [Step 3/7] Downloading AI models...
echo This may take 10-15 minutes (first time only)...
echo.

echo Pulling llama3.1:8b (text generation)...
ollama pull llama3.1:8b
if errorlevel 1 (
    echo [ERROR] Failed to pull llama3.1:8b
    echo Make sure Ollama service is running
    pause
    exit /b 1
)

echo.
echo Pulling llava:7b (vision analysis)...
ollama pull llava:7b
if errorlevel 1 (
    echo [ERROR] Failed to pull llava:7b
    pause
    exit /b 1
)

echo [OK] All models downloaded
echo.

REM ═══════════════════════════════════════════════════════════════════
REM STEP 4: Create Virtual Environment
REM ═══════════════════════════════════════════════════════════════════

echo [Step 4/7] Setting up Python virtual environment...

if exist venv (
    echo [OK] Virtual environment already exists
) else (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)
echo.

REM ═══════════════════════════════════════════════════════════════════
REM STEP 5: Activate Virtual Environment and Upgrade pip
REM ═══════════════════════════════════════════════════════════════════

echo [Step 5/7] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

echo [OK] Virtual environment activated
echo.

echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel
echo.

REM ═══════════════════════════════════════════════════════════════════
REM STEP 6: Install Python Dependencies
REM ═══════════════════════════════════════════════════════════════════

echo [Step 6/7] Installing Python packages...
echo This may take a few minutes...
echo.

pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install Python packages
    pause
    exit /b 1
)

echo [OK] All Python packages installed
echo.

REM ═══════════════════════════════════════════════════════════════════
REM STEP 7: Download spaCy Model
REM ═══════════════════════════════════════════════════════════════════

echo [Step 7/7] Downloading spaCy language model...

python -m spacy download en_core_web_sm
if errorlevel 1 (
    echo [WARNING] Failed to download spaCy model
    echo You may need to download it manually later
)

echo [OK] spaCy model downloaded
echo.

REM ═══════════════════════════════════════════════════════════════════
REM Create Data Directories
REM ═══════════════════════════════════════════════════════════════════

echo Creating data directories...
if not exist "uploaded_documents" mkdir uploaded_documents
if not exist "data\geothermal_documents" mkdir data\geothermal_documents
echo [OK] Directories created
echo.

REM ═══════════════════════════════════════════════════════════════════
REM SETUP COMPLETE
REM ═══════════════════════════════════════════════════════════════════

echo ════════════════════════════════════════════════════════════════
echo   [SUCCESS] SETUP COMPLETE!
echo ════════════════════════════════════════════════════════════════
echo.
echo To start the web UI, run:
echo.
echo    run.bat
echo.
echo    OR manually:
echo    venv\Scripts\activate
echo    python gradio_app.py
echo.
echo The web interface will open at: http://localhost:7860
echo.
echo ════════════════════════════════════════════════════════════════
echo.
pause
