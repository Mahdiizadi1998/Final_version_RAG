@echo off
REM ═══════════════════════════════════════════════════════════════════
REM Advanced Multi-Modal RAG System - Windows Run Script
REM ═══════════════════════════════════════════════════════════════════

echo ════════════════════════════════════════════════════════════════
echo   Advanced Multi-Modal RAG System - Starting Web UI
echo ════════════════════════════════════════════════════════════════
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run setup first: setup.bat
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if Ollama is running
echo Checking Ollama service...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if errorlevel 1 (
    echo [WARNING] Ollama may not be running
    echo Starting Ollama in background...
    start /B ollama serve
    timeout /t 3 /nobreak >nul
    echo [OK] Ollama started
) else (
    echo [OK] Ollama is already running
)
echo.

REM Start Gradio app
echo Starting Gradio web interface...
echo.
echo Web UI will be available at: http://localhost:7860
echo.
echo Press Ctrl+C to stop the server
echo.
echo ════════════════════════════════════════════════════════════════
echo.

python gradio_app.py
