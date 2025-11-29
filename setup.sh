#!/bin/bash

# ═══════════════════════════════════════════════════════════════════
# Advanced Multi-Modal RAG System - Automatic Setup Script
# ═══════════════════════════════════════════════════════════════════

set -e  # Exit on any error

echo "════════════════════════════════════════════════════════════════"
echo "  Advanced Multi-Modal RAG System - Automatic Setup"
echo "════════════════════════════════════════════════════════════════"
echo ""

# ═══════════════════════════════════════════════════════════════════
# STEP 1: Check Python Installation
# ═══════════════════════════════════════════════════════════════════

echo "🔍 Step 1: Checking Python installation..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed!"
    echo "   Please install Python 3.8 or higher from https://www.python.org/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d " " -f 2)
echo "✅ Python $PYTHON_VERSION found"
echo ""

# ═══════════════════════════════════════════════════════════════════
# STEP 2: Check/Install Ollama
# ═══════════════════════════════════════════════════════════════════

echo "🔍 Step 2: Checking Ollama installation..."

if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama is not installed!"
    echo "   Installing Ollama..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "   Please install Ollama manually from: https://ollama.com/download"
        echo "   Then run this script again."
        exit 1
    else
        echo "   Please install Ollama manually from: https://ollama.com/download"
        echo "   Then run this script again."
        exit 1
    fi
else
    echo "✅ Ollama is already installed"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════
# STEP 3: Start Ollama Server
# ═══════════════════════════════════════════════════════════════════

echo "🔍 Step 3: Starting Ollama server..."

# Check if Ollama is already running
if pgrep -x "ollama" > /dev/null; then
    echo "✅ Ollama server is already running"
else
    echo "   Starting Ollama server in background..."
    nohup ollama serve > /dev/null 2>&1 &
    sleep 3
    echo "✅ Ollama server started"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════
# STEP 4: Pull Required Models
# ═══════════════════════════════════════════════════════════════════

echo "🔍 Step 4: Downloading AI models..."
echo "   This may take several minutes (first time only)..."
echo ""

echo "   📥 Pulling llama3.1:8b (text generation)..."
ollama pull llama3.1:8b

echo ""
echo "   📥 Pulling llava:7b (vision analysis)..."
ollama pull llava:7b

echo ""
echo "✅ All models downloaded"
echo ""

# ═══════════════════════════════════════════════════════════════════
# STEP 5: Create Virtual Environment
# ═══════════════════════════════════════════════════════════════════

echo "🔍 Step 5: Setting up Python environment..."

if [ ! -d "venv" ]; then
    echo "   Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

echo "✅ Virtual environment activated"
echo ""

# ═══════════════════════════════════════════════════════════════════
# STEP 6: Install Python Dependencies
# ═══════════════════════════════════════════════════════════════════

echo "🔍 Step 6: Installing Python dependencies..."
echo "   This may take a few minutes..."
echo ""

pip install --upgrade pip setuptools wheel

pip install -r requirements.txt

echo ""
echo "✅ All Python packages installed"
echo ""

# ═══════════════════════════════════════════════════════════════════
# STEP 7: Download spaCy Model
# ═══════════════════════════════════════════════════════════════════

echo "🔍 Step 7: Downloading spaCy language model..."

python3 -m spacy download en_core_web_sm

echo "✅ spaCy model downloaded"
echo ""

# ═══════════════════════════════════════════════════════════════════
# STEP 8: Create Data Directory
# ═══════════════════════════════════════════════════════════════════

echo "🔍 Step 8: Creating data directories..."

mkdir -p uploaded_documents
mkdir -p data/geothermal_documents

echo "✅ Directories created"
echo ""

# ═══════════════════════════════════════════════════════════════════
# SETUP COMPLETE
# ═══════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════"
echo "  ✅ SETUP COMPLETE!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "🚀 To start the web UI, run:"
echo ""
echo "   ./run.sh"
echo ""
echo "   OR manually:"
echo "   source venv/bin/activate && python3 gradio_app.py"
echo ""
echo "📖 The web interface will open at: http://localhost:7860"
echo ""
echo "════════════════════════════════════════════════════════════════"
