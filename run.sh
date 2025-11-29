#!/bin/bash

# ═══════════════════════════════════════════════════════════════════
# Advanced Multi-Modal RAG System - Simple Run Script
# ═══════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════"
echo "  🌋 Advanced Multi-Modal RAG System"
echo "  Starting Web Interface..."
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please run setup first: ./setup.sh"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "⚠️  Ollama server not running. Starting it now..."
    nohup ollama serve > /dev/null 2>&1 &
    sleep 3
    echo "✅ Ollama server started"
fi

echo "🚀 Starting Gradio web interface..."
echo ""
echo "📖 Web UI will be available at: http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# Run the Gradio app
python3 gradio_app.py
