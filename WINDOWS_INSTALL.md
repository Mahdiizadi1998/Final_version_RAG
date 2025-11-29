# 🪟 Windows Installation Guide

## Quick Start for Windows Users

### Step 1: Install Ollama (Required)

1. Download Ollama for Windows: https://ollama.com/download/windows
2. Run the installer
3. Ollama will start automatically

### Step 2: Run Setup

Open **Command Prompt** or **PowerShell** in the project folder and run:

```cmd
setup.bat
```

This will automatically:
- ✅ Check Python installation
- ✅ Download AI models (llama3.1:8b, llava:7b)
- ✅ Create virtual environment
- ✅ Install all Python packages
- ✅ Download spaCy model
- ✅ Create data folders

**Time:** ~10-15 minutes (mostly downloading models)

### Step 3: Start the Web UI

```cmd
run.bat
```

Then open your browser to: **http://localhost:7860**

---

## Troubleshooting

### "Python is not recognized"

Install Python 3.8+ from: https://www.python.org/downloads/

**Important:** Check "Add Python to PATH" during installation!

### "Ollama is not recognized"

1. Install Ollama from: https://ollama.com/download/windows
2. Restart your computer
3. Run `setup.bat` again

### "Failed to pull models"

Make sure Ollama is running:
1. Open Task Manager (Ctrl+Shift+Esc)
2. Look for "ollama.exe" in processes
3. If not running, open a new terminal and type: `ollama serve`

### Port 7860 already in use

Edit `gradio_app.py` and change the port:
```python
demo.launch(server_port=7870)  # Use different port
```

### Virtual environment issues

Delete the `venv` folder and run `setup.bat` again:
```cmd
rmdir /s /q venv
setup.bat
```

---

## Manual Installation (if setup.bat fails)

```cmd
# 1. Install Ollama manually from website

# 2. Pull models
ollama pull llama3.1:8b
ollama pull llava:7b

# 3. Create virtual environment
python -m venv venv

# 4. Activate it
venv\Scripts\activate

# 5. Install packages
pip install --upgrade pip
pip install -r requirements.txt

# 6. Download spaCy model
python -m spacy download en_core_web_sm

# 7. Run the app
python gradio_app.py
```

---

## System Requirements

- **OS:** Windows 10/11
- **Python:** 3.8 or higher
- **RAM:** 8GB minimum (16GB recommended)
- **Storage:** 10GB free space for models
- **Internet:** Required for initial setup

---

## Next Steps

Once the web UI is running:
1. Go to "📁 Upload Documents" tab
2. Upload PDF/DOCX/XLSX files
3. Click "📤 Process Documents"
4. Go to "❓ Ask Questions" tab
5. Type your question and get answers!

---

## Additional Notes

### Using WSL (Windows Subsystem for Linux)

If you prefer Linux commands on Windows:

```bash
# Install WSL
wsl --install

# Restart computer, then:
wsl
cd /mnt/c/Users/YourUsername/Downloads/Final_version_RAG-main
./setup.sh
./run.sh
```

### Using Git Bash

If you have Git for Windows installed:

```bash
bash setup.sh
bash run.sh
```

---

**Need help?** Open an issue on GitHub!
