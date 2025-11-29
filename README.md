# 🌋 Advanced Multi-Modal RAG System for Geothermal Well Reports

A production-ready Retrieval-Augmented Generation (RAG) system designed specifically for analyzing geothermal well reports. This system processes multi-modal documents (text, tables, images) and provides intelligent answers to technical questions about wells, formations, temperatures, pressures, and more.

## ✨ Key Features

### 🎯 Core Capabilities
- **Multi-Modal Processing**: Extracts text, tables, and technical images from PDF, DOCX, and XLSX files
- **Vision Analysis**: Uses llava:7b to caption and classify technical diagrams, plots, and schematics
- **Triple Metadata Extraction**: Combines Regex + NLP + LLM for comprehensive metadata (7x faster)
- **Advanced Chunking**: Late Chunking + Contextual Enrichment (49% better retrieval)
- **RAPTOR Tree**: Hierarchical document summarization for complex queries
- **Hybrid Retrieval**: FAISS vector search + BM25 sparse retrieval + Knowledge Graph traversal
- **Intelligent Routing**: Automatically selects optimal strategy based on query type
- **Grounded Answers**: Citations from source documents with confidence scoring

### ⚡ Performance Optimizations
- **Document-level Metadata**: Extract once per document (7x faster than per-chunk)
- **Regex Chunk Detection**: 0.01s vs 4s with LLM-based detection
- **Batch Encoding**: Process 32 documents simultaneously
- **Table Quality Filtering**: Automatically skip garbled/low-quality tables
- **Duplicate Column Handling**: Robust DataFrame processing

### 🖥️ Easy-to-Use Web Interface
- **Gradio Web UI**: Beautiful, intuitive interface
- **Upload Tab**: Drag-and-drop document upload with progress tracking
- **Query Tab**: Natural language question answering with source citations
- **One-Click Setup**: Automated installation script

## 🚀 Quick Start

### Option 1: Automatic Setup (Recommended)

#### **Linux / macOS:**
```bash
# Clone the repository
git clone https://github.com/Mahdiizadi1998/Final_version_RAG.git
cd Final_version_RAG

# Run automatic setup (installs everything)
./setup.sh

# Start the web interface
./run.sh
```

#### **Windows:**
```cmd
# Download or clone the repository
cd Final_version_RAG-main

# Run automatic setup (installs everything)
setup.bat

# Start the web interface
run.bat
```

📖 **Windows users:** See [WINDOWS_INSTALL.md](WINDOWS_INSTALL.md) for detailed instructions

The web UI will open at: **http://localhost:7860**

### Option 2: Manual Setup

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull required models
ollama pull llama3.1:8b
ollama pull llava:7b

# 3. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download spaCy model
python3 -m spacy download en_core_web_sm

# 6. Run the web interface
python3 gradio_app.py
```

## 📖 Usage

### Web Interface

1. **Upload Documents**
   - Navigate to "📁 Upload Documents" tab
   - Drag and drop or click to select PDF/DOCX/XLSX files
   - Click "📤 Process Documents"
   - Wait for processing to complete

2. **Ask Questions**
   - Navigate to "❓ Ask Questions" tab
   - Type your question in natural language
   - Click "🔍 Get Answer"
   - View answer with confidence score and sources

### Example Questions

```
- What is the temperature in well ADK-GT-01?
- Compare temperatures between all wells
- Summarize all wells in the Slochteren Formation
- Which well has the highest temperature and at what depth?
- What formations are mentioned in the documents?
- List all pressure measurements
```

### Python API (Advanced)

```python
from agentic_rag import AdvancedAgenticRAG
from ingestion_pipeline import DocumentIngestionPipeline

# Initialize system
pipeline = DocumentIngestionPipeline(...)
rag_system = AdvancedAgenticRAG(pipeline=pipeline, ...)

# Ingest documents
pipeline.ingest_directory('./documents', ['*.pdf', '*.docx'])

# Query system
result = rag_system.query("What is the temperature in well ADK-GT-01?")
print(result['answer'])
print(f"Confidence: {result['confidence']:.1%}")

# Interactive mode
rag_system.interactive_mode()
```

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Gradio Web UI                           │
│              (Upload Documents & Ask Questions)              │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  Agentic RAG System                         │
│  (Query Routing → Retrieval → Answer Generation)           │
└─────┬────────────────────────────────────────┬─────────────┘
      │                                        │
┌─────▼──────────────────┐         ┌──────────▼─────────────┐
│  Ingestion Pipeline    │         │   Query Processing     │
│  • Document Parser     │         │   • Query Router       │
│  • Vision Processor    │         │   • Answer Generator   │
│  • Metadata Extractor  │         │   • Confidence Scorer  │
│  • Semantic Chunker    │         └────────────────────────┘
│  • RAPTOR Tree Builder │
└─────┬──────────────────┘
      │
┌─────▼──────────────────────────────────────────────────────┐
│                    Storage Layer                            │
│  • FAISS (Vector Search)                                   │
│  • BM25 (Sparse Retrieval)                                 │
│  • NetworkX (Knowledge Graph)                              │
│  • SQLite (Structured Tables)                              │
│  • RAPTOR Tree (Hierarchical Summaries)                    │
└────────────────────────────────────────────────────────────┘
```

### File Structure

```
Final_version_RAG/
├── gradio_app.py              # Web UI interface
├── agentic_rag.py             # Main RAG orchestration
├── ingestion_pipeline.py      # Document processing pipeline
├── query_router.py            # Query classification & routing
├── answer_generator.py        # Grounded answer generation
├── document_parser.py         # Multi-format document parsing
├── vision_processor.py        # Image analysis with llava
├── metadata_extractor.py      # Triple metadata extraction
├── semantic_chunker.py        # Late chunking + enrichment
├── raptor_tree.py             # Hierarchical summarization
├── hybrid_store.py            # FAISS + BM25 + Graph storage
├── sql_store.py               # Structured table storage
├── ollama_client.py           # Ollama API wrapper
├── setup.sh                   # Automatic setup script
├── run.sh                     # Simple run script
├── requirements.txt           # Python dependencies
├── advanced_rag_system.ipynb  # Jupyter notebook demo
└── README.md                  # This file
```

## 📋 Requirements

- **Python**: 3.8 or higher
- **Ollama**: Latest version
- **Models**: llama3.1:8b, llava:7b
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB for models + data

## 🔧 Configuration

### Environment Variables

```bash
# Ollama server settings (optional)
export OLLAMA_HOST="0.0.0.0:11434"
export OLLAMA_ORIGINS="*"

# Gradio settings (optional)
export GRADIO_SERVER_PORT=7860
```

### Customization

Edit `gradio_app.py` to customize:
- Upload directory path
- Supported file types
- UI theme and layout
- Example questions

## 🧪 Testing

Run the complete test suite using the Jupyter notebook:

```bash
jupyter notebook advanced_rag_system.ipynb
```

Or test individual components:

```bash
python3 ollama_client.py
python3 document_parser.py
python3 metadata_extractor.py
# ... etc
```

## 📊 Performance

- **Processing Speed**: 2,000+ document elements in 10-15 minutes (CPU)
- **Metadata Extraction**: 7x faster with document-level approach
- **Chunk Detection**: 0.01s with regex vs 4s with LLM
- **Retrieval Quality**: 49% improvement with contextual enrichment
- **Query Response**: ~2-5 seconds per question

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ollama**: Local LLM inference
- **Gradio**: Beautiful web interfaces
- **FAISS**: Efficient similarity search
- **LangChain/LlamaIndex**: RAG architecture inspiration
- **Jina AI**: Late Chunking technique
- **Anthropic**: Contextual Enrichment method
- **RAPTOR**: Hierarchical summarization approach

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: Mahdiizadi1998

---

**Made with ❤️ for the Geothermal Energy Community**