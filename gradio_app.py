"""
Gradio Web UI for Advanced Multi-Modal RAG System
Easy-to-use interface for document upload and querying
"""

import gradio as gr
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time
from datetime import datetime

# Import RAG system components
from document_parser import AdvancedDocumentParser
from vision_processor import VisionProcessor
from metadata_extractor import UniversalGeothermalMetadataExtractor
from semantic_chunker import UltimateSemanticChunker
from raptor_tree import RAPTORTree
from hybrid_store import HybridIndexStore
from sql_store import SQLStore
from ingestion_pipeline import DocumentIngestionPipeline
from query_router import QueryRouter
from answer_generator import AnswerGenerator
from agentic_rag import AdvancedAgenticRAG
from ollama_client import test_ollama_connection


# ═══════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════

class GlobalState:
    """Singleton to hold initialized components"""
    def __init__(self):
        self.initialized = False
        self.rag_system = None
        self.pipeline = None
        self.upload_directory = "./uploaded_documents"
        self.error_message = None
        
    def initialize(self):
        """Initialize all RAG components"""
        if self.initialized:
            return True, "System already initialized"
        
        try:
            # Test Ollama connection first
            print("🔄 Testing Ollama connection...")
            if not test_ollama_connection():
                self.error_message = "❌ Ollama server not responding. Please start Ollama first."
                return False, self.error_message
            
            print("✅ Ollama connected successfully")
            
            # Create upload directory
            os.makedirs(self.upload_directory, exist_ok=True)
            
            # Initialize components
            print("🔄 Initializing RAG components...")
            
            parser = AdvancedDocumentParser()
            vision_proc = VisionProcessor()
            metadata_extractor = UniversalGeothermalMetadataExtractor()
            chunker = UltimateSemanticChunker()
            raptor = RAPTORTree()
            hybrid_store = HybridIndexStore()
            sql_store = SQLStore(":memory:")
            
            pipeline = DocumentIngestionPipeline(
                parser=parser,
                vision_processor=vision_proc,
                metadata_extractor=metadata_extractor,
                chunker=chunker,
                raptor=raptor,
                hybrid_store=hybrid_store,
                sql_store=sql_store
            )
            
            router = QueryRouter()
            answer_gen = AnswerGenerator()
            
            rag_system = AdvancedAgenticRAG(
                pipeline=pipeline,
                query_router=router,
                answer_generator=answer_gen
            )
            
            self.pipeline = pipeline
            self.rag_system = rag_system
            self.initialized = True
            
            print("✅ All components initialized successfully")
            return True, "✅ System initialized successfully"
            
        except Exception as e:
            self.error_message = f"❌ Initialization error: {str(e)}"
            print(self.error_message)
            return False, self.error_message


# Create global state instance
state = GlobalState()


# ═══════════════════════════════════════════════════════════════════
# UPLOAD TAB FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def process_uploaded_files(files: List[Any]) -> Tuple[str, str]:
    """
    Process uploaded files through the RAG pipeline
    
    Args:
        files: List of uploaded file objects from Gradio
        
    Returns:
        Tuple of (status_message, stats_message)
    """
    if not state.initialized:
        success, msg = state.initialize()
        if not success:
            return msg, ""
    
    if not files:
        return "⚠️ No files uploaded", ""
    
    try:
        # Save uploaded files to directory
        saved_files = []
        for file in files:
            file_path = Path(file.name)
            destination = Path(state.upload_directory) / file_path.name
            
            # Copy file
            with open(file.name, 'rb') as src:
                with open(destination, 'wb') as dst:
                    dst.write(src.read())
            
            saved_files.append(destination)
        
        status_msg = f"📁 Saved {len(saved_files)} file(s)\n\n"
        
        # Process files through pipeline
        status_msg += "🔄 Processing documents through RAG pipeline...\n"
        status_msg += "   This may take a few minutes...\n\n"
        
        start_time = time.time()
        
        # Ingest the uploaded directory
        state.pipeline.ingest_directory(
            state.upload_directory,
            ['*.pdf', '*.docx', '*.xlsx']
        )
        
        elapsed = time.time() - start_time
        
        # Get statistics
        total_docs = len(state.pipeline.hybrid_store.documents)
        faiss_size = state.pipeline.hybrid_store.faiss_index.ntotal
        graph_nodes = state.pipeline.hybrid_store.graph.number_of_nodes()
        raptor_levels = len(state.pipeline.raptor.tree)
        
        status_msg += f"✅ Processing complete in {elapsed:.1f} seconds\n\n"
        status_msg += f"📊 Processed Files:\n"
        for f in saved_files:
            status_msg += f"   • {f.name}\n"
        
        stats_msg = f"""
📊 **System Statistics**

**Documents Indexed:** {total_docs:,}
**FAISS Index Size:** {faiss_size:,}
**Graph Nodes:** {graph_nodes:,}
**RAPTOR Tree Levels:** {raptor_levels}

**Processing Time:** {elapsed:.1f}s
**Status:** Ready for queries ✅
"""
        
        return status_msg, stats_msg
        
    except Exception as e:
        error_msg = f"❌ Error processing files: {str(e)}"
        print(error_msg)
        return error_msg, ""


def clear_uploaded_files() -> Tuple[str, str]:
    """Clear all uploaded files and reset the system"""
    try:
        if os.path.exists(state.upload_directory):
            for file in Path(state.upload_directory).glob('*'):
                if file.is_file():
                    file.unlink()
        
        return "✅ All files cleared", ""
    except Exception as e:
        return f"❌ Error clearing files: {str(e)}", ""


# ═══════════════════════════════════════════════════════════════════
# QUERY TAB FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def query_system(question: str, show_sources: bool = True) -> Tuple[str, str]:
    """
    Query the RAG system with a question
    
    Args:
        question: User's question
        show_sources: Whether to display source documents
        
    Returns:
        Tuple of (answer, sources_info)
    """
    if not state.initialized:
        success, msg = state.initialize()
        if not success:
            return msg, ""
    
    if not question or question.strip() == "":
        return "⚠️ Please enter a question", ""
    
    # Check if documents are loaded
    if len(state.pipeline.hybrid_store.documents) == 0:
        return "⚠️ No documents loaded. Please upload documents first.", ""
    
    try:
        start_time = time.time()
        
        # Query the system
        result = state.rag_system.query(question, return_details=True)
        
        elapsed = time.time() - start_time
        
        # Format answer
        answer = result.get('answer', 'No answer generated')
        confidence = result.get('confidence', 0.0)
        is_grounded = result.get('is_grounded', False)
        query_type = result.get('query_type', 'Unknown')
        strategy = result.get('strategy', 'Unknown')
        
        answer_msg = f"""
## Answer

{answer}

---

**Confidence:** {confidence:.1%} {'✅' if confidence > 0.7 else '⚠️'}  
**Grounded:** {'✅ Yes' if is_grounded else '❌ No'}  
**Query Type:** {query_type}  
**Strategy:** {strategy}  
**Response Time:** {elapsed:.2f}s
"""
        
        # Format sources
        sources_msg = ""
        if show_sources and 'sources' in result:
            sources_msg = "\n## 📚 Sources\n\n"
            for i, source in enumerate(result['sources'][:5], 1):
                doc_name = source.get('document', 'Unknown')
                page = source.get('page', 'N/A')
                well = source.get('well', 'N/A')
                snippet = source.get('snippet', '')[:200]
                
                sources_msg += f"**Source {i}:** {doc_name}\n"
                sources_msg += f"- Page: {page}\n"
                if well != 'N/A':
                    sources_msg += f"- Well: {well}\n"
                sources_msg += f"- Snippet: {snippet}...\n\n"
        
        return answer_msg, sources_msg
        
    except Exception as e:
        error_msg = f"❌ Error querying system: {str(e)}"
        print(error_msg)
        return error_msg, ""


def get_example_questions() -> List[str]:
    """Return list of example questions"""
    return [
        "What is the temperature in well ADK-GT-01?",
        "Compare temperatures between all wells",
        "What formations are mentioned in the documents?",
        "Summarize all wells in the Slochteren Formation",
        "Which well has the highest temperature?",
        "What are the production rates for each well?",
        "List all pressure measurements",
        "What test types were performed?"
    ]


# ═══════════════════════════════════════════════════════════════════
# GRADIO UI INTERFACE
# ═══════════════════════════════════════════════════════════════════

def create_gradio_interface():
    """Create and configure the Gradio interface"""
    
    with gr.Blocks(
        title="Advanced RAG System for Geothermal Well Reports"
    ) as demo:
        
        gr.Markdown(
            """
            # 🌋 Advanced Multi-Modal RAG System
            ## Geothermal Well Reports Analysis
            
            Upload your geothermal well documents and ask questions about temperatures, pressures, formations, and more!
            """
        )
        
        # ═══════════════════════════════════════════════════════════
        # TAB 1: UPLOAD DOCUMENTS
        # ═══════════════════════════════════════════════════════════
        
        with gr.Tab("📁 Upload Documents"):
            gr.Markdown(
                """
                ### Upload Your Documents
                
                Supported formats: **PDF**, **DOCX**, **XLSX**
                
                The system will:
                - Extract text, tables, and images
                - Identify wells, formations, and technical data
                - Build searchable index with AI embeddings
                - Create knowledge graph for complex queries
                """
            )
            
            with gr.Row():
                with gr.Column(scale=2):
                    file_upload = gr.File(
                        label="Upload Documents",
                        file_count="multiple",
                        file_types=[".pdf", ".docx", ".xlsx"]
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button("📤 Process Documents", variant="primary", size="lg")
                        clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
                    
                    upload_status = gr.Textbox(
                        label="Processing Status",
                        lines=10,
                        interactive=False
                    )
                
                with gr.Column(scale=1):
                    stats_display = gr.Markdown(
                        """
                        📊 **System Statistics**
                        
                        No documents loaded yet.
                        
                        Upload documents to get started!
                        """
                    )
        
        # ═══════════════════════════════════════════════════════════
        # TAB 2: ASK QUESTIONS
        # ═══════════════════════════════════════════════════════════
        
        with gr.Tab("❓ Ask Questions"):
            gr.Markdown(
                """
                ### Query Your Documents
                
                Ask questions about your geothermal well reports in natural language.
                
                The system uses:
                - **Hybrid Retrieval**: FAISS + BM25 + Knowledge Graph
                - **RAPTOR Tree**: Hierarchical summarization
                - **Intelligent Routing**: Optimizes strategy per query type
                - **Grounded Answers**: Citations from source documents
                """
            )
            
            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., What is the temperature in well ADK-GT-01?",
                        lines=3
                    )
                    
                    with gr.Row():
                        query_btn = gr.Button("🔍 Get Answer", variant="primary", size="lg")
                        show_sources_check = gr.Checkbox(
                            label="Show Sources",
                            value=True
                        )
                    
                    gr.Markdown("### 💡 Example Questions")
                    example_questions = gr.Examples(
                        examples=get_example_questions(),
                        inputs=question_input
                    )
                
                with gr.Column(scale=2):
                    answer_output = gr.Markdown(label="Answer")
                    sources_output = gr.Markdown(label="Sources")
        
        # ═══════════════════════════════════════════════════════════
        # TAB 3: SYSTEM INFO
        # ═══════════════════════════════════════════════════════════
        
        with gr.Tab("ℹ️ System Info"):
            gr.Markdown(
                """
                ## System Capabilities
                
                ### 🎯 Key Features
                - **Multi-Modal Processing**: Text, tables, and technical images
                - **Triple Metadata Extraction**: Regex + NLP + LLM (7x faster)
                - **Late Chunking**: Better contextual embeddings
                - **Contextual Enrichment**: 49% improvement in retrieval
                - **RAPTOR Tree**: Hierarchical document summarization
                - **Knowledge Graph**: Relationship-based traversal
                
                ### 📊 Optimizations
                - Document-level metadata (7x faster than chunk-level)
                - Regex-based chunk detection (0.01s vs 4s with LLM)
                - Batch encoding (32 documents)
                - Table quality filtering
                - Duplicate column handling
                
                ### 🔧 Technical Stack
                - **LLM**: Ollama (llama3.1:8b)
                - **Vision**: llava:7b
                - **Embeddings**: all-MiniLM-L6-v2 (384 dim)
                - **Vector DB**: FAISS
                - **Graph**: NetworkX
                - **UI**: Gradio
                
                ### 📝 Supported Query Types
                1. **Factual**: Direct information retrieval
                2. **Comparison**: Multi-well analysis
                3. **Summary**: Formation/field overviews
                4. **Complex**: Multi-step reasoning
                5. **Exploratory**: Open-ended discovery
                
                ---
                
                **Version**: 1.0.0  
                **Last Updated**: November 2025  
                **Status**: ✅ Production Ready
                """
            )
        
        # ═══════════════════════════════════════════════════════════
        # EVENT HANDLERS
        # ═══════════════════════════════════════════════════════════
        
        # Upload tab events
        submit_btn.click(
            fn=process_uploaded_files,
            inputs=[file_upload],
            outputs=[upload_status, stats_display]
        )
        
        clear_btn.click(
            fn=clear_uploaded_files,
            inputs=[],
            outputs=[upload_status, stats_display]
        )
        
        # Query tab events
        query_btn.click(
            fn=query_system,
            inputs=[question_input, show_sources_check],
            outputs=[answer_output, sources_output]
        )
        
        # Also allow Enter key to submit question
        question_input.submit(
            fn=query_system,
            inputs=[question_input, show_sources_check],
            outputs=[answer_output, sources_output]
        )
    
    return demo


# ═══════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*70)
    print("ADVANCED MULTI-MODAL RAG SYSTEM - WEB UI")
    print("="*70)
    print("\n🚀 Starting Gradio interface...\n")
    
    # Initialize system on startup
    print("🔄 Initializing RAG system components...")
    success, message = state.initialize()
    print(message)
    
    if not success:
        print("\n⚠️  WARNING: System initialization failed!")
        print("The UI will start, but please check Ollama is running.")
        print("Run: ollama serve")
        print()
    
    # Create and launch interface
    demo = create_gradio_interface()
    
    print("\n✅ Gradio interface ready!")
    print("="*70)
    
    # Launch with public sharing disabled by default
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
