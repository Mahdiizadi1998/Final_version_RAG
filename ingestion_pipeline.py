"""
Complete Document Ingestion Pipeline
Orchestrates parsing, metadata extraction, vision processing, chunking, and indexing
"""

import os
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from document_parser import AdvancedDocumentParser
from vision_processor import VisionProcessor
from metadata_extractor import UniversalGeothermalMetadataExtractor
from semantic_chunker import UltimateSemanticChunker
from raptor_tree import RAPTORTree
from hybrid_store import HybridIndexStore


class DocumentIngestionPipeline:
    """
    Complete ingestion pipeline orchestrating all processing stages.
    
    Stages:
    1. Document parsing (multi-modal)
    2. Document-level metadata extraction (once per doc)
    3. Vision processing (image captioning)
    4. Semantic chunking with contextual enrichment
    5. RAPTOR tree building
    6. Hybrid indexing (FAISS + BM25 + Graph)
    """
    
    def __init__(
        self,
        parser: AdvancedDocumentParser,
        vision_proc: VisionProcessor,
        metadata_extractor: UniversalGeothermalMetadataExtractor,
        chunker: UltimateSemanticChunker,
        raptor: RAPTORTree,
        hybrid_store: HybridIndexStore
    ):
        """
        Initialize pipeline with all components.
        
        Args:
            parser: Document parser
            vision_proc: Vision processor
            metadata_extractor: Metadata extractor
            chunker: Semantic chunker
            raptor: RAPTOR tree
            hybrid_store: Hybrid index store
        """
        self.parser = parser
        self.vision_proc = vision_proc
        self.metadata_extractor = metadata_extractor
        self.chunker = chunker
        self.raptor = raptor
        self.hybrid_store = hybrid_store
        
        self.stats = {
            'files_processed': 0,
            'elements_extracted': 0,
            'chunks_created': 0,
            'images_processed': 0,
            'tables_extracted': 0
        }
        
        print(f"✓ DocumentIngestionPipeline initialized")
    
    def ingest_directory(
        self,
        directory: str,
        file_patterns: List[str] = ['*.pdf', '*.docx', '*.xlsx']
    ) -> None:
        """
        Ingest all documents from directory.
        
        Args:
            directory: Directory path
            file_patterns: File patterns to match
        """
        print(f"\n{'='*70}")
        print("STARTING DOCUMENT INGESTION")
        print(f"{'='*70}")
        print(f"Directory: {directory}")
        print(f"Patterns: {file_patterns}")
        
        # Find all matching files
        files = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            print(f"✗ Directory not found: {directory}")
            return
        
        for pattern in file_patterns:
            files.extend(directory_path.glob(pattern))
        
        if not files:
            print(f"✗ No files found matching patterns")
            return
        
        print(f"\nFound {len(files)} files")
        
        # Process each file
        all_chunks = []
        
        for filepath in tqdm(files, desc="Processing files"):
            print(f"\n{'='*70}")
            print(f"File: {filepath.name}")
            print(f"{'='*70}")
            
            try:
                chunks = self._process_single_document(str(filepath))
                all_chunks.extend(chunks)
                
                self.stats['files_processed'] += 1
                self.stats['chunks_created'] += len(chunks)
                
                # Run garbage collection
                gc.collect()
                
            except Exception as e:
                print(f"✗ Error processing {filepath.name}: {e}")
                continue
        
        print(f"\n{'='*70}")
        print("BUILDING RAPTOR TREE")
        print(f"{'='*70}")
        
        # Build RAPTOR tree from all chunks
        if all_chunks:
            self.raptor.build_tree(all_chunks, max_levels=3)
        
        print(f"\n{'='*70}")
        print("INDEXING CHUNKS")
        print(f"{'='*70}")
        
        # Index all chunks in hybrid store
        if all_chunks:
            try:
                self.hybrid_store.add_documents(all_chunks, show_progress=True)
            except Exception as e:
                print(f"✗ Error indexing documents: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        # Print summary
        print(f"\n{'='*70}")
        print("INGESTION COMPLETE")
        print(f"{'='*70}")
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Elements extracted: {self.stats['elements_extracted']}")
        print(f"Images processed: {self.stats['images_processed']}")
        print(f"Tables extracted: {self.stats['tables_extracted']}")
        print(f"Chunks created: {self.stats['chunks_created']}")
        print(f"{'='*70}")
    
    def _process_single_document(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Process single document with hybrid metadata extraction.
        
        OPTIMIZATION: Document-level metadata extracted ONCE (not per chunk)
        
        Args:
            filepath: Path to document
            
        Returns:
            List of chunk dictionaries
        """
        filename = Path(filepath).name
        
        # Step 1: Parse document (multi-strategy)
        print("\n[1/4] Parsing document...")
        elements = self._parse_file(filepath)
        
        if not elements:
            print("  ✗ No elements extracted")
            return []
        
        self.stats['elements_extracted'] += len(elements)
        print(f"  ✓ Extracted {len(elements)} elements")
        
        # Count element types
        text_elements = [e for e in elements if e.type == 'text']
        image_elements = [e for e in elements if e.type == 'image']
        table_elements = [e for e in elements if e.type == 'table']
        
        print(f"    - Text: {len(text_elements)}")
        print(f"    - Images: {len(image_elements)}")
        print(f"    - Tables: {len(table_elements)}")
        
        self.stats['images_processed'] += len(image_elements)
        self.stats['tables_extracted'] += len(table_elements)
        
        # Step 2: DOCUMENT-LEVEL METADATA (ONCE per doc, SLOW but comprehensive)
        print("\n[2/4] Extracting document-level metadata...")
        
        # Get first 20 text elements for context
        context_elements = [e for e in elements if e.type == 'text'][:20]
        context_text = ' '.join([e.content for e in context_elements])[:5000]
        
        # Extract metadata ONCE for entire document
        doc_metadata = self.metadata_extractor.extract_all(context_text, filename)
        
        # Build global metadata structure
        global_metadata = {
            'source': filename,
            'filename': filename,
            'all_wells': doc_metadata.get('wells', []),
            'primary_well': doc_metadata.get('primary_well'),
            'formations': doc_metadata.get('formations', []),
            'test_types': doc_metadata.get('test_types', []),
            'doc_type': doc_metadata.get('doc_type'),
            'temperature': doc_metadata.get('temperature', {}),
            'pressure': doc_metadata.get('pressure', {}),
            'depth': doc_metadata.get('depth', {})
        }
        
        print(f"  ✓ Metadata extracted:")
        print(f"    - Wells: {global_metadata['all_wells']}")
        print(f"    - Primary well: {global_metadata['primary_well']}")
        print(f"    - Formations: {global_metadata['formations']}")
        print(f"    - Doc type: {global_metadata['doc_type']}")
        
        # Step 3: Vision Processing (if images exist)
        if image_elements:
            print(f"\n[3/4] Processing {len(image_elements)} images...")
            
            for img_element in image_elements[:10]:  # Limit to first 10 images
                try:
                    img_path = img_element.content
                    
                    # Build context for image
                    img_context = f"Document: {filename}, Page: {img_element.page}"
                    if global_metadata['primary_well']:
                        img_context += f", Well: {global_metadata['primary_well']}"
                    
                    # Classify image type
                    img_type = self.vision_proc.classify_image_type(img_path)
                    
                    # Generate detailed caption
                    captions = self.vision_proc.caption_technical_image(img_path, img_context)
                    
                    # Replace image path with caption text
                    caption_text = f"[Image: {img_type}]\n"
                    caption_text += f"Description: {captions.get('primary_caption', '')}\n"
                    caption_text += f"Data: {captions.get('data_extraction', '')}"
                    
                    img_element.content = caption_text
                    img_element.metadata['image_type'] = img_type
                    img_element.metadata['original_path'] = img_path
                    
                    print(f"    ✓ Image {img_element.page}: {img_type}")
                    
                except Exception as e:
                    print(f"    ✗ Error processing image: {e}")
        else:
            print("\n[3/4] No images to process")
        
        # Step 4: Semantic Chunking with GLOBAL METADATA
        print("\n[4/4] Semantic chunking...")
        
        # Combine all elements into text
        combined_elements = []
        
        for element in elements:
            if element.type == 'text':
                combined_elements.append(element.content)
            elif element.type == 'table':
                # Convert table to text representation
                table_text = f"[Table from page {element.page}]\n"
                if isinstance(element.content, list):
                    # Format as rows
                    for row in element.content[:10]:  # Limit rows
                        row_text = ' | '.join([f"{k}: {v}" for k, v in row.items()])
                        table_text += row_text + "\n"
                else:
                    table_text += str(element.content)
                combined_elements.append(table_text)
            elif element.type == 'image':
                # Use caption text
                combined_elements.append(element.content)
        
        full_text = '\n\n'.join(combined_elements)
        
        # Add page numbers to metadata
        global_metadata['page'] = 1  # Default page
        
        # Chunk with global metadata passed in
        # Chunker will do FAST chunk-level refinement (regex only, 0.01s per chunk)
        chunks = self.chunker.chunk_text(full_text, global_metadata)
        
        print(f"  ✓ Created {len(chunks)} chunks")
        
        return chunks
    
    def _parse_file(self, filepath: str) -> List[Any]:
        """
        Route to appropriate parser based on extension.
        
        Args:
            filepath: Path to file
            
        Returns:
            List of DocumentElement objects
        """
        ext = Path(filepath).suffix.lower()
        
        if ext == '.pdf':
            return self.parser.parse_pdf(filepath)
        elif ext == '.docx':
            return self.parser.parse_docx(filepath)
        elif ext in ['.xlsx', '.xls']:
            return self.parser.parse_xlsx(filepath)
        else:
            print(f"  ✗ Unsupported file type: {ext}")
            return []
    
    def save_pipeline_state(self, prefix: str = "pipeline") -> None:
        """
        Save all pipeline components.
        
        Args:
            prefix: Prefix for output files
        """
        print(f"\n{'='*70}")
        print("SAVING PIPELINE STATE")
        print(f"{'='*70}")
        
        # Save RAPTOR tree
        self.raptor.save(f"{prefix}_raptor.pkl")
        
        # Save hybrid store
        self.hybrid_store.save(f"{prefix}_hybrid")
        
        print(f"{'='*70}")
        print(f"✓ Pipeline state saved with prefix: {prefix}")
        print(f"{'='*70}")
    
    def load_pipeline_state(self, prefix: str = "pipeline") -> None:
        """
        Load all pipeline components.
        
        Args:
            prefix: Prefix for input files
        """
        print(f"\n{'='*70}")
        print("LOADING PIPELINE STATE")
        print(f"{'='*70}")
        
        # Load RAPTOR tree
        self.raptor.load(f"{prefix}_raptor.pkl")
        
        # Load hybrid store
        self.hybrid_store.load(f"{prefix}_hybrid")
        
        print(f"{'='*70}")
        print(f"✓ Pipeline state loaded from prefix: {prefix}")
        print(f"{'='*70}")


if __name__ == "__main__":
    # Test the pipeline
    print("DocumentIngestionPipeline initialized and ready.")
    print("Ready to ingest documents with full multi-modal processing.")
