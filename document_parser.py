"""
Advanced Multi-Modal Document Parser
Extracts text, images, and tables from PDF, DOCX, and XLSX files
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import fitz  # PyMuPDF
import pdfplumber
import camelot
import pandas as pd
from docx import Document
from PIL import Image


@dataclass
class DocumentElement:
    """Represents a parsed document element (text, image, or table)."""
    type: str  # 'text', 'image', 'table'
    content: Any
    page: int
    bbox: Optional[Tuple[float, float, float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


class AdvancedDocumentParser:
    """Multi-strategy document parser for PDF, DOCX, and XLSX files."""
    
    def __init__(self, output_dir: str = "parsed_elements"):
        """
        Initialize parser with output directory for extracted elements.
        
        Args:
            output_dir: Directory to save extracted images and tables
        """
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.tables_dir = self.output_dir / "tables"
        
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Parser initialized: {self.output_dir}")
    
    def parse_pdf(self, pdf_path: str) -> List[DocumentElement]:
        """
        Parse PDF using multi-strategy extraction.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of DocumentElement objects
        """
        elements = []
        pdf_path = str(pdf_path)
        
        print(f"\nParsing PDF: {Path(pdf_path).name}")
        
        # Strategy 1: PyMuPDF for text and images
        print("  [1/3] Extracting text and images (PyMuPDF)...")
        text_elements, image_elements = self._extract_with_pymupdf(pdf_path)
        elements.extend(text_elements)
        elements.extend(image_elements)
        print(f"    ✓ {len(text_elements)} text blocks, {len(image_elements)} images")
        
        # Strategy 2: Camelot for tables
        print("  [2/3] Extracting tables (Camelot)...")
        table_elements = self._extract_tables_camelot(pdf_path)
        elements.extend(table_elements)
        print(f"    ✓ {len(table_elements)} tables")
        
        # Strategy 3: PDFPlumber as fallback
        print("  [3/3] Fallback extraction (PDFPlumber)...")
        fallback_elements = self._extract_with_pdfplumber(pdf_path)
        elements.extend(fallback_elements)
        print(f"    ✓ {len(fallback_elements)} additional elements")
        
        return elements
    
    def _extract_with_pymupdf(self, pdf_path: str) -> Tuple[List[DocumentElement], List[DocumentElement]]:
        """Extract text blocks and images using PyMuPDF."""
        text_elements = []
        image_elements = []
        
        try:
            doc = fitz.open(pdf_path)
            pdf_name = Path(pdf_path).stem
            
            for page_num, page in enumerate(doc, start=1):
                # Extract text blocks with bounding boxes
                blocks = page.get_text("blocks")
                for block in blocks:
                    if len(block) >= 5:
                        bbox = block[:4]
                        text = block[4].strip()
                        
                        if text and len(text) > 10:  # Filter very short text
                            element = DocumentElement(
                                type="text",
                                content=text,
                                page=page_num,
                                bbox=bbox,
                                metadata={"source": "pymupdf"}
                            )
                            text_elements.append(element)
                
                # Extract images
                image_list = page.get_images(full=True)
                for img_index, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Save image
                        image_filename = f"{pdf_name}_page{page_num}_img{img_index}.{image_ext}"
                        image_path = self.images_dir / image_filename
                        
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        element = DocumentElement(
                            type="image",
                            content=str(image_path),
                            page=page_num,
                            metadata={
                                "source": "pymupdf",
                                "format": image_ext,
                                "size": len(image_bytes)
                            }
                        )
                        image_elements.append(element)
                        
                    except Exception as e:
                        print(f"      Warning: Could not extract image {img_index}: {e}")
            
            doc.close()
            
        except Exception as e:
            print(f"    Error with PyMuPDF extraction: {e}")
        
        return text_elements, image_elements
    
    def _extract_tables_camelot(self, pdf_path: str) -> List[DocumentElement]:
        """Extract tables using Camelot with quality filters."""
        table_elements = []
        pdf_name = Path(pdf_path).stem
        
        try:
            # Try lattice method first (better for bordered tables)
            try:
                tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            except:
                # Fall back to stream method
                tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
            
            for idx, table in enumerate(tables):
                try:
                    # Get DataFrame and make a copy
                    df = table.df.copy()
                    
                    # CRITICAL: Fix duplicate columns IMMEDIATELY
                    new_columns = []
                    column_counts = {}
                    
                    for i, col in enumerate(df.columns):
                        # Replace empty column names
                        if col == '' or pd.isna(col):
                            col = f'col_{i}'
                        
                        # Track duplicates and add suffix
                        if col in column_counts:
                            column_counts[col] += 1
                            new_col = f"{col}_{column_counts[col]}"
                        else:
                            column_counts[col] = 0
                            new_col = col
                        
                        new_columns.append(new_col)
                    
                    df.columns = new_columns
                    
                    # Apply quality filters
                    if df.shape[0] < 2 or df.shape[1] < 2:
                        continue  # Too small
                    
                    # Check non-empty cells percentage
                    non_empty = df.astype(str).replace('', pd.NA).notna().sum().sum()
                    total_cells = df.shape[0] * df.shape[1]
                    if non_empty / total_cells < 0.3:
                        continue  # Less than 30% non-empty
                    
                    # Check accuracy
                    if hasattr(table, 'accuracy') and table.accuracy < 50:
                        continue  # Low accuracy
                    
                    # Filter garbled tables (chromatogram labels, scattered units)
                    text_content = ' '.join(df.astype(str).values.flatten())
                    garbled_indicators = [
                        r'mAU\s+min',  # Chromatogram labels
                        r'\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+',  # Scattered numbers
                        r'^\d+$'  # All numeric single digits
                    ]
                    
                    if any(re.search(pattern, text_content) for pattern in garbled_indicators):
                        # Check if >50% are single-character cells
                        single_char = (df.astype(str).str.len() == 1).sum().sum()
                        if single_char / total_cells > 0.5:
                            continue
                    
                    # Save as CSV
                    table_filename = f"{pdf_name}_table{idx}.csv"
                    table_path = self.tables_dir / table_filename
                    df.to_csv(table_path, index=False)
                    
                    # Convert to dict records
                    table_dict = df.to_dict('records')
                    
                    element = DocumentElement(
                        type="table",
                        content=table_dict,
                        page=table.page,
                        metadata={
                            "source": "camelot",
                            "csv_path": str(table_path),
                            "shape": df.shape,
                            "accuracy": getattr(table, 'accuracy', None)
                        }
                    )
                    table_elements.append(element)
                    
                except Exception as e:
                    print(f"      Warning: Could not process table {idx}: {e}")
        
        except Exception as e:
            print(f"    Error with Camelot extraction: {e}")
        
        return table_elements
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> List[DocumentElement]:
        """Extract tables using PDFPlumber as fallback."""
        elements = []
        pdf_name = Path(pdf_path).stem
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    tables = page.extract_tables()
                    
                    for table_idx, table in enumerate(tables):
                        if not table or len(table) < 2:
                            continue
                        
                        try:
                            # Convert to DataFrame
                            df = pd.DataFrame(table[1:], columns=table[0])
                            
                            # CRITICAL: Fix duplicate columns
                            new_columns = []
                            column_counts = {}
                            
                            for i, col in enumerate(df.columns):
                                if col is None or col == '':
                                    col = f'col_{i}'
                                
                                if col in column_counts:
                                    column_counts[col] += 1
                                    new_col = f"{col}_{column_counts[col]}"
                                else:
                                    column_counts[col] = 0
                                    new_col = col
                                
                                new_columns.append(new_col)
                            
                            df.columns = new_columns
                            
                            # Quality filters
                            if df.shape[0] < 2 or df.shape[1] < 2:
                                continue
                            
                            # Save and create element
                            table_filename = f"{pdf_name}_pdfplumber_page{page_num}_table{table_idx}.csv"
                            table_path = self.tables_dir / table_filename
                            df.to_csv(table_path, index=False)
                            
                            element = DocumentElement(
                                type="table",
                                content=df.to_dict('records'),
                                page=page_num,
                                metadata={
                                    "source": "pdfplumber",
                                    "csv_path": str(table_path),
                                    "shape": df.shape
                                }
                            )
                            elements.append(element)
                            
                        except Exception as e:
                            print(f"      Warning: PDFPlumber table error: {e}")
        
        except Exception as e:
            print(f"    Error with PDFPlumber extraction: {e}")
        
        return elements
    
    def parse_xlsx(self, xlsx_path: str) -> List[DocumentElement]:
        """
        Parse Excel file.
        
        Args:
            xlsx_path: Path to Excel file
            
        Returns:
            List of DocumentElement objects (one per sheet)
        """
        elements = []
        xlsx_path = str(xlsx_path)
        xlsx_name = Path(xlsx_path).stem
        
        print(f"\nParsing XLSX: {Path(xlsx_path).name}")
        
        try:
            excel_file = pd.ExcelFile(xlsx_path)
            
            for sheet_idx, sheet_name in enumerate(excel_file.sheet_names):
                df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
                
                if df.empty:
                    continue
                
                # Save as CSV
                table_filename = f"{xlsx_name}_sheet{sheet_idx}_{sheet_name}.csv"
                table_path = self.tables_dir / table_filename
                df.to_csv(table_path, index=False)
                
                element = DocumentElement(
                    type="table",
                    content=df.to_dict('records'),
                    page=sheet_idx + 1,
                    metadata={
                        "source": "excel",
                        "sheet_name": sheet_name,
                        "csv_path": str(table_path),
                        "shape": df.shape
                    }
                )
                elements.append(element)
            
            print(f"  ✓ {len(elements)} sheets extracted")
            
        except Exception as e:
            print(f"  Error parsing XLSX: {e}")
        
        return elements
    
    def parse_docx(self, docx_path: str) -> List[DocumentElement]:
        """
        Parse Word document.
        
        Args:
            docx_path: Path to DOCX file
            
        Returns:
            List with single DocumentElement containing all text
        """
        elements = []
        docx_path = str(docx_path)
        
        print(f"\nParsing DOCX: {Path(docx_path).name}")
        
        try:
            doc = Document(docx_path)
            
            # Extract all paragraph text
            paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
            full_text = '\n\n'.join(paragraphs)
            
            if full_text:
                element = DocumentElement(
                    type="text",
                    content=full_text,
                    page=1,
                    metadata={
                        "source": "docx",
                        "paragraph_count": len(paragraphs)
                    }
                )
                elements.append(element)
            
            print(f"  ✓ {len(paragraphs)} paragraphs extracted")
            
        except Exception as e:
            print(f"  Error parsing DOCX: {e}")
        
        return elements


if __name__ == "__main__":
    # Test the parser
    parser = AdvancedDocumentParser()
    print("Document parser initialized and ready.")
