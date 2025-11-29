"""
Vision Processor
Uses Vision-Language Model (VLM) for technical image captioning and classification
"""

import os
from typing import Dict, List, Optional
from pathlib import Path
from ollama_client import ollama_generate


class VisionProcessor:
    """Process images using vision-language models for captioning and classification."""
    
    def __init__(self, model: str = "llava:7b"):
        """
        Initialize vision processor with VLM model.
        
        Args:
            model: Ollama vision model name (default: llava:7b)
        """
        self.model = model
        print(f"✓ VisionProcessor initialized with model: {model}")
    
    def caption_technical_image(
        self,
        image_path: str,
        context: str = ""
    ) -> Dict[str, str]:
        """
        Generate detailed captions for technical images.
        
        Args:
            image_path: Path to image file
            context: Optional document context (well name, page, etc.)
            
        Returns:
            Dictionary with 'primary_caption' and 'data_extraction' keys
        """
        if not os.path.exists(image_path):
            return {
                'primary_caption': f"Error: Image not found at {image_path}",
                'data_extraction': ""
            }
        
        # Prompt 1: Primary description
        primary_prompt = f"""You are analyzing a technical image from a geothermal well document.

Context: {context if context else 'Not provided'}

Describe this image in detail:
1. Type of visualization (plot, chart, diagram, photo, schematic, etc.)
2. Axes labels and units (if applicable)
3. Key trends, patterns, or features
4. Important numerical values or ranges
5. Any annotations, legends, or labels
6. Technical significance

Provide a clear, technical description suitable for retrieval and question-answering."""
        
        # Prompt 2: Data extraction
        data_prompt = f"""You are extracting data from a technical image.

Context: {context if context else 'Not provided'}

Extract ALL numerical data, labels, and text visible in this image:
- All numbers with their units
- All axis labels and scales
- All legend entries
- All annotations and text
- Any measurements or values

List everything in a structured format."""
        
        results = {}
        
        # Generate primary caption
        try:
            primary_caption = ollama_generate(
                model=self.model,
                prompt=primary_prompt,
                images=[image_path],
                temperature=0.1,
                timeout=120
            )
            results['primary_caption'] = primary_caption.strip()
        except Exception as e:
            results['primary_caption'] = f"Error generating caption: {e}"
        
        # Generate data extraction
        try:
            data_extraction = ollama_generate(
                model=self.model,
                prompt=data_prompt,
                images=[image_path],
                temperature=0.1,
                timeout=120
            )
            results['data_extraction'] = data_extraction.strip()
        except Exception as e:
            results['data_extraction'] = f"Error extracting data: {e}"
        
        return results
    
    def classify_image_type(self, image_path: str) -> str:
        """
        Classify image into predefined categories.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Category string: plot, schematic, table, photo, map, chart, other
        """
        if not os.path.exists(image_path):
            return "error"
        
        prompt = """Classify this image into ONE of these categories:
- plot: Line plots, scatter plots, time series
- schematic: Technical diagrams, flowcharts, schematics
- table: Data tables, spreadsheets
- photo: Photographs, core samples, equipment photos
- map: Maps, geological cross-sections, spatial data
- chart: Bar charts, pie charts, histograms
- other: Anything else

Respond with ONLY the category word, nothing else."""
        
        try:
            response = ollama_generate(
                model=self.model,
                prompt=prompt,
                images=[image_path],
                temperature=0.0,
                timeout=60
            )
            
            # Extract category
            category = response.strip().lower()
            valid_categories = ['plot', 'schematic', 'table', 'photo', 'map', 'chart', 'other']
            
            for valid in valid_categories:
                if valid in category:
                    return valid
            
            return 'other'
            
        except Exception as e:
            print(f"Error classifying image: {e}")
            return 'other'


if __name__ == "__main__":
    # Test the vision processor
    print("VisionProcessor initialized and ready.")
    processor = VisionProcessor()
    print("Ready to process images with llava:7b model.")
