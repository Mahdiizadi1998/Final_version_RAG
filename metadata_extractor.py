"""
Universal Geothermal Metadata Extractor
Triple extraction approach: Regex + NLP + LLM
"""

import re
import json
import spacy
from typing import Dict, List, Any, Optional
from ollama_client import ollama_generate


class UniversalGeothermalMetadataExtractor:
    """Extract metadata from geothermal well documents using multiple strategies."""
    
    def __init__(self, llm_model: str = "llama3.1:8b"):
        """
        Initialize extractor with LLM model and regex patterns.
        
        Args:
            llm_model: Ollama model name for LLM extraction
        """
        self.llm_model = llm_model
        
        # Load spaCy model for NLP extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✓ spaCy model loaded")
        except:
            print("⚠ spaCy model not found, NLP extraction disabled")
            self.nlp = None
        
        # Define universal regex patterns for 6 entity types
        self.patterns = {
            'well_name': [
                r'\b[A-Z]{2,4}[-_][A-Z]{1,3}[-_]\d{1,3}[A-Z]?\b',  # ADK-GT-01
                r'\b[A-Z][a-z]+[-_][A-Z]{2,4}[-_]\d{1,3}\b',  # Bergen-GWT-03
                r'\bWell\s+[A-Z\-\d]{3,}\b',  # Well ABC-123
                r'\b[A-Z]{2,}\s*\d{1,3}[A-Z]?\b',  # GT01, ABC123
            ],
            'formation': [
                # Dutch formations
                r'\b(Slochteren|Rotliegend|Zechstein|Buntsandstein|Ten Boer|Muschelkalk)\b',
                # Generic patterns
                r'\b([A-Z][a-z]+)\s+Formation\b',
                r'\b([A-Z][a-z]+)\s+Sandstone\b',
                r'\b([A-Z][a-z]+)\s+Group\b',
                # International formations
                r'\b(Permian|Triassic|Carboniferous|Jurassic)\b',
            ],
            'depth': [
                r'\d+(?:\.\d+)?\s*(?:m|meters?)\s+(?:TVDSS|MD|TVD|MSL)',
                r'(?:depth|Depth):\s*\d+(?:\.\d+)?\s*(?:m|meters?)',
                r'\d+\s*[-–]\s*\d+\s*(?:m|meters?)',
                r'(?:from|From)\s+\d+(?:\.\d+)?\s*(?:m|meters?)',
            ],
            'temperature': [
                r'\d+(?:\.\d+)?°C',
                r'\d+(?:\.\d+)?\s*(?:degrees?\s*)?C(?:elsius)?',
                r'(?:temp|temperature|Temp|Temperature):\s*\d+(?:\.\d+)?',
            ],
            'pressure': [
                r'\d+(?:\.\d+)?\s*bar(?:a)?',
                r'\d+(?:\.\d+)?\s*(?:kPa|MPa|psi)',
                r'(?:pressure|Pressure):\s*\d+(?:\.\d+)?',
            ],
            'test_type': [
                r'\b(PVT|DST|PLT|RFT|MDT)\b',
                r'\b(Production|Injection)\s+Test\b',
                r'\b(Well|Drill|Formation)\s+Test\b',
                r'\b(Pressure|Temperature)\s+Survey\b',
                r'\b(Logging|Coring)\s+Operation\b',
            ]
        }
    
    def extract_all(self, text: str, filename: str = "") -> Dict[str, Any]:
        """
        Extract all metadata using triple approach.
        
        Args:
            text: Document text
            filename: Optional filename for hints
            
        Returns:
            Dictionary with extracted metadata
        """
        # Strategy 1: Regex extraction (fast, universal)
        regex_results = self._extract_with_regex(text)
        
        # Strategy 2: NLP extraction (entity recognition)
        nlp_results = self._extract_with_nlp(text) if self.nlp else {}
        
        # Strategy 3: LLM extraction (slow but comprehensive)
        llm_results = {}
        if not regex_results.get('wells') and len(text) > 500:
            llm_results = self._extract_with_llm_universal(text[:5000])
        
        # Merge results with priority: Regex > NLP > LLM
        merged = self._merge_results(regex_results, nlp_results, llm_results)
        
        # Add filename-based hints
        if filename:
            filename_data = self._extract_from_filename_universal(filename)
            merged.update(filename_data)
        
        # Identify primary well
        if merged.get('wells'):
            merged['primary_well'] = self._identify_primary_well(
                merged['wells'],
                filename
            )
        
        return merged
    
    def _extract_with_regex(self, text: str) -> Dict[str, Any]:
        """Extract entities using regex patterns."""
        results = {}
        
        for entity_type, patterns in self.patterns.items():
            matches = []
            
            for pattern in patterns:
                found = re.findall(pattern, text, re.IGNORECASE)
                matches.extend(found)
            
            # Clean and deduplicate
            if entity_type in ['depth', 'temperature', 'pressure']:
                # Parse numerical values
                results[entity_type] = self._parse_numerical(matches)
            else:
                cleaned = self._clean_entity_list(matches, entity_type)
                results[entity_type] = cleaned
        
        # Map to standard keys
        return {
            'wells': results.get('well_name', []),
            'formations': results.get('formation', []),
            'depth': results.get('depth', {}),
            'temperature': results.get('temperature', {}),
            'pressure': results.get('pressure', {}),
            'test_types': results.get('test_type', [])
        }
    
    def _clean_entity_list(self, entities: List[str], entity_type: str) -> List[str]:
        """
        Clean and validate entity list.
        
        Args:
            entities: Raw extracted entities
            entity_type: Type of entity
            
        Returns:
            Cleaned list (max 10 items)
        """
        if not entities:
            return []
        
        # Normalize
        normalized = []
        for entity in entities:
            if isinstance(entity, tuple):
                entity = entity[0] if entity else ""
            
            entity = str(entity).strip()
            
            # Validate well names (must have letters AND numbers)
            if entity_type == 'well_name':
                if not (re.search(r'[A-Za-z]', entity) and re.search(r'\d', entity)):
                    continue
            
            # Filter false positives
            if entity.upper() in ['TABLE', 'FIGURE', 'PAGE', 'CHAPTER']:
                continue
            
            if entity and entity not in normalized:
                normalized.append(entity)
        
        # Return max 10 entities
        return normalized[:10]
    
    def _parse_numerical(self, values: List[str]) -> Dict[str, Any]:
        """
        Parse numerical values with units.
        
        Returns:
            Dict with min, max, all_values
        """
        numbers = []
        
        for val in values:
            # Extract numeric part
            match = re.search(r'\d+(?:\.\d+)?', str(val))
            if match:
                try:
                    numbers.append(float(match.group()))
                except:
                    pass
        
        if not numbers:
            return {}
        
        return {
            'min': min(numbers),
            'max': max(numbers),
            'all_values': numbers[:10]  # Limit to 10 values
        }
    
    def _extract_with_nlp(self, text: str) -> Dict[str, Any]:
        """Extract entities using spaCy NER."""
        if not self.nlp:
            return {}
        
        # Limit text length for performance
        text = text[:10000]
        doc = self.nlp(text)
        
        potential_wells = []
        
        # Look for organization/facility names that might be wells
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'FAC', 'PRODUCT']:
                # Check if matches well pattern
                if re.search(r'[A-Z]{2,}[-_\s]?\d+', ent.text):
                    potential_wells.append(ent.text)
        
        return {
            'wells': self._clean_entity_list(potential_wells, 'well_name')
        }
    
    def _extract_with_llm_universal(self, text: str) -> Dict[str, Any]:
        """Extract metadata using LLM with JSON output."""
        prompt = f"""Extract geothermal well metadata from this text. Return ONLY valid JSON.

Text: {text}

Return JSON with these fields:
{{
    "wells": ["well names"],
    "formations": ["geological formations"],
    "depth_min": null or number,
    "depth_max": null or number,
    "temperature_min": null or number,
    "temperature_max": null or number,
    "pressure_min": null or number,
    "pressure_max": null or number,
    "test_types": ["test types"]
}}

JSON:"""
        
        try:
            response = ollama_generate(
                model=self.llm_model,
                prompt=prompt,
                temperature=0.1,
                timeout=60
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                # Convert to standard format
                return {
                    'wells': data.get('wells', [])[:10],
                    'formations': data.get('formations', [])[:10],
                    'depth': {
                        'min': data.get('depth_min'),
                        'max': data.get('depth_max')
                    } if data.get('depth_min') or data.get('depth_max') else {},
                    'temperature': {
                        'min': data.get('temperature_min'),
                        'max': data.get('temperature_max')
                    } if data.get('temperature_min') or data.get('temperature_max') else {},
                    'pressure': {
                        'min': data.get('pressure_min'),
                        'max': data.get('pressure_max')
                    } if data.get('pressure_min') or data.get('pressure_max') else {},
                    'test_types': data.get('test_types', [])[:10]
                }
        except Exception as e:
            print(f"    LLM extraction error: {e}")
        
        return {}
    
    def _extract_from_filename_universal(self, filename: str) -> Dict[str, Any]:
        """Extract metadata from filename."""
        results = {}
        
        # Extract well names from filename
        for pattern in self.patterns['well_name']:
            matches = re.findall(pattern, filename)
            if matches:
                results['filename_well'] = matches[0]
                break
        
        # Detect document type
        doc_types = {
            'PVT': r'pvt|pressure.*volume.*temperature',
            'Litholog': r'litho|log|lithology',
            'Well Report': r'well.*report|report.*well',
            'Test Report': r'test.*report|dsr|dst',
            'Geological': r'geo|geology|geological',
        }
        
        for doc_type, pattern in doc_types.items():
            if re.search(pattern, filename, re.IGNORECASE):
                results['doc_type'] = doc_type
                break
        
        return results
    
    def _identify_primary_well(self, wells: List[str], filename: str = "") -> Optional[str]:
        """
        Identify the primary well from list.
        
        Args:
            wells: List of well names
            filename: Optional filename for matching
            
        Returns:
            Primary well name or None
        """
        if not wells:
            return None
        
        # Check if any well appears in filename
        if filename:
            for well in wells:
                if well.lower() in filename.lower():
                    return well
        
        # Otherwise return first well
        return wells[0]
    
    def _merge_results(self, *result_dicts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple extraction results with deduplication.
        
        Priority: First dict > Second dict > Third dict
        """
        merged = {}
        
        for results in result_dicts:
            for key, value in results.items():
                if key not in merged or not merged[key]:
                    merged[key] = value
                elif isinstance(value, list) and isinstance(merged[key], list):
                    # Merge lists and deduplicate
                    combined = merged[key] + value
                    merged[key] = list(dict.fromkeys(combined))[:10]
                elif isinstance(value, dict) and isinstance(merged[key], dict):
                    # Merge dicts
                    merged[key].update({k: v for k, v in value.items() if v is not None})
        
        return merged


if __name__ == "__main__":
    # Test the extractor
    extractor = UniversalGeothermalMetadataExtractor()
    
    test_text = """
    Well ADK-GT-01 was drilled in the Slochteren Formation.
    Depth: 2500 m TVDSS
    Temperature: 85°C
    Pressure: 250 bar
    DST performed successfully.
    """
    
    result = extractor.extract_all(test_text, "ADK-GT-01_PVT_Report.pdf")
    print("Test extraction result:")
    print(json.dumps(result, indent=2))
