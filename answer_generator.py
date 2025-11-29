"""
Answer Generator
Generates grounded answers with citations from retrieved context
"""

import re
from typing import Dict, List, Any, Optional
from ollama_client import ollama_generate


class AnswerGenerator:
    """
    Generate grounded answers with strict citation requirements.
    
    Ensures answers are factual and traceable to source documents.
    """
    
    def __init__(self, llm_model: str = "llama3.1:8b"):
        """
        Initialize answer generator.
        
        Args:
            llm_model: Ollama model for answer generation
        """
        self.llm_model = llm_model
        print(f"✓ AnswerGenerator initialized with {llm_model}")
    
    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        max_context_chars: int = 3000
    ) -> Dict[str, Any]:
        """
        Generate grounded answer from retrieved chunks.
        
        Args:
            query: User query
            retrieved_chunks: Retrieved document chunks
            max_context_chars: Maximum context length
            
        Returns:
            Dictionary with:
                - answer: Generated answer text
                - confidence: Confidence score (0-1)
                - sources: List of source chunks used
                - is_grounded: Whether answer is grounded in context
        """
        if not retrieved_chunks:
            return {
                'answer': "I don't have enough information to answer this question.",
                'confidence': 0.0,
                'sources': [],
                'is_grounded': False
            }
        
        # Build context from top chunks
        context = self._build_context(retrieved_chunks, max_context_chars)
        
        # Create grounded answering prompt
        prompt = self._create_prompt(query, context)
        
        # Generate answer
        try:
            response = ollama_generate(
                model=self.llm_model,
                prompt=prompt,
                temperature=0.1,
                timeout=120
            )
            
            # Extract answer and confidence
            answer_text = response.strip()
            confidence = self._extract_confidence(answer_text)
            
            # Verify grounding
            is_grounded = self._verify_grounding(answer_text, context)
            
            # Extract sources
            sources = self._extract_sources(retrieved_chunks[:5])
            
            return {
                'answer': answer_text,
                'confidence': confidence,
                'sources': sources,
                'is_grounded': is_grounded,
                'num_chunks_used': len(retrieved_chunks)
            }
            
        except Exception as e:
            return {
                'answer': f"Error generating answer: {e}",
                'confidence': 0.0,
                'sources': [],
                'is_grounded': False
            }
    
    def _build_context(
        self,
        chunks: List[Dict[str, Any]],
        max_chars: int
    ) -> str:
        """
        Build context string from retrieved chunks.
        
        Args:
            chunks: Retrieved chunks
            max_chars: Maximum characters
            
        Returns:
            Context string
        """
        context_parts = []
        total_chars = 0
        
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get('text', '')
            metadata = chunk.get('metadata', {})
            
            # Build source line
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', '?')
            well = metadata.get('primary_well', metadata.get('chunk_wells', [''])[0] if metadata.get('chunk_wells') else '')
            
            chunk_header = f"\n[Source {i}: {source}"
            if page:
                chunk_header += f", Page {page}"
            if well:
                chunk_header += f", Well {well}"
            chunk_header += "]\n"
            
            chunk_text = chunk_header + text
            
            if total_chars + len(chunk_text) > max_chars:
                break
            
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)
        
        return '\n'.join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create grounded answering prompt with citation requirements.
        """
        prompt = f"""You are an expert geothermal engineer assistant. Answer the question based ONLY on the provided context.

STRICT RULES:
1. Only use information explicitly stated in the context
2. If the context doesn't contain the answer, say "The provided documents do not contain this information"
3. Include specific citations (well names, values, document names)
4. Be precise with numerical values and units
5. If you're uncertain, say so and explain why
6. At the end, rate your confidence (0-100%)

Context:
{context}

Question: {query}

Answer (with citations and confidence):"""
        
        return prompt
    
    def _extract_confidence(self, answer: str) -> float:
        """
        Extract confidence score from answer text.
        
        Looks for patterns like "Confidence: 85%" or "85% confident"
        """
        # Look for percentage patterns
        patterns = [
            r'confidence[:\s]+(\d+)%',
            r'(\d+)%\s*confident',
            r'confidence[:\s]+(\d+)/100',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                try:
                    score = int(match.group(1))
                    return score / 100.0
                except:
                    pass
        
        # Default: high confidence if answer is substantial
        if len(answer) > 100 and "don't" not in answer.lower():
            return 0.7
        elif "don't" in answer.lower() or "cannot" in answer.lower():
            return 0.1
        else:
            return 0.5
    
    def _verify_grounding(self, answer: str, context: str) -> bool:
        """
        Verify that answer is grounded in provided context.
        
        Checks if key facts in answer appear in context.
        """
        if "don't have" in answer.lower() or "not contain" in answer.lower():
            return True  # Correctly stating lack of info
        
        # Extract key terms from answer (exclude common words)
        answer_lower = answer.lower()
        context_lower = context.lower()
        
        # Look for specific entities and numbers
        # Extract numbers with units
        answer_values = set(re.findall(r'\d+(?:\.\d+)?\s*[a-z°]+', answer_lower))
        
        if answer_values:
            # Check if at least 50% of specific values appear in context
            found_values = sum(1 for val in answer_values if val in context_lower)
            if found_values / len(answer_values) < 0.5:
                return False
        
        # Extract well names
        well_pattern = r'\b[A-Z]{2,4}[-_][A-Z]{1,3}[-_]\d{1,3}\b'
        answer_wells = set(re.findall(well_pattern, answer))
        
        if answer_wells:
            # All mentioned wells should be in context
            context_wells = set(re.findall(well_pattern, context))
            if not answer_wells.issubset(context_wells):
                return False
        
        return True
    
    def _extract_sources(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Extract source information from chunks.
        
        Returns:
            List of source dictionaries
        """
        sources = []
        
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            
            source_info = {
                'document': metadata.get('source', 'Unknown'),
                'page': metadata.get('page', '?'),
                'well': metadata.get('primary_well', ''),
                'snippet': chunk.get('text', '')[:150] + '...'
            }
            
            sources.append(source_info)
        
        return sources


if __name__ == "__main__":
    # Test the answer generator
    print("AnswerGenerator initialized and ready.")
    generator = AnswerGenerator()
    
    test_chunks = [
        {
            'text': 'Well ADK-GT-01 has a temperature of 85°C at 2500m depth.',
            'metadata': {'source': 'test.pdf', 'page': 1, 'primary_well': 'ADK-GT-01'}
        }
    ]
    
    result = generator.generate_answer("What is the temperature?", test_chunks)
    print(f"\nTest result:")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Grounded: {result['is_grounded']}")
