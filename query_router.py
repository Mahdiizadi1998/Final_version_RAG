"""
Query Router
Classifies queries and determines optimal retrieval strategy
"""

import re
import json
from typing import Dict, List, Any, Optional
from ollama_client import ollama_generate


class QueryRouter:
    """
    Route queries to appropriate retrieval strategies.
    
    Classifies query type and extracts filters for targeted retrieval.
    """
    
    def __init__(self, llm_model: str = "llama3.1:8b"):
        """
        Initialize query router.
        
        Args:
            llm_model: Ollama model for query classification
        """
        self.llm_model = llm_model
        
        # Regex patterns for entity extraction
        self.well_patterns = [
            r'\b[A-Z]{2,4}[-_][A-Z]{1,3}[-_]\d{1,3}[A-Z]?\b',
            r'\b[A-Z][a-z]+[-_][A-Z]{2,4}[-_]\d{1,3}\b',
            r'\bWell\s+[A-Z\-\d]{3,}\b',
        ]
        
        self.formation_patterns = [
            r'\b(Slochteren|Rotliegend|Zechstein|Buntsandstein)\b',
            r'\b([A-Z][a-z]+)\s+Formation\b',
        ]
        
        print(f"✓ QueryRouter initialized with {llm_model}")
    
    def route(self, query: str) -> Dict[str, Any]:
        """
        Route query and determine retrieval strategy.
        
        Args:
            query: User query
            
        Returns:
            Routing decision dictionary with:
                - query_type: factual, comparison, summary, complex, exploratory
                - strategy: dense, hybrid, graph, raptor
                - filters: metadata filters (wells, formations, etc.)
                - explanation: routing rationale
        """
        print(f"\n{'='*70}")
        print("ROUTING QUERY")
        print(f"{'='*70}")
        print(f"Query: {query}")
        
        # Classify query type using LLM
        query_type = self._classify_query_type(query)
        print(f"\nQuery type: {query_type}")
        
        # Extract entities using regex
        filters = self._extract_filters(query)
        print(f"Filters: {filters}")
        
        # Determine retrieval strategy
        strategy = self._determine_strategy(query_type, filters)
        print(f"Strategy: {strategy}")
        
        routing_decision = {
            'query': query,
            'query_type': query_type,
            'strategy': strategy,
            'filters': filters,
            'explanation': self._get_explanation(query_type, strategy)
        }
        
        print(f"{'='*70}")
        
        return routing_decision
    
    def _classify_query_type(self, query: str) -> str:
        """
        Classify query into type using LLM.
        
        Types:
            - factual: Simple fact retrieval
            - comparison: Comparing multiple entities
            - summary: Requesting overview/summary
            - complex: Multi-step reasoning required
            - exploratory: Open-ended exploration
        """
        prompt = f"""Classify this query into ONE category:

Query: "{query}"

Categories:
- factual: Simple fact retrieval (e.g., "What is the temperature in ADK-GT-01?")
- comparison: Comparing multiple entities (e.g., "Compare ADK-GT-01 and ADK-GT-02")
- summary: Requesting overview/summary (e.g., "Summarize all wells")
- complex: Multi-step reasoning (e.g., "Which wells are suitable for production?")
- exploratory: Open-ended exploration (e.g., "Tell me about geothermal potential")

Respond with ONLY the category word, nothing else."""
        
        try:
            response = ollama_generate(
                model=self.llm_model,
                prompt=prompt,
                temperature=0.0,
                timeout=30
            )
            
            response_lower = response.strip().lower()
            
            for query_type in ['factual', 'comparison', 'summary', 'complex', 'exploratory']:
                if query_type in response_lower:
                    return query_type
            
            return 'factual'  # Default
            
        except Exception as e:
            print(f"  Warning: Classification failed ({e}), defaulting to factual")
            return 'factual'
    
    def _extract_filters(self, query: str) -> Dict[str, List[str]]:
        """
        Extract entity filters from query using regex.
        
        Returns:
            Dictionary with wells, formations, depths, etc.
        """
        filters = {}
        
        # Extract well names
        wells = []
        for pattern in self.well_patterns:
            matches = re.findall(pattern, query)
            wells.extend(matches)
        
        if wells:
            filters['wells'] = list(set(wells))
        
        # Extract formations
        formations = []
        for pattern in self.formation_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if isinstance(matches[0], tuple) if matches else False:
                formations.extend([m for m in matches if m])
            else:
                formations.extend(matches)
        
        if formations:
            filters['formations'] = list(set(formations))
        
        # Extract depth mentions
        depth_pattern = r'\d+\s*(?:m|meters?|ft|feet)'
        depth_matches = re.findall(depth_pattern, query, re.IGNORECASE)
        if depth_matches:
            filters['depth_mentioned'] = True
        
        # Extract temperature mentions
        temp_pattern = r'\d+\s*(?:°C|celsius|degrees?)'
        temp_matches = re.findall(temp_pattern, query, re.IGNORECASE)
        if temp_matches:
            filters['temperature_mentioned'] = True
        
        # Detect document type mentions
        doc_types = {
            'PVT': r'\bpvt\b',
            'Test Report': r'\btest\s+report\b',
            'Litholog': r'\blitholog\b',
        }
        
        for doc_type, pattern in doc_types.items():
            if re.search(pattern, query, re.IGNORECASE):
                filters['doc_type'] = doc_type
                break
        
        return filters
    
    def _determine_strategy(
        self,
        query_type: str,
        filters: Dict[str, Any]
    ) -> str:
        """
        Determine optimal retrieval strategy.
        
        Strategies:
            - dense: Dense vector search (semantic)
            - hybrid: Dense + BM25 (keyword + semantic)
            - graph: Knowledge graph traversal (related docs)
            - raptor: Hierarchical tree search (summaries)
        """
        # Summary queries → RAPTOR (hierarchical summaries)
        if query_type == 'summary':
            return 'raptor'
        
        # Comparison queries → Graph (find related documents)
        if query_type == 'comparison':
            return 'graph'
        
        # Complex queries → Hybrid (keyword + semantic)
        if query_type == 'complex':
            return 'hybrid'
        
        # Queries with specific entities → Hybrid (better filtering)
        if filters.get('wells') or filters.get('formations'):
            return 'hybrid'
        
        # Exploratory queries → RAPTOR (broad to specific)
        if query_type == 'exploratory':
            return 'raptor'
        
        # Default: Dense vector search
        return 'dense'
    
    def _get_explanation(self, query_type: str, strategy: str) -> str:
        """Get human-readable explanation of routing decision."""
        explanations = {
            ('factual', 'dense'): "Simple fact retrieval using semantic search",
            ('factual', 'hybrid'): "Fact retrieval with keyword matching for precision",
            ('comparison', 'graph'): "Finding related documents via knowledge graph",
            ('summary', 'raptor'): "Using hierarchical summaries for overview",
            ('complex', 'hybrid'): "Multi-faceted search combining keywords and semantics",
            ('exploratory', 'raptor'): "Broad to specific exploration using tree structure",
        }
        
        return explanations.get(
            (query_type, strategy),
            f"{query_type} query using {strategy} strategy"
        )


if __name__ == "__main__":
    # Test the router
    print("QueryRouter initialized and ready.")
    router = QueryRouter()
    
    test_query = "What is the temperature in well ADK-GT-01?"
    result = router.route(test_query)
    print(f"\nRouting result: {json.dumps(result, indent=2)}")
