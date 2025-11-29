"""
Advanced Agentic RAG System
Complete end-to-end RAG system with query routing and answer generation
"""

from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from ingestion_pipeline import DocumentIngestionPipeline
from query_router import QueryRouter
from answer_generator import AnswerGenerator


class AdvancedAgenticRAG:
    """
    Complete Agentic RAG system with intelligent query routing.
    
    Features:
    - Multi-modal document processing
    - Intelligent query routing
    - Hybrid retrieval (FAISS + BM25 + Graph + RAPTOR)
    - Grounded answer generation with citations
    - Confidence scoring
    """
    
    def __init__(
        self,
        pipeline: DocumentIngestionPipeline,
        router: QueryRouter,
        answer_gen: AnswerGenerator
    ):
        """
        Initialize agentic RAG system.
        
        Args:
            pipeline: Document ingestion pipeline
            router: Query router
            answer_gen: Answer generator
        """
        self.pipeline = pipeline
        self.router = router
        self.answer_gen = answer_gen
        self.console = Console()
        
        print(f"✓ AdvancedAgenticRAG initialized")
        print(f"  - Multi-modal processing enabled")
        print(f"  - Intelligent query routing")
        print(f"  - Hybrid retrieval (4 strategies)")
        print(f"  - Grounded answer generation")
    
    def query(
        self,
        question: str,
        return_details: bool = False
    ) -> Dict[str, Any]:
        """
        Process query end-to-end.
        
        Args:
            question: User question
            return_details: Return detailed routing and retrieval info
            
        Returns:
            Result dictionary with answer, confidence, sources
        """
        self.console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
        self.console.print(f"[bold cyan]PROCESSING QUERY[/bold cyan]")
        self.console.print(f"[bold cyan]{'='*70}[/bold cyan]")
        self.console.print(f"\n[bold]Question:[/bold] {question}\n")
        
        # Step 1: Route query
        self.console.print("[bold yellow]Step 1: Routing Query[/bold yellow]")
        routing = self.router.route(question)
        
        # Step 2: Retrieve based on strategy
        self.console.print(f"\n[bold yellow]Step 2: Retrieving Documents[/bold yellow]")
        self.console.print(f"Strategy: [cyan]{routing['strategy']}[/cyan]")
        
        retrieved_chunks = self._retrieve(question, routing)
        
        self.console.print(f"Retrieved: [green]{len(retrieved_chunks)}[/green] chunks")
        
        # Step 3: Generate answer
        self.console.print(f"\n[bold yellow]Step 3: Generating Answer[/bold yellow]")
        
        result = self.answer_gen.generate_answer(question, retrieved_chunks)
        
        # Step 4: Display results
        self._display_results(question, result, routing)
        
        # Add routing info if requested
        if return_details:
            result['routing'] = routing
            result['retrieved_chunks'] = retrieved_chunks
        
        return result
    
    def _retrieve(
        self,
        query: str,
        routing: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents based on routing strategy.
        
        Args:
            query: User query
            routing: Routing decision
            
        Returns:
            List of retrieved chunks
        """
        strategy = routing['strategy']
        filters = routing.get('filters', {})
        
        # Apply metadata filters if present
        filtered_ids = None
        if filters.get('wells'):
            # Filter by well names
            well_filter = {'primary_well': filters['wells'][0]}
            filtered_ids = self.pipeline.hybrid_store.search_metadata(well_filter)
            self.console.print(f"  Filtered to {len(filtered_ids)} docs by well")
        
        # Execute retrieval based on strategy
        if strategy == 'dense':
            # Dense vector search
            results = self.pipeline.hybrid_store.search_dense(query, k=10)
            
        elif strategy == 'hybrid':
            # Combine dense + BM25
            dense_results = self.pipeline.hybrid_store.search_dense(query, k=10)
            bm25_results = self.pipeline.hybrid_store.search_bm25(query, k=10)
            
            # Merge and rerank
            results = self._merge_results(dense_results, bm25_results)
            
        elif strategy == 'graph':
            # Graph traversal from seed documents
            dense_results = self.pipeline.hybrid_store.search_dense(query, k=5)
            seed_ids = [r['doc_id'] for r in dense_results if 'doc_id' in r]
            
            if seed_ids:
                graph_results = self.pipeline.hybrid_store.graph_traverse(
                    seed_ids,
                    max_hops=2,
                    max_nodes=20
                )
                results = dense_results + graph_results
            else:
                results = dense_results
            
        elif strategy == 'raptor':
            # Hierarchical tree search
            results = self.pipeline.raptor.search_tree(query, k_per_level=3)
            
        else:
            # Default to dense
            results = self.pipeline.hybrid_store.search_dense(query, k=10)
        
        # Apply post-filtering if needed
        if filtered_ids is not None and filters.get('wells'):
            # For RAPTOR, filter by metadata since summaries inherit well metadata
            if strategy == 'raptor':
                target_well = filters['wells'][0]
                results = [
                    r for r in results
                    if self._matches_well_filter(r, target_well)
                ]
            else:
                # For other strategies, filter by doc_id
                results = [
                    r for r in results
                    if r.get('doc_id') in filtered_ids
                ]
        
        return results[:15]  # Limit to top 15
    
    def _matches_well_filter(self, doc: Dict[str, Any], target_well: str) -> bool:
        """
        Check if document matches well filter.
        Checks both primary_well and wells list in metadata.
        
        Args:
            doc: Document dict
            target_well: Target well name
            
        Returns:
            True if document is about the target well
        """
        metadata = doc.get('metadata', {})
        
        # Check primary_well
        if metadata.get('primary_well') == target_well:
            return True
        
        # Check wells list
        wells = metadata.get('wells', [])
        if isinstance(wells, list) and target_well in wells:
            return True
        elif wells == target_well:
            return True
        
        return False
    
    def _merge_results(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge and rerank results from multiple retrievers.
        
        Uses reciprocal rank fusion (RRF).
        """
        # Reciprocal Rank Fusion
        k = 60  # RRF constant
        scores = {}
        doc_map = {}
        
        # Add dense results
        for rank, doc in enumerate(dense_results, 1):
            doc_id = doc.get('doc_id', id(doc))
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
            doc_map[doc_id] = doc
        
        # Add BM25 results
        for rank, doc in enumerate(bm25_results, 1):
            doc_id = doc.get('doc_id', id(doc))
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
        
        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        merged = []
        for doc_id in sorted_ids:
            doc = doc_map[doc_id].copy()
            doc['score'] = scores[doc_id]
            merged.append(doc)
        
        return merged
    
    def _display_results(
        self,
        question: str,
        result: Dict[str, Any],
        routing: Dict[str, Any]
    ) -> None:
        """Display results in a nice format."""
        self.console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
        self.console.print(f"[bold cyan]RESULTS[/bold cyan]")
        self.console.print(f"[bold cyan]{'='*70}[/bold cyan]")
        
        # Answer
        answer = result['answer']
        confidence = result['confidence']
        is_grounded = result['is_grounded']
        
        # Confidence color
        if confidence >= 0.7:
            conf_color = "green"
        elif confidence >= 0.4:
            conf_color = "yellow"
        else:
            conf_color = "red"
        
        self.console.print(Panel(
            answer,
            title="[bold]Answer[/bold]",
            border_style="cyan"
        ))
        
        # Metadata
        self.console.print(f"\n[bold]Metadata:[/bold]")
        self.console.print(f"  Confidence: [{conf_color}]{confidence:.1%}[/{conf_color}]")
        self.console.print(f"  Grounded: [{'green' if is_grounded else 'red'}]{is_grounded}[/{'green' if is_grounded else 'red'}]")
        self.console.print(f"  Query Type: [cyan]{routing['query_type']}[/cyan]")
        self.console.print(f"  Strategy: [cyan]{routing['strategy']}[/cyan]")
        
        # Sources
        sources = result.get('sources', [])
        if sources:
            self.console.print(f"\n[bold]Sources ({len(sources)}):[/bold]")
            for i, source in enumerate(sources[:3], 1):
                self.console.print(f"  {i}. [cyan]{source['document']}[/cyan] (Page {source['page']})")
                if source.get('well'):
                    self.console.print(f"     Well: {source['well']}")
                self.console.print(f"     {source['snippet']}")
        
        self.console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple questions in batch.
        
        Args:
            questions: List of questions
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for i, question in enumerate(questions, 1):
            self.console.print(f"\n[bold magenta]Question {i}/{len(questions)}[/bold magenta]")
            result = self.query(question, return_details=False)
            results.append(result)
        
        return results
    
    def interactive_mode(self) -> None:
        """
        Interactive Q&A loop.
        
        Type 'exit' or 'quit' to exit.
        """
        self.console.print(Panel(
            "[bold cyan]Advanced Agentic RAG - Interactive Mode[/bold cyan]\n\n"
            "Ask questions about your geothermal well documents.\n"
            "Type [bold]'exit'[/bold] or [bold]'quit'[/bold] to exit.\n"
            "Type [bold]'help'[/bold] for tips.",
            border_style="cyan"
        ))
        
        while True:
            try:
                # Get user input
                question = input("\n🔍 Your question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['exit', 'quit', 'q']:
                    self.console.print("\n[bold green]Goodbye![/bold green]")
                    break
                
                if question.lower() == 'help':
                    self._show_help()
                    continue
                
                # Process query
                self.query(question)
                
            except KeyboardInterrupt:
                self.console.print("\n\n[bold yellow]Interrupted[/bold yellow]")
                break
            except Exception as e:
                self.console.print(f"\n[bold red]Error:[/bold red] {e}")
    
    def _show_help(self) -> None:
        """Show help message."""
        help_text = """
[bold]Query Types:[/bold]
  • Factual: "What is the temperature in ADK-GT-01?"
  • Comparison: "Compare ADK-GT-01 and ADK-GT-02"
  • Summary: "Summarize all wells in Slochteren Formation"
  • Complex: "Which wells are suitable for production?"
  • Exploratory: "Tell me about geothermal potential"

[bold]Tips:[/bold]
  • Be specific with well names and formations
  • Include units when asking about measurements
  • Ask for comparisons to see relationships
  • Request summaries for overviews
        """
        
        self.console.print(Panel(help_text, title="Help", border_style="cyan"))


if __name__ == "__main__":
    # Test the agentic RAG system
    print("AdvancedAgenticRAG initialized and ready.")
    print("Ready for end-to-end question answering with intelligent routing.")
