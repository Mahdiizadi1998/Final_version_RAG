"""
Hybrid Index Store
Combines FAISS vector search + BM25 keyword search + Knowledge Graph traversal
"""

import os
import pickle
import numpy as np
import faiss
import networkx as nx
from typing import List, Dict, Any, Optional, Set
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from collections import defaultdict
from tqdm import tqdm


class HybridIndexStore:
    """
    Hybrid retrieval combining:
    1. FAISS - Dense vector search
    2. BM25 - Sparse keyword search
    3. NetworkX - Knowledge graph traversal
    """
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize hybrid index store.
        
        Args:
            embedding_model: SentenceTransformer model name
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        
        # FAISS index for dense vector search
        self.faiss_index = faiss.IndexFlatL2(self.dimension)
        
        # Document storage
        self.documents = []  # All documents
        self.document_texts = []  # For BM25
        
        # BM25 index for sparse keyword search
        self.bm25 = None
        
        # Metadata index (inverted index: field -> value -> doc_ids)
        self.metadata_index = defaultdict(lambda: defaultdict(set))
        
        # Knowledge graph
        self.graph = nx.Graph()
        
        print(f"✓ HybridIndexStore initialized")
        print(f"  - Embedding model: {embedding_model}")
        print(f"  - Vector dimension: {self.dimension}")
        print(f"  - FAISS index: IndexFlatL2")
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> None:
        """
        Add documents to all three indices.
        
        Args:
            documents: List of document dicts with 'text' and 'metadata'
            show_progress: Show progress bar
        """
        if not documents:
            return
        
        start_idx = len(self.documents)
        
        print(f"\nIndexing {len(documents)} documents...")
        
        # Extract texts
        texts = [doc.get('text', '') for doc in documents]
        
        # Compute embeddings
        print("  [1/4] Computing embeddings...")
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Normalize embeddings for FAISS
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        print("  [2/4] Adding to FAISS index...")
        self.faiss_index.add(embeddings.astype('float32'))
        
        # Store documents
        self.documents.extend(documents)
        self.document_texts.extend(texts)
        
        # Rebuild BM25 index
        print("  [3/4] Building BM25 index...")
        tokenized_corpus = [text.lower().split() for text in self.document_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Update metadata index
        print("  [4/4] Updating metadata and graph...")
        for idx, doc in enumerate(documents, start=start_idx):
            metadata = doc.get('metadata', {})
            
            # Index each metadata field
            for field, value in metadata.items():
                if value is not None:
                    # Handle lists
                    if isinstance(value, list):
                        for v in value:
                            self.metadata_index[field][str(v)].add(idx)
                    # Handle dicts
                    elif isinstance(value, dict):
                        for k, v in value.items():
                            if v is not None:
                                self.metadata_index[f"{field}.{k}"][str(v)].add(idx)
                    else:
                        self.metadata_index[field][str(value)].add(idx)
        
        # Build knowledge graph
        self._build_graph(documents, start_idx)
        
        print(f"✓ Indexed {len(documents)} documents")
        print(f"  Total documents: {len(self.documents)}")
        print(f"  Graph nodes: {self.graph.number_of_nodes()}")
        print(f"  Graph edges: {self.graph.number_of_edges()}")
    
    def _build_graph(self, documents: List[Dict[str, Any]], start_idx: int) -> None:
        """
        Build knowledge graph with similarity edges.
        
        Args:
            documents: New documents to add
            start_idx: Starting document index
        """
        # Add nodes with text snippets
        for idx, doc in enumerate(documents, start=start_idx):
            text_snippet = doc.get('text', '')[:200]
            self.graph.add_node(
                idx,
                text=text_snippet,
                metadata=doc.get('metadata', {})
            )
        
        # Compute pairwise similarities for new documents
        if len(documents) > 1:
            # Get embeddings for new documents
            end_idx = start_idx + len(documents)
            new_embeddings = []
            
            for i in range(start_idx, end_idx):
                # Get embedding from FAISS (already normalized)
                vec = self.faiss_index.reconstruct(i)
                new_embeddings.append(vec)
            
            new_embeddings = np.array(new_embeddings)
            
            # Compute similarities
            similarities = np.dot(new_embeddings, new_embeddings.T)
            
            # Add edges for high similarity (> 0.7)
            for i in range(len(documents)):
                for j in range(i + 1, len(documents)):
                    sim = similarities[i, j]
                    if sim > 0.7:
                        self.graph.add_edge(
                            start_idx + i,
                            start_idx + j,
                            weight=float(sim)
                        )
    
    def search_dense(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Dense vector search using FAISS.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of documents with scores
        """
        if self.faiss_index.ntotal == 0:
            return []
        
        # Embed query
        query_embedding = self.embedding_model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True
        )
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS
        k = min(k, self.faiss_index.ntotal)
        distances, indices = self.faiss_index.search(
            query_embedding.astype('float32'),
            k
        )
        
        # Convert to results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['score'] = float(1.0 / (1.0 + dist))  # Convert distance to similarity
                doc['doc_id'] = int(idx)
                results.append(doc)
        
        return results
    
    def search_bm25(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Sparse keyword search using BM25.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of documents with scores
        """
        if self.bm25 is None or len(self.documents) == 0:
            return []
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        k = min(k, len(scores))
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        # Convert to results
        results = []
        for idx in top_k_indices:
            doc = self.documents[idx].copy()
            doc['score'] = float(scores[idx])
            doc['doc_id'] = int(idx)
            results.append(doc)
        
        return results
    
    def search_metadata(self, filters: Dict[str, Any]) -> Set[int]:
        """
        Search by metadata filters.
        
        Args:
            filters: Dictionary of field -> value filters
            
        Returns:
            Set of matching document IDs
        """
        if not filters:
            return set(range(len(self.documents)))
        
        matching_ids = None
        
        for field, value in filters.items():
            # Get documents matching this filter
            field_matches = self.metadata_index.get(field, {}).get(str(value), set())
            
            # Intersect with previous filters
            if matching_ids is None:
                matching_ids = field_matches.copy()
            else:
                matching_ids &= field_matches
        
        return matching_ids if matching_ids else set()
    
    def graph_traverse(
        self,
        seed_doc_ids: List[int],
        max_hops: int = 2,
        max_nodes: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Traverse knowledge graph from seed documents.
        
        Args:
            seed_doc_ids: Starting document IDs
            max_hops: Maximum traversal hops
            max_nodes: Maximum nodes to collect
            
        Returns:
            List of related documents
        """
        if not seed_doc_ids or self.graph.number_of_nodes() == 0:
            return []
        
        visited = set()
        current_level = set(seed_doc_ids)
        
        # Traverse for max_hops
        for hop in range(max_hops):
            next_level = set()
            
            for node in current_level:
                if node not in visited and node in self.graph:
                    visited.add(node)
                    
                    # Add neighbors
                    neighbors = self.graph.neighbors(node)
                    next_level.update(neighbors)
                    
                    if len(visited) >= max_nodes:
                        break
            
            if len(visited) >= max_nodes:
                break
            
            current_level = next_level - visited
        
        # Convert to documents
        results = []
        for doc_id in visited:
            if doc_id < len(self.documents):
                doc = self.documents[doc_id].copy()
                doc['doc_id'] = doc_id
                results.append(doc)
        
        return results[:max_nodes]
    
    def save(self, path_prefix: str) -> None:
        """
        Save all indices to disk.
        
        Args:
            path_prefix: Prefix for output files
        """
        try:
            # Save FAISS index
            faiss.write_index(self.faiss_index, f"{path_prefix}_faiss.index")
            
            # Save documents and metadata
            with open(f"{path_prefix}_data.pkl", 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'document_texts': self.document_texts,
                    'metadata_index': dict(self.metadata_index)
                }, f)
            
            # Save graph
            nx.write_gpickle(self.graph, f"{path_prefix}_graph.gpickle")
            
            print(f"✓ Hybrid index saved to {path_prefix}_*")
            
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def load(self, path_prefix: str) -> None:
        """
        Load all indices from disk.
        
        Args:
            path_prefix: Prefix for input files
        """
        try:
            # Load FAISS index
            self.faiss_index = faiss.read_index(f"{path_prefix}_faiss.index")
            
            # Load documents and metadata
            with open(f"{path_prefix}_data.pkl", 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.document_texts = data['document_texts']
                self.metadata_index = defaultdict(lambda: defaultdict(set), data['metadata_index'])
            
            # Rebuild BM25
            tokenized_corpus = [text.lower().split() for text in self.document_texts]
            self.bm25 = BM25Okapi(tokenized_corpus)
            
            # Load graph
            self.graph = nx.read_gpickle(f"{path_prefix}_graph.gpickle")
            
            print(f"✓ Hybrid index loaded from {path_prefix}_*")
            print(f"  Documents: {len(self.documents)}")
            print(f"  Graph nodes: {self.graph.number_of_nodes()}")
            print(f"  Graph edges: {self.graph.number_of_edges()}")
            
        except Exception as e:
            print(f"Error loading index: {e}")


if __name__ == "__main__":
    # Test the hybrid store
    print("HybridIndexStore initialized and ready.")
    store = HybridIndexStore()
    print("Ready for hybrid retrieval: FAISS + BM25 + Knowledge Graph.")
