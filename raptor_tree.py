"""
RAPTOR Tree Implementation
Recursive Abstractive Processing for Tree-Organized Retrieval
"""

import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
from ollama_client import ollama_generate


class RAPTORTree:
    """
    RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
    Creates hierarchical summaries for multi-level retrieval
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama3.1:8b",
        max_clusters: int = 10
    ):
        """
        Initialize RAPTOR tree.
        
        Args:
            embedding_model: SentenceTransformer model name
            llm_model: Ollama model for summarization
            max_clusters: Maximum clusters per level
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.llm_model = llm_model
        self.max_clusters = max_clusters
        
        # Tree structure: {level: [nodes]}
        self.tree = {}
        self.embeddings = {}  # {level: embeddings array}
        
        print(f"✓ RAPTORTree initialized")
        print(f"  - Embedding model: {embedding_model}")
        print(f"  - LLM model: {llm_model}")
        print(f"  - Max clusters: {max_clusters}")
    
    def build_tree(
        self,
        documents: List[Dict[str, Any]],
        max_levels: int = 3
    ) -> None:
        """
        Build hierarchical tree from documents.
        
        Args:
            documents: List of document dicts with 'text' and 'metadata'
            max_levels: Maximum tree depth
        """
        print(f"\n{'='*70}")
        print("BUILDING RAPTOR TREE")
        print(f"{'='*70}")
        print(f"Documents: {len(documents)}")
        print(f"Max levels: {max_levels}")
        
        # Level 0: Store original documents
        self.tree[0] = documents
        print(f"\nLevel 0: {len(documents)} original documents")
        
        current_level_docs = documents
        
        # Build hierarchical levels
        for level in range(1, max_levels + 1):
            print(f"\nLevel {level}:")
            
            # Embed current level documents
            texts = [doc.get('text', '') for doc in current_level_docs]
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Cluster documents
            print(f"  Clustering {len(current_level_docs)} documents...")
            clusters = self._cluster_documents(embeddings, texts)
            
            if len(clusters) <= 1:
                print(f"  Converged at level {level-1} (only {len(clusters)} cluster)")
                break
            
            print(f"  Created {len(clusters)} clusters")
            
            # Summarize each cluster
            summaries = []
            for cluster_id, cluster_docs in clusters.items():
                print(f"    Summarizing cluster {cluster_id} ({len(cluster_docs)} docs)...")
                summary = self._summarize_cluster(cluster_docs)
                
                summaries.append({
                    'text': summary,
                    'metadata': {
                        'level': level,
                        'cluster_id': cluster_id,
                        'num_children': len(cluster_docs),
                        'source': 'raptor_summary'
                    }
                })
            
            # Store this level
            self.tree[level] = summaries
            print(f"  Level {level}: {len(summaries)} summaries")
            
            # Prepare for next level
            current_level_docs = summaries
        
        print(f"\n{'='*70}")
        print(f"✓ RAPTOR TREE BUILT: {len(self.tree)} levels")
        print(f"{'='*70}")
    
    def _cluster_documents(
        self,
        embeddings: np.ndarray,
        texts: List[str]
    ) -> Dict[int, List[str]]:
        """
        Cluster documents using HDBSCAN.
        
        Args:
            embeddings: Document embeddings
            texts: Document texts
            
        Returns:
            Dictionary mapping cluster_id to list of documents
        """
        if len(embeddings) < 2:
            return {0: texts}
        
        # Use HDBSCAN for clustering
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(2, len(embeddings) // self.max_clusters),
                min_samples=1,
                metric='euclidean'
            )
            cluster_labels = clusterer.fit_predict(embeddings)
        except Exception as e:
            print(f"    Warning: Clustering failed ({e}), using single cluster")
            return {0: texts}
        
        # Group documents by cluster
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label == -1:  # Noise points
                label = max(cluster_labels) + 1 + idx  # Give unique cluster
            
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(texts[idx])
        
        # Limit to max_clusters
        if len(clusters) > self.max_clusters:
            # Keep largest clusters
            sorted_clusters = sorted(
                clusters.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )
            clusters = dict(sorted_clusters[:self.max_clusters])
        
        return clusters
    
    def _summarize_cluster(self, documents: List[str]) -> str:
        """
        Summarize cluster of documents using LLM.
        
        Args:
            documents: List of document texts
            
        Returns:
            Summary text
        """
        # Concatenate documents (limit to 3000 chars total)
        combined_text = ' '.join(documents)
        if len(combined_text) > 3000:
            combined_text = combined_text[:3000] + "..."
        
        # Build summarization prompt
        prompt = f"""Summarize the following geothermal well documents into a comprehensive overview.

Preserve key information:
- Well names and identifiers
- Geological formations
- Depth, temperature, pressure data
- Test results and operations
- Technical findings

Documents:
{combined_text}

Summary:"""
        
        try:
            summary = ollama_generate(
                model=self.llm_model,
                prompt=prompt,
                temperature=0.1,
                timeout=120
            )
            return summary.strip()
        except Exception as e:
            print(f"      Error summarizing: {e}")
            return combined_text[:500]  # Fallback to truncated text
    
    def search_tree(
        self,
        query: str,
        k_per_level: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search tree across all levels.
        
        Args:
            query: Search query
            k_per_level: Top-k results per level
            
        Returns:
            Combined results from all levels
        """
        if not self.tree:
            return []
        
        # Embed query
        query_embedding = self.embedding_model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True
        )[0]
        
        results = []
        
        # Search each level (high to low for broader to specific)
        for level in sorted(self.tree.keys(), reverse=True):
            level_docs = self.tree[level]
            
            if not level_docs:
                continue
            
            # Get texts and embed
            texts = [doc.get('text', '') for doc in level_docs]
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Compute similarities
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1),
                embeddings
            )[0]
            
            # Get top-k indices
            top_k_indices = np.argsort(similarities)[-k_per_level:][::-1]
            
            # Add to results
            for idx in top_k_indices:
                result = level_docs[idx].copy()
                result['score'] = float(similarities[idx])
                result['level'] = level
                results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return results
    
    def save(self, path: str) -> None:
        """
        Save tree to disk.
        
        Args:
            path: File path to save to
        """
        try:
            with open(path, 'wb') as f:
                pickle.dump({
                    'tree': self.tree,
                    'max_clusters': self.max_clusters
                }, f)
            print(f"✓ RAPTOR tree saved to {path}")
        except Exception as e:
            print(f"Error saving tree: {e}")
    
    def load(self, path: str) -> None:
        """
        Load tree from disk.
        
        Args:
            path: File path to load from
        """
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.tree = data['tree']
                self.max_clusters = data['max_clusters']
            print(f"✓ RAPTOR tree loaded from {path}")
            print(f"  Levels: {len(self.tree)}")
        except Exception as e:
            print(f"Error loading tree: {e}")


if __name__ == "__main__":
    # Test the RAPTOR tree
    print("RAPTORTree initialized and ready.")
    raptor = RAPTORTree()
    print("Ready to build hierarchical retrieval trees.")
