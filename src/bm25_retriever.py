"""
BM25 retriever module

BM25-based keyword retrieval for email chunks.
Complements dense embedding retrieval for hybrid search.
"""

from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
import re


class BM25Retriever:

    def __init__(self, chunks: List[Dict]):
        """
        Initialize BM25 retriever with chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'chunk_id'
        """
        self.chunks = chunks
        self.chunk_ids = [chunk.get("chunk_id", "") for chunk in chunks]
        
        # tokenize chunks for bm25
        tokenized_chunks = [self._tokenize(chunk.get("text", "")) for chunk in chunks]
        
        # initialize bm25
        self.bm25 = BM25Okapi(tokenized_chunks)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # simple tokenization: lowercase, split on whitespace and punctuation
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search chunks using BM25.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries with chunk_id, text, score, metadata
        """
        tokenized_query = self._tokenize(query)
        
        scores = self.bm25.get_scores(tokenized_query)
        
        # get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0: # only include positive scores
                chunk = self.chunks[idx]
                results.append({
                    "chunk_id": chunk.get("chunk_id", ""),
                    "text": chunk.get("text", ""),
                    "score": float(scores[idx]),
                    "metadata": chunk.get("metadata", {})
                })
        
        return results
    
    def get_scores(self, query: str) -> List[float]:
        """
        Get BM25 scores for all chunks.
        
        Args:
            query: Search query string
            
        Returns:
            List of scores for all chunks
        """
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        return scores.tolist()

