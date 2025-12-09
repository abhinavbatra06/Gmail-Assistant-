"""
Vector Database Module using ChromaDB

Stores and retrieves email chunk embeddings for semantic search.
"""

import os
import yaml
import chromadb
from typing import List, Dict, Optional
from chromadb.config import Settings


class EmailVectorDB:
    """
    Manages email chunk embeddings in ChromaDB for semantic search.
    """
    
    def __init__(self, config_path="config.yaml"):
        """
        Initialize ChromaDB vector store.
        
        Args:
            config_path: Path to config file
        """
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        
        # Get vector store config
        vs_cfg = self.cfg.get("vectorstore", {})
        self.persist_dir = vs_cfg.get("persist_directory", "data/vector_index")
        self.collection_name = vs_cfg.get("collection_name", "email_chunks")
        self.distance_metric = vs_cfg.get("distance_metric", "cosine")
        
        # Create persist directory if not exists
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric}
        )
        
        print(f"Vector DB initialized:")
        print(f"   Collection: {self.collection_name}")
        print(f"   Persist directory: {self.persist_dir}")
        print(f"   Distance metric: {self.distance_metric}")
        print(f"   Current vectors: {self.collection.count()}")
    
    def add_chunks(self, chunks: List[Dict], skip_existing=True):
        """
        Add chunks with embeddings to the vector database.
        
        Args:
            chunks: List of dicts with keys: chunk_id, embedding, text, metadata
            skip_existing: If True, skip chunks that already exist in the collection
        """
        if not chunks:
            return
        
        # Check for existing chunks if skip_existing is True
        if skip_existing:
            existing_ids = set()
            try:
                # Get all existing chunk IDs for these message_ids
                chunk_ids_to_check = [c["chunk_id"] for c in chunks]
                existing = self.collection.get(ids=chunk_ids_to_check)
                existing_ids = set(existing["ids"])
            except:
                # If error, proceed without checking (collection might be empty)
                pass
            
            # Filter out existing chunks
            chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
            
            if not chunks:
                print("⏭️  All chunks already exist in vector DB")
                return
            
            if len(existing_ids) > 0:
                print(f"⏭️  Skipping {len(existing_ids)} existing chunks")
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            ids.append(chunk["chunk_id"])
            embeddings.append(chunk["embedding"])
            documents.append(chunk["text"])
            
            # Prepare metadata (ChromaDB requires all values to be strings, ints, floats, or bools)
            metadata = chunk["metadata"].copy()
            
            # Convert any nested objects to strings
            for key, value in metadata.items():
                if isinstance(value, (list, dict)):
                    metadata[key] = str(value)
            
            metadatas.append(metadata)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"✅ Added {len(chunks)} chunks to vector DB")
    
    def search(self, 
               query_embedding: List[float], 
               n_results: int = 5,
               where: Optional[Dict] = None,
               where_document: Optional[Dict] = None) -> Dict:
        """
        Semantic search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Metadata filter (e.g., {"from": {"$contains": "nyu.edu"}})
            where_document: Document text filter
            
        Returns:
            Dict with ids, documents, metadatas, distances
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document
        )
        
        return results
    
    def get_by_id(self, chunk_ids: List[str]) -> Dict:
        """
        Retrieve chunks by their IDs.
        
        Args:
            chunk_ids: List of chunk IDs
            
        Returns:
            Dict with ids, documents, metadatas, embeddings
        """
        return self.collection.get(ids=chunk_ids)
    
    def delete_by_message_id(self, message_id: str):
        """
        Delete all chunks for a specific email message.
        
        Args:
            message_id: Email message ID
        """
        # Query for all chunks with this message_id
        results = self.collection.get(
            where={"message_id": message_id}
        )
        
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            print(f"Deleted {len(results['ids'])} chunks for message {message_id}")
    
    def count(self) -> int:
        """Get total number of vectors in collection"""
        return self.collection.count()
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        count = self.collection.count()
        
        # Get sample to analyze metadata
        sample = self.collection.get(limit=min(10, count)) if count > 0 else {"metadatas": []}
        
        # Count unique message_ids
        message_ids = set()
        if sample["metadatas"]:
            message_ids = {m.get("message_id") for m in sample["metadatas"] if m.get("message_id")}
        
        return {
            "total_chunks": count,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_dir,
            "sample_message_ids": len(message_ids)
        }
    
    def reset_collection(self):
        """Delete and recreate the collection (USE WITH CAUTION!)"""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric}
        )
        print(f"Collection '{self.collection_name}' reset")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vector database operations")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--stats", action="store_true", help="Show collection statistics")
    parser.add_argument("--reset", action="store_true", help="Reset collection (delete all)")
    
    args = parser.parse_args()
    
    db = EmailVectorDB(config_path=args.config)
    
    if args.stats:
        stats = db.get_stats()
        print("\n" + "="*60)
        print("VECTOR DATABASE STATISTICS")
        print("="*60)
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    elif args.reset:
        confirm = input("This will delete all vectors. Type 'yes' to confirm: ")
        if confirm.lower() == 'yes':
            db.reset_collection()
        else:
            print("Reset cancelled")
    
    else:
        print("\nUse --stats to view statistics or --reset to clear the collection")
