"""
Email Embedder Module

Generates OpenAI embeddings for email chunks.
"""

import os
import json
import yaml
from typing import List, Dict
from openai import OpenAI
from src.db_helper import DBHelper
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class EmailEmbedder:
    """
    Generate embeddings for email chunks using OpenAI API.
    Reads chunks from JSON files, generates embeddings, and prepares for vector DB insertion.
    """
    
    def __init__(self, config_path="config.yaml", db_helper=None):
        """
        Initialize the embedder.
        
        Args:
            config_path: Path to config file
            db_helper: DBHelper instance (optional, will create if not provided)
        """
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        
        # Get embedding config
        embedding_cfg = self.cfg.get("embedding", {})
        self.model = embedding_cfg.get("model", "text-embedding-3-small")
        self.batch_size = embedding_cfg.get("batch_size", 100)
        
        # Get OpenAI API key from environment
        api_key_env = embedding_cfg.get("openai_api_key_env", "OPENAI_API_KEY")
        api_key = os.getenv(api_key_env)
        
        if not api_key:
            raise ValueError(f"OpenAI API key not found in environment variable: {api_key_env}")
        
        self.client = OpenAI(api_key=api_key)
        
        # Database helper
        if db_helper:
            self.db = db_helper
        else:
            db_path = self.cfg["paths"]["db_path"]
            self.db = DBHelper(db_path)
        
        # Chunks directory
        self.chunks_dir = self.cfg["paths"].get("chunks", "data/chunks")
        
        print(f"Embedder initialized:")
        print(f"   Model: {self.model}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Chunks directory: {self.chunks_dir}")
    
    def _load_chunks(self, message_id: str) -> List[Dict]:
        """Load chunks from JSON file for a message"""
        chunk_file = os.path.join(self.chunks_dir, f"{message_id}_chunks.json")
        
        if not os.path.exists(chunk_file):
            print(f"Chunk file not found: {chunk_file}")
            return []
        
        with open(chunk_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both formats: list of chunks or dict with 'chunks' key
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'chunks' in data:
            return data['chunks']
        else:
            print(f"⚠️  Unexpected chunk file format: {chunk_file}")
            return []
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI API.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (each vector is a list of floats)
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            embeddings = [item.embedding for item in response.data]
            return embeddings
            
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            raise
    
    def embed_message(self, message_id: str) -> List[Dict]:
        """
        Generate embeddings for all chunks of a single message.
        
        Args:
            message_id: Email message ID
            
        Returns:
            List of dicts with chunk_id, embedding, text, and metadata
        """
        # Check if already embedded
        if self.db.is_embedded(message_id):
            print(f"Already embedded: {message_id}")
            return []
        
        # Load chunks
        chunks = self._load_chunks(message_id)
        if not chunks:
            return []
        
        print(f"\n{'='*60}")
        print(f"Embedding: {message_id}")
        print(f"{'='*60}")
        print(f"Total chunks: {len(chunks)}")
        
        # Extract texts
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            print(f"   Processing batch {i//self.batch_size + 1} ({len(batch_texts)} chunks)...")
            
            batch_embeddings = self._generate_embeddings(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        # Combine chunks with embeddings
        embedded_chunks = []
        for chunk, embedding in zip(chunks, all_embeddings):
            embedded_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "embedding": embedding,
                "text": chunk["text"],
                "metadata": chunk["metadata"]
            })
        
        # Mark as embedded in database
        self.db.mark_as_embedded(message_id, len(embedded_chunks), self.model)
        
        print(f"Embedded {len(embedded_chunks)} chunks")
        
        return embedded_chunks
    
    def embed_all_messages(self) -> List[Dict]:
        """
        Generate embeddings for all unembedded messages.
        
        Returns:
            List of all embedded chunks across all messages
        """
        # Get unembedded messages
        unembedded = self.db.get_unembedded_messages()
        
        if not unembedded:
            print("All messages are already embedded!")
            return []
        
        print(f"\n📧 Found {len(unembedded)} messages to embed\n")
        
        all_embedded_chunks = []
        
        for idx, message_id in enumerate(unembedded, 1):
            print(f"\n[{idx}/{len(unembedded)}]")
            
            try:
                embedded_chunks = self.embed_message(message_id)
                all_embedded_chunks.extend(embedded_chunks)
                
            except Exception as e:
                print(f"Failed to embed {message_id}: {str(e)}")
                continue
        
        return all_embedded_chunks
    
    def get_stats(self):
        """Print embedding statistics"""
        stats = self.db.get_embedding_stats()
        
        print("\n" + "="*60)
        print("EMBEDDING STATISTICS")
        print("="*60)
        print(f"Chunked emails: {stats['chunked_emails']}")
        print(f"Embedded emails: {stats['embedded_emails']}")
        print(f"Unembedded emails: {stats['unembedded_emails']}")
        print(f"Total embeddings: {stats['total_embeddings']}")
        print(f"Avg embeddings/email: {stats['avg_embeddings_per_email']}")
        
        return stats
    
    def close(self):
        """Close database connection"""
        self.db.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for email chunks")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--msg-id", help="Embed a specific message ID")
    parser.add_argument("--stats", action="store_true", help="Show embedding statistics")
    
    args = parser.parse_args()
    
    embedder = EmailEmbedder(config_path=args.config)
    
    try:
        if args.stats:
            embedder.get_stats()
        elif args.msg_id:
            embedder.embed_message(args.msg_id)
        else:
            embedder.embed_all_messages()
            embedder.get_stats()
    finally:
        embedder.close()
