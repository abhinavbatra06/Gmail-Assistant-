"""
Quick Search Test

Test semantic search on your email vector database.
"""

from openai import OpenAI
from src.vector_db import EmailVectorDB
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    print("="*60)
    print("QUICK SEARCH TEST")
    print("="*60)
    
    # Initialize
    print("\nInitializing...")
    vector_db = EmailVectorDB()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    print(f"✓ Vector DB ready with {vector_db.count()} vectors")
    
    # Get user query
    query = input("\nEnter your search query: ")
    
    if not query.strip():
        query = "events and seminars this week"
        print(f"Using default query: '{query}'")
    
    # Generate query embedding
    print("\n🔍 Searching...")
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = response.data[0].embedding
    
    # Search
    results = vector_db.search(
        query_embedding=query_embedding,
        n_results=3
    )
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    if not results["ids"][0]:
        print("No results found!")
        return
    
    for i, (chunk_id, doc, metadata, distance) in enumerate(zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ), 1):
        print(f"\n{'='*60}")
        print(f"Result {i}")
        print(f"{'='*60}")
        print(f"Chunk ID: {chunk_id}")
        print(f"Similarity: {1 - distance:.3f}")  # Convert distance to similarity
        print(f"\nFrom: {metadata.get('from', 'N/A')}")
        print(f"Subject: {metadata.get('subject', 'N/A')}")
        print(f"Date: {metadata.get('date', 'N/A')}")
        print(f"Source: {metadata.get('source_type', 'N/A')}")
        print(f"\nText Preview:")
        print(doc[:300] + "..." if len(doc) > 300 else doc)
    
    print("\n" + "="*60)
    print("✅ Search test complete!")
    print("="*60)


if __name__ == "__main__":
    main()
