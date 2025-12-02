"""
Embed and Index Pipeline

Generates embeddings for chunks and stores them in ChromaDB.
Handles all scenarios: fresh start, partial completion, or re-indexing.

"""

import yaml
import sqlite3
from src.db_helper import DBHelper
from src.embedder import EmailEmbedder
from src.vector_db import EmailVectorDB


def main():
    print("="*60)
    print("EMBED AND INDEX PIPELINE")
    print("="*60)
    
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    print("\nInitializing components...")
    db = DBHelper(config["paths"]["db_path"])
    embedder = EmailEmbedder(db_helper=db)
    vector_db = EmailVectorDB()
    
    # Check current status
    print("\nChecking status...")
    embedding_stats = db.get_embedding_stats()
    vector_count = vector_db.count()
    
    print(f"   Chunked emails: {embedding_stats['chunked_emails']}")
    print(f"   Embedded emails: {embedding_stats['embedded_emails']}")
    print(f"   Unembedded emails: {embedding_stats['unembedded_emails']}")
    print(f"   Total embeddings in DB: {embedding_stats['total_embeddings']}")
    print(f"   Vectors in ChromaDB: {vector_count}")
    
    # Determine what needs to be done
    needs_embedding = embedding_stats['unembedded_emails'] > 0
    needs_indexing = vector_count < embedding_stats['total_embeddings']
    
    # Scenario 1: New emails need embedding
    if needs_embedding:
        print(f"\nGenerating embeddings for {embedding_stats['unembedded_emails']} new emails...")
        new_chunks = embedder.embed_all_messages()
        
        if new_chunks:
            print(f"\nAdding {len(new_chunks)} new chunks to ChromaDB...")
            vector_db.add_chunks(new_chunks)
        
        # Refresh stats
        embedding_stats = db.get_embedding_stats()
        vector_count = vector_db.count()
        needs_indexing = vector_count < embedding_stats['total_embeddings']
    
    # Scenario 2: Embeddings exist but ChromaDB is incomplete
    if needs_indexing:
        missing = embedding_stats['total_embeddings'] - vector_count
        print(f"\nChromaDB missing {missing} vectors!")
        print("   Regenerating embeddings to populate ChromaDB...")
        
        print("\nClearing embedding status...")
        conn = sqlite3.connect(config["paths"]["db_path"])
        cur = conn.cursor()
        cur.execute("DELETE FROM embedding_status")
        conn.commit()
        conn.close()
        
        print("Embedding status cleared")
        
        print("\nRegenerating embeddings...")
        # Reinitialize with fresh state
        db_new = DBHelper(config["paths"]["db_path"])
        embedder_new = EmailEmbedder(db_helper=db_new)
        
        all_chunks = embedder_new.embed_all_messages()
        
        if all_chunks:
            print(f"\nAdding {len(all_chunks)} chunks to ChromaDB...")
            vector_db.add_chunks(all_chunks)
        
        db_new.close()
        
        # Refresh stats
        embedding_stats = db.get_embedding_stats()
        vector_count = vector_db.count()
    
    # Scenario 3: Everything already done
    if not needs_embedding and not needs_indexing:
        print("\nEverything already up to date!")
        print(f"   {embedding_stats['embedded_emails']} emails embedded")
        print(f"   {vector_count} vectors in ChromaDB")
    
    # Final statistics
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    
    final_embedding_stats = db.get_embedding_stats()
    final_vector_count = vector_db.count()
    
    print(f"Emails:")
    print(f"   Chunked: {final_embedding_stats['chunked_emails']}")
    print(f"   Embedded: {final_embedding_stats['embedded_emails']}")
    print(f"   Total embeddings: {final_embedding_stats['total_embeddings']}")
    
    print(f"\nChromaDB:")
    print(f"   Vectors: {final_vector_count}")
    print(f"   Collection: {vector_db.collection_name}")
    print(f"   Location: {vector_db.persist_dir}")
    
    # Verify sync
    if final_vector_count == final_embedding_stats['total_embeddings']:
        print(f"\nSUCCESS: Embeddings and ChromaDB are in sync!")
    else:
        print(f"\nWARNING: Mismatch detected!")
        print(f"   Embeddings: {final_embedding_stats['total_embeddings']}")
        print(f"   Vectors: {final_vector_count}")
    
    # Close connections
    db.close()
    
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
