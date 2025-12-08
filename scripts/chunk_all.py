"""
Chunk All Emails - Direct Execution
No prompts, just chunks everything.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.db_helper import DBHelper
from src.chunker import EmailChunker


def main():
    print("="*60)
    print("CHUNKING ALL EMAILS")
    print("="*60)
    
    # Load config
    print("\n Loading config...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    chunking_config = config.get("chunking", {})
    chunk_size = chunking_config.get("chunk_size", 600)
    overlap = chunking_config.get("overlap", 100)
    
    print(f"   Chunk size: {chunk_size} tokens")
    print(f"   Overlap: {overlap} tokens")
    
    # Connect to database
    print("\n Connecting to database...")
    db = DBHelper("db/emails.db")
    
    # Show stats
    print("\n Current Statistics:")
    stats = db.get_chunking_stats()
    print(f"   Total emails: {stats['total_emails']}")
    print(f"   Already chunked: {stats['chunked_emails']}")
    print(f"   Need chunking: {stats['unchunked_emails']}")
    
    if stats['unchunked_emails'] == 0:
        print("\n All emails already chunked!")
        db.close()
        return
    
    # Create chunker
    print("\n🔧 Initializing chunker...")

    # added
    # get Predict DB path from config if available
    rag_config = config.get("rag", {})
    predict_db_path = rag_config.get("predict_db_path", "db/events.db") if rag_config.get("enable_predict", True) else None

    chunker = EmailChunker(
        db_helper=db,
        chunk_size=chunk_size,
        overlap=overlap,
        predict_db_path=predict_db_path # added for predict DB integration
    )
    
    # Chunk all emails
    print(f"\n Starting to chunk {stats['unchunked_emails']} emails...\n")
    chunker.chunk_all_emails()
    
    # Final stats
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    final_stats = db.get_chunking_stats()
    print(f"   Total emails: {final_stats['total_emails']}")
    print(f"   Chunked emails: {final_stats['chunked_emails']}")
    print(f"   Total chunks: {final_stats['total_chunks']}")
    print(f"   Avg chunks/email: {final_stats['avg_chunks_per_email']}")
    
    db.close()
    print("\n All done!")


if __name__ == "__main__":
    main()
