"""
Test Script for Email Chunker
------------------------------
Tests the chunking functionality on your emails.

This script:
1. Loads config for chunking settings
2. Connects to database
3. Tests chunking on one email first
4. Shows results and statistics
"""

import yaml
from src.db_helper import DBHelper
from src.chunker import EmailChunker


def main():
    print("="*60)
    print("EMAIL CHUNKER TEST")
    print("="*60)
    
    # Step 1: Load config
    print("\nLoading config...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Get chunking settings (with defaults if not in config)
    chunking_config = config.get("chunking", {})
    chunk_size = chunking_config.get("chunk_size", 600)
    overlap = chunking_config.get("overlap", 100)
    
    print(f"   Chunk size: {chunk_size} tokens")
    print(f"   Overlap: {overlap} tokens")
    
    # Connect to database
    print("\nConnecting to database...")
    db = DBHelper("db/emails.db")
    
    # Show current stats
    print("\nCurrent Statistics:")
    stats = db.get_chunking_stats()
    print(f"   Total emails: {stats['total_emails']}")
    print(f"   Already chunked: {stats['chunked_emails']}")
    print(f"   Need chunking: {stats['unchunked_emails']}")
    print(f"   Total chunks: {stats['total_chunks']}")
    
    # Get unchunked emails
    unchunked = db.get_unchunked_emails()
    
    if not unchunked:
        print("\nAll emails are already chunked!")
        print("\nTo re-chunk, delete entries from chunking_status table:")
        print("   DELETE FROM chunking_status WHERE message_id='<message_id>';")
        db.close()
        return
    
    print(f"\nFound {len(unchunked)} emails to chunk")
    
    # Create chunker
    print("\nInitializing chunker...")
    chunker = EmailChunker(
        db_helper=db,
        chunk_size=chunk_size,
        overlap=overlap
    )
    
    # Test on ONE email first
    print("\n" + "="*60)
    print("TESTING ON FIRST EMAIL")
    print("="*60)
    
    test_email_id = unchunked[0]
    print(f"\nTest email ID: {test_email_id}")
    
    # Chunk it
    chunks = chunker.chunk_single_email(test_email_id)
    
    # Show results
    if chunks:
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"\nSuccessfully created {len(chunks)} chunks")
        
        # Show first chunk as example
        print("\nFirst Chunk Preview:")
        first_chunk = chunks[0]
        print(f"   Chunk ID: {first_chunk['chunk_id']}")
        print(f"   Source: {first_chunk['metadata']['source_type']}")
        print(f"   From: {first_chunk['metadata']['from']}")
        print(f"   Subject: {first_chunk['metadata']['subject']}")
        print(f"   Tokens: {first_chunk['metadata']['token_count']}")
        print(f"\n   Text Preview:")
        preview_text = first_chunk['text'][:200]
        print(f"   {preview_text}...")
        
        # Ask if user wants to continue with all emails
        print("\n" + "="*60)
        print(f"Remaining emails to chunk: {len(unchunked) - 1}")
        
        if len(unchunked) > 1:
            response = input("\nChunk all remaining emails? (y/n): ").strip().lower()
            
            if response == 'y':
                print("\nChunking all emails...")
                chunker.chunk_all_emails()
            else:
                print("\nSkipping remaining emails.")
                print("   Run this script again or use chunker.chunk_all_emails()")
    else:
        print("\nNo chunks created. Check if Docling file exists.")
    
    # Final stats
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    final_stats = db.get_chunking_stats()
    print(f"   Chunked emails: {final_stats['chunked_emails']}")
    print(f"   Total chunks: {final_stats['total_chunks']}")
    print(f"   Avg chunks/email: {final_stats['avg_chunks_per_email']}")
    
    # Close database
    db.close()
    print("\nTest complete!")


if __name__ == "__main__":
    main()
