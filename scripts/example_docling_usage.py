"""
Example script showing how to use Docling to process emails
fetched by gmail_ingest.py and prepare them for chunking.
Does not need to be deployed as part of regular application code.
"""

import os
import sys
import json

# Add parent directory to path to import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.docling_processor import DoclingProcessor


def main():
    # Initialize the processor
    print("Initializing Docling Processor...")
    processor = DoclingProcessor(config_path="config.yaml")

    try:
        # Option 1: Process all emails
        print("\n📧 Processing all emails...")
        docs = processor.process_all_emails(limit=10)  # Process first 10
        print(f"✅ Processed {len(docs)} emails")

        # Option 2: Process a specific email
        if docs:
            msg_id = docs[0]["metadata"]["id"]
            print(f"\n📄 Processing specific email: {msg_id}")
            doc = processor.process_email(msg_id)
            if doc:
                print(f"   Subject: {doc['metadata'].get('subject', 'N/A')}")
                print(f"   Text length: {len(doc.get('text', ''))} characters")

        # Option 3: Get chunkable text for an email
        if docs:
            msg_id = docs[0]["metadata"]["id"]
            print(f"\n📝 Getting chunkable text for: {msg_id}")
            chunkable_text = processor.get_chunkable_text(msg_id)
            print(f"   Chunkable text length: {len(chunkable_text)} characters")
            print(f"   Preview:\n{chunkable_text[:200]}...")

        # Option 4: Process attachments
        if docs:
            msg_id = docs[0]["metadata"]["id"]
            print(f"\n📎 Processing attachments for: {msg_id}")
            attachments = processor.process_attachments(msg_id)
            print(f"   Processed {len(attachments)} attachments")
            for att in attachments:
                print(f"   - {att['filename']}: {len(att.get('text', ''))} chars, "
                      f"type: {att['metadata'].get('content_type', 'unknown')}")

        # Option 6: Get chunkable text with attachments
        if docs:
            msg_id = docs[0]["metadata"]["id"]
            print(f"\n📝 Getting chunkable text WITH attachments for: {msg_id}")
            chunkable_text_with_att = processor.get_chunkable_text(msg_id, include_attachments=True)
            print(f"   Total length: {len(chunkable_text_with_att)} characters")
            
            print(f"\n📝 Getting chunkable text WITHOUT attachments for: {msg_id}")
            chunkable_text_no_att = processor.get_chunkable_text(msg_id, include_attachments=False)
            print(f"   Total length: {len(chunkable_text_no_att)} characters")
            print(f"   Difference: {len(chunkable_text_with_att) - len(chunkable_text_no_att)} chars from attachments")

        # Option 5: Load and inspect a processed document
        if docs:
            msg_id = docs[0]["metadata"]["id"]
            docling_path = processor.paths.path_for_docling(msg_id)
            print(f"\n📂 Inspecting processed document: {docling_path}")
            with open(docling_path, 'r', encoding='utf-8') as f:
                processed_doc = json.load(f)
            
            print(f"   Keys: {list(processed_doc.keys())}")
            print(f"   Has tables: {len(processed_doc.get('tables', [])) > 0}")
            print(f"   Has sections: {len(processed_doc.get('sections', [])) > 0}")

    finally:
        processor.close()
        print("\n✅ Done!")


if __name__ == "__main__":
    main()

