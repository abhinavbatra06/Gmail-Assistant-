"""
Email Chunker Module
--------------------
Takes processed emails (Docling JSONs) and splits them into smaller chunks
for embedding and vector search.

Each chunk includes:
- The text content
- Metadata (sender, date, subject, source)
"""

import os
import json
from datetime import datetime


class EmailChunker:
    """
    Handles chunking of emails and attachments into manageable pieces.
    """
    
    def __init__(self, db_helper, chunk_size=600, overlap=100):
        """
        Initialize the chunker.
        
        Args:
            db_helper: Instance of DBHelper for tracking
            chunk_size: Target size for each chunk (rough word/token count)
            overlap: How many words to overlap between chunks
        """
        self.db = db_helper
        self.chunk_size = chunk_size  # Target ~600 words per chunk
        self.overlap = overlap  # ~100 words overlap
        self.chunk_dir = "data/chunks"
        
        # Create chunks directory if it doesn't exist
        os.makedirs(self.chunk_dir, exist_ok=True)
        print(f"Chunks directory: {self.chunk_dir}")
    
    
    def chunk_single_email(self, message_id):
        """
        Chunk a single email (body + attachments).
        
        Args:
            message_id: The message ID to chunk
            
        Returns:
            List of chunk dictionaries
        """
        print(f"\n{'='*60}")
        print(f"Processing: {message_id}")
        print(f"{'='*60}")
        
        # Check if already chunked
        if self.db.is_chunked(message_id):
            print(f"Already chunked. Skipping.")
            return self._load_chunks_from_file(message_id)
        
        # Load the Docling JSON file
        docling_path = os.path.join("data/docling", f"{message_id}.json")
        
        if not os.path.exists(docling_path):
            print(f"Docling file not found: {docling_path}")
            return []
        
        print(f"Loading: {docling_path}")
        
        with open(docling_path, 'r', encoding='utf-8') as f:
            email_data = json.load(f)
        
        # Chunk the email
        all_chunks = []
        
        # Chunk email body
        print(f"Chunking email body...")
        body_chunks = self._chunk_email_body(email_data)
        all_chunks.extend(body_chunks)
        print(f"   → Created {len(body_chunks)} chunks from body")
        
        # Chunk attachments
        attachments = email_data.get("attachments_processed", [])
        if attachments:
            print(f" Found {len(attachments)} attachments...")
            for i, attachment in enumerate(attachments, 1):
                att_name = attachment.get('filename', 'unknown')
                print(f"   → Chunking attachment {i}: {att_name}")
                att_chunks = self._chunk_attachment(email_data, attachment)
                all_chunks.extend(att_chunks)
                print(f"     Created {len(att_chunks)} chunks")
        
        # Step 4: Save chunks to file
        chunk_file_path = self._save_chunks_to_file(message_id, all_chunks)
        
        # Step 5: Mark in database as done
        self.db.mark_as_chunked(message_id, chunk_file_path, len(all_chunks))
        
        print(f"Total chunks: {len(all_chunks)}")
        return all_chunks
    
    
    def _chunk_email_body(self, email_data):
        """
        Chunk the email body text.
        
        Args:
            email_data: The Docling JSON data
            
        Returns:
            List of chunks
        """
        chunks = []
        
        # Get email body text
        body_text = email_data.get("text", "")
        
        if not body_text or len(body_text.strip()) == 0:
            print("Email body is empty")
            return chunks
        
        # Get metadata for this email
        metadata = email_data.get("metadata", {})
        
        # Simple chunking: Split by paragraphs (double newline)
        paragraphs = body_text.split("\n\n")
        
        current_chunk_text = ""
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Rough token estimate: ~4 characters = 1 token
            # (This is approximate but works for most text)
            estimated_tokens = len(current_chunk_text + para) / 4
            
            # If adding this paragraph would exceed chunk size
            if estimated_tokens > self.chunk_size and current_chunk_text:
                # Save current chunk
                chunk = self._create_chunk(
                    text=current_chunk_text,
                    chunk_index=chunk_index,
                    source_type="email_body",
                    email_data=email_data,
                    metadata=metadata
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Start new chunk with overlap (last sentence of previous)
                sentences = current_chunk_text.split(". ")
                if len(sentences) > 1:
                    overlap_text = sentences[-1] + ". "
                else:
                    overlap_text = ""
                
                current_chunk_text = overlap_text + para
            else:
                # Add paragraph to current chunk
                if current_chunk_text:
                    current_chunk_text += "\n\n" + para
                else:
                    current_chunk_text = para
        
        # Add final chunk if any text remains
        if current_chunk_text:
            chunk = self._create_chunk(
                text=current_chunk_text,
                chunk_index=chunk_index,
                source_type="email_body",
                email_data=email_data,
                metadata=metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    
    def _chunk_attachment(self, email_data, attachment):
        """
        Chunk a single attachment.
        
        Args:
            email_data: The email's Docling JSON data
            attachment: The attachment data from attachments_processed
            
        Returns:
            List of chunks
        """
        chunks = []
        
        # Get attachment text
        att_text = attachment.get("text", "")
        
        if not att_text or len(att_text.strip()) == 0:
            return chunks
        
        # Get metadata
        metadata = email_data.get("metadata", {})
        att_filename = attachment.get("filename", "unknown")
        
        # Simple chunking by paragraphs (same logic as email body)
        paragraphs = att_text.split("\n\n")
        
        current_chunk_text = ""
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            estimated_tokens = len(current_chunk_text + para) / 4
            
            if estimated_tokens > self.chunk_size and current_chunk_text:
                chunk = self._create_chunk(
                    text=current_chunk_text,
                    chunk_index=chunk_index,
                    source_type="attachment",
                    email_data=email_data,
                    metadata=metadata,
                    attachment_name=att_filename
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Overlap
                sentences = current_chunk_text.split(". ")
                if len(sentences) > 1:
                    overlap_text = sentences[-1] + ". "
                else:
                    overlap_text = ""
                
                current_chunk_text = overlap_text + para
            else:
                if current_chunk_text:
                    current_chunk_text += "\n\n" + para
                else:
                    current_chunk_text = para
        
        # Final chunk
        if current_chunk_text:
            chunk = self._create_chunk(
                text=current_chunk_text,
                chunk_index=chunk_index,
                source_type="attachment",
                email_data=email_data,
                metadata=metadata,
                attachment_name=att_filename
            )
            chunks.append(chunk)
        
        return chunks
    
    
    def _create_chunk(self, text, chunk_index, source_type, email_data, 
                     metadata, attachment_name=None):
        """
        Create a chunk dictionary with text and metadata.
        
        Args:
            text: The chunk text
            chunk_index: Index of this chunk
            source_type: "email_body" or "attachment"
            email_data: Full email data
            metadata: Email metadata
            attachment_name: Name of attachment (if applicable)
            
        Returns:
            Chunk dictionary
        """
        message_id = metadata.get("id", "unknown")
        
        # Build unique chunk ID
        if source_type == "email_body":
            chunk_id = f"{message_id}_email_{chunk_index}"
        else:
            # Clean attachment name for ID (replace special chars)
            clean_name = attachment_name.replace(".", "_").replace("/", "_").replace("\\", "_")
            chunk_id = f"{message_id}_att_{clean_name}_{chunk_index}"
        
        # Create chunk object
        chunk = {
            "chunk_id": chunk_id,
            "text": text,
            "metadata": {
                "chunk_index": chunk_index,
                "source_type": source_type,
                "message_id": message_id,
                "from": metadata.get("from", ""),
                "to": metadata.get("to", []),
                "subject": metadata.get("subject", ""),
                "date": metadata.get("date", ""),
                "timestamp": self._parse_timestamp(metadata.get("date", "")),
                "token_count": len(text) // 4  # Rough estimate
            }
        }
        
        # Add attachment info if applicable
        if attachment_name:
            chunk["metadata"]["attachment_name"] = attachment_name
            # Extract file extension
            ext = attachment_name.split(".")[-1] if "." in attachment_name else ""
            chunk["metadata"]["attachment_type"] = ext
        
        return chunk
    
    
    def _parse_timestamp(self, date_string):
        """
        Convert date string to Unix timestamp.
        Simple implementation - returns current time if parsing fails.
        
        Args:
            date_string: Date string to parse
            
        Returns:
            Unix timestamp (integer)
        """
        if not date_string:
            return int(datetime.now().timestamp())
        
        try:
            # Try parsing common formats
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(date_string)
            return int(dt.timestamp())
        except:
            # Fallback to current time if parsing fails
            return int(datetime.now().timestamp())
    
    
    def _save_chunks_to_file(self, message_id, chunks):
        """
        Save chunks to a JSON file.
        
        Args:
            message_id: The message ID
            chunks: List of chunk dictionaries
            
        Returns:
            Path to saved file
        """
        file_path = os.path.join(self.chunk_dir, f"{message_id}_chunks.json")
        
        output = {
            "message_id": message_id,
            "chunked_at": datetime.now().isoformat(),
            "total_chunks": len(chunks),
            "chunks": chunks
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Saved to: {file_path}")
        return file_path
    
    
    def _load_chunks_from_file(self, message_id):
        """
        Load chunks from existing file.
        
        Args:
            message_id: The message ID
            
        Returns:
            List of chunks
        """
        file_path = os.path.join(self.chunk_dir, f"{message_id}_chunks.json")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data["chunks"]
    
    
    def chunk_all_emails(self):
        """
        Chunk all emails that haven't been chunked yet.
        """
        # Get list of unchunked emails
        unchunked = self.db.get_unchunked_emails()
        
        if not unchunked:
            print("\nAll emails are already chunked!")
            return
        
        print(f"\n{'='*60}")
        print(f"Found {len(unchunked)} emails to chunk")
        print(f"{'='*60}\n")
        
        # Process each email
        for i, message_id in enumerate(unchunked, 1):
            print(f"\n[{i}/{len(unchunked)}]")
            try:
                self.chunk_single_email(message_id)
            except Exception as e:
                print(f"Error chunking {message_id}: {e}")
                continue
        
        # Show final stats
        print(f"\n{'='*60}")
        print("CHUNKING COMPLETE!")
        print(f"{'='*60}")
        stats = self.db.get_chunking_stats()
        print(f"Chunked emails: {stats['chunked_emails']}")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Avg chunks/email: {stats['avg_chunks_per_email']}")
