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
import hashlib
from datetime import datetime
from typing import Optional # added

class EmailChunker:
    """
    Handles chunking of emails and attachments into manageable pieces.
    """
    
    def __init__(self, db_helper, chunk_size=600, overlap=100, predict_db_path: Optional[str] = None): # modified
        """
        Initialize the chunker.
        
        Args:
            db_helper: Instance of DBHelper for tracking
            chunk_size: Target size for each chunk (rough word/token count)
            overlap: How many words to overlap between chunks
            predict_db_path: Optional path to Predict event database (for ICS parsing) # added
        """
        self.db = db_helper
        self.chunk_size = chunk_size  # Target ~600 words per chunk
        self.overlap = overlap  # ~100 words overlap
        self.chunk_dir = "data/chunks"

        # added
        # initialize Predict module if path provided
        self.predict = None
        if predict_db_path:
            try:
                from src.predict import Predict
                self.predict = Predict(predict_db_path)
                print(f"Predict module initialized: {predict_db_path}")
            except Exception as e:
                print(f"Warning: Could not initialize Predict module: {str(e)}")
        
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

        # added:
        # extract events from email body text if Predict module is available
        if self.predict and body_chunks:
            email_metadata = email_data.get("metadata", {})
            email_date = email_metadata.get("date", "")
            email_subject = email_metadata.get("subject", "")
            # get full email text for event extraction
            email_text = email_data.get("text", "")
            if email_text and len(email_text.strip()) > 50:
                event_count = self.predict.extract_events_from_text(
                    message_id, email_text, email_date, email_subject
                )
                if event_count > 0:
                    print(f"     Extracted {event_count} events from email body text")
        
        # Chunk attachments
        attachments = email_data.get("attachments_processed", [])
        if attachments:
            print(f" Found {len(attachments)} attachments...")
            for i, attachment in enumerate(attachments, 1):
                att_name = attachment.get('filename', 'unknown')
                att_metadata = attachment.get("metadata", {}) # added
                print(f"   → Chunking attachment {i}: {att_name}")

                # added: special handling for .ics files
                # extract events to Predict database if this is a calendar file
                if self.predict and att_metadata.get("is_calendar"):
                    att_path = attachment.get("path", "")
                    if att_path and os.path.exists(att_path):
                        # get email date for deduplication
                        email_metadata = email_data.get("metadata", {})
                        email_date = email_metadata.get("date", "")
                        event_count = self.predict.extract_events_from_email(
                            message_id, att_path, att_name, email_date
                        )
                        if event_count > 0:
                            print(f"     Extracted {event_count} events to Predict database (with deduplication)")

                # added 
                # extract events from attachment text (for images, pdfs, etc processed by docling)
                if self.predict:
                    att_text = attachment.get("text", "")
                    if att_text and len(att_text.strip()) > 50:
                        # check if this looks like event-related content
                        event_keywords = ["event", "meeting", "workshop", "session", "date", "time", "location", 
                                         "venue", "register", "rsvp", "calendar", "schedule"]
                        att_text_lower = att_text.lower()
                        if any(keyword in att_text_lower for keyword in event_keywords):
                            email_metadata = email_data.get("metadata", {})
                            email_date = email_metadata.get("date", "")
                            email_subject = email_metadata.get("subject", "")
                            # use attachment filename as context
                            event_context = f"Attachment: {att_name}\n\n{att_text}"
                            event_count = self.predict.extract_events_from_text(
                                message_id, event_context, email_date, email_subject
                            )
                            if event_count > 0:
                                print(f"     Extracted {event_count} events from attachment text")

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
        Uses event-aware chunking for calendar files (keeps events as atomic units). # added 

        Args:
            email_data: The email's Docling JSON data
            attachment: The attachment data from attachments_processed
            
        Returns:
            List of chunks
        """
        chunks = []

        # added:
        metadata = email_data.get("metadata", {})
        att_filename = attachment.get("filename", "unknown")
        att_metadata = attachment.get("metadata", {})
        
        # check if this is a calendar attachment with structured events
        if att_metadata.get("is_calendar") and attachment.get("events"):
            # event-aware chunking: keep each event as an atomic unit
            return self._chunk_calendar_events(email_data, attachment, metadata)
        
        # standard chunking for other attachments follows:
        
        # Get attachment text
        att_text = attachment.get("text", "")
        
        if not att_text or len(att_text.strip()) == 0:
            return chunks
        
        # # Get metadata
        # metadata = email_data.get("metadata", {})
        # att_filename = attachment.get("filename", "unknown")
        
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
    
    # added
    def _chunk_calendar_events(self, email_data, attachment, metadata):
        """
        Event-aware chunking: keep each calendar event as an atomic unit.
        
        Args:
            email_data: The email's Docling JSON data
            attachment: The attachment data with events
            metadata: Email metadata
            
        Returns:
            List of chunks (one per event)
        """
        chunks = []
        events = attachment.get("events", [])
        att_filename = attachment.get("filename", "unknown")
        
        for event_idx, event in enumerate(events):
            # each event becomes one chunk (atomic unit)
            event_text = event.get("text", "")
            
            # add event metadata to chunk
            chunk = self._create_chunk(
                text=event_text,
                chunk_index=event_idx,
                source_type="calendar",
                email_data=email_data,
                metadata=metadata,
                attachment_name=att_filename,
                event_data=event  # pass structured event data
            )
            chunks.append(chunk)
        
        return chunks


    def _create_chunk(self, text, chunk_index, source_type, email_data, 
                     metadata, attachment_name=None, event_data=None): # modified
        """
        Create a chunk dictionary with text and metadata.
        
        Args:
            text: The chunk text
            chunk_index: Index of this chunk
            source_type: "email_body", "attachment", or "calendar" # modified
            email_data: Full email data
            metadata: Email metadata
            attachment_name: Name of attachment (if applicable)
            event_data: Structured event data (for calendar events) # added
            
        Returns:
            Chunk dictionary
        """
        message_id = metadata.get("id", "unknown")
        
        # Create a unique hash from the text content to ensure uniqueness
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
        
        # Build unique chunk ID
        if source_type == "email_body":
            chunk_id = f"{message_id}_email_{chunk_index}_{text_hash}"

        # added
        elif source_type == "calendar":
            # for calendar events, use event uid if available
            event_uid = event_data.get("uid", "") if event_data else ""
            if event_uid:
                chunk_id = f"{message_id}_event_{event_uid}_{chunk_index}_{text_hash}"
            else:
                clean_name = attachment_name.replace(".", "_").replace("/", "_").replace("\\", "_") if attachment_name else "calendar"
                chunk_id = f"{message_id}_event_{clean_name}_{chunk_index}_{text_hash}"

        else:
            # Clean attachment name for ID (replace special chars)
            clean_name = attachment_name.replace(".", "_").replace("/", "_").replace("\\", "_")
            chunk_id = f"{message_id}_att_{clean_name}_{chunk_index}_{text_hash}"
        
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
        
        # added
        # add event-specific metadata for calendar chunks
        if event_data and source_type == "calendar":
            chunk["metadata"]["is_calendar_event"] = True
            chunk["metadata"]["event_summary"] = event_data.get("summary", "")
            chunk["metadata"]["event_location"] = event_data.get("location", "")
            chunk["metadata"]["event_organizer"] = event_data.get("organizer", "")
            chunk["metadata"]["event_uid"] = event_data.get("uid", "")
            
            # add timestamps for event filtering (handle both datetime objects and ISO strings)
            from datetime import datetime
            
            start_time = event_data.get("start_time")
            if start_time:
                if isinstance(start_time, datetime):
                    chunk["metadata"]["event_start_time"] = int(start_time.timestamp())
                elif isinstance(start_time, str):
                    # parse ISO format string (from JSON)
                    try:
                        dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                        chunk["metadata"]["event_start_time"] = int(dt.timestamp())
                    except:
                        pass
            
            end_time = event_data.get("end_time")
            if end_time:
                if isinstance(end_time, datetime):
                    chunk["metadata"]["event_end_time"] = int(end_time.timestamp())
                elif isinstance(end_time, str):
                    # parse ISO format string (from JSON)
                    try:
                        dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                        chunk["metadata"]["event_end_time"] = int(dt.timestamp())
                    except:
                        pass
            if event_data.get("start_time_str"):
                chunk["metadata"]["event_start_str"] = event_data["start_time_str"]
            if event_data.get("end_time_str"):
                chunk["metadata"]["event_end_str"] = event_data["end_time_str"]

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
