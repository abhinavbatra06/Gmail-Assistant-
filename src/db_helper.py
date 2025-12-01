import sqlite3
from datetime import datetime

class DBHelper:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self._create_tables()

    def _create_tables(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS emails (
            id TEXT PRIMARY KEY,
            sender TEXT,
            subject TEXT,
            date TEXT,
            eml_path TEXT,
            metadata_path TEXT,
            status TEXT DEFAULT 'downloaded',
            fetched_at TEXT
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS attachments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id TEXT,
            filename TEXT,
            path TEXT
        )""")

        # create table to track emails that have been chunked 

        cur.execute("""
        CREATE TABLE IF NOT EXISTS chunking_status (
            message_id TEXT PRIMARY KEY,              
            chunked_at TEXT,
            total_chunks INTEGER,
            chunk_file_path TEXT,
            chunking_version TEXT DEFAULT 'v1',
            FOREIGN KEY (message_id) REFERENCES emails(id)  
        )""")

        # create table to track emails that have been embedded
        cur.execute("""
        CREATE TABLE IF NOT EXISTS embedding_status (
            message_id TEXT PRIMARY KEY,
            embedded_at TEXT,
            total_embeddings INTEGER,
            embedding_model TEXT,
            FOREIGN KEY (message_id) REFERENCES emails(id)
        )""")

        self.conn.commit()

    def email_exists(self, msg_id):
        cur = self.conn.cursor()
        cur.execute("SELECT 1 FROM emails WHERE id=?", (msg_id,))
        return cur.fetchone() is not None

    def insert_email(self, msg_id, sender, subject, date, eml_path, meta_path):
        cur = self.conn.cursor()
        cur.execute("""
            INSERT OR IGNORE INTO emails
            (id, sender, subject, date, eml_path, metadata_path, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (msg_id, sender, subject, date, eml_path, meta_path,
             datetime.now().isoformat()))
        self.conn.commit()

    def update_status(self, msg_id, status):
        cur = self.conn.cursor()
        cur.execute("UPDATE emails SET status=? WHERE id=?", (status, msg_id))
        self.conn.commit()
    
    def insert_attachment(self, message_id, filename, path):
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO attachments (message_id, filename, path)
            VALUES (?, ?, ?)""",
            (message_id, filename, path))
        self.conn.commit()
    
    def get_attachments_for_email(self, msg_id):
        cur = self.conn.cursor()
        cur.execute("SELECT filename, path FROM attachments WHERE message_id=?", (msg_id,))
        return cur.fetchall()


    # check if a given email has been chunked
    def is_chunked(self, message_id):            
        cur = self.conn.cursor()
        cur.execute("SELECT 1 FROM chunking_status WHERE message_id=?", (message_id,))
        return cur.fetchone() is not None
    
    # mark an email as chunked
    def mark_as_chunked(self, message_id, chunk_file_path, total_chunks):  
        cur = self.conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO chunking_status 
            (message_id, chunked_at, total_chunks, chunk_file_path, chunking_version)
            VALUES (?, ?, ?, ?, 'v1')""",
            (message_id, datetime.now().isoformat(), total_chunks, chunk_file_path))
        self.conn.commit()

    # fetch emails that are not yet chunked 

    def get_unchunked_emails(self):
        cur = self.conn.cursor()
        cur.execute("""
            SELECT e.id 
            FROM emails e
            LEFT JOIN chunking_status c ON e.id = c.message_id  
            WHERE c.message_id IS NULL                           
        """)
        return [row[0] for row in cur.fetchall()]
    
    # get some statistics about chunked emails
    def get_chunking_stats(self):

        """Get chunking statistics"""
        cur = self.conn.cursor()

        # Total emails
        cur.execute("SELECT COUNT(*) FROM emails")
        total_emails = cur.fetchone()[0]

        # Chunked emails
        cur.execute("SELECT COUNT(*) FROM chunking_status")
        chunked_emails = cur.fetchone()[0]

        # Total chunks
        cur.execute("SELECT SUM(total_chunks) FROM chunking_status")
        result = cur.fetchone()[0]
        total_chunks = result if result else 0

        return {
            "total_emails": total_emails,
            "chunked_emails": chunked_emails,
            "unchunked_emails": total_emails - chunked_emails,
            "total_chunks": total_chunks,
            "avg_chunks_per_email": round(total_chunks / chunked_emails, 2) if chunked_emails > 0 else 0
        }
    
    # helper to get chunk file path for a given message id
    
    def get_chunk_file_path(self, message_id):
        
        """Get the path to chunks file for a message"""
        
        cur = self.conn.cursor()
        cur.execute("""
            SELECT chunk_file_path 
            FROM chunking_status 
            WHERE message_id=?
        """, (message_id,))
        result = cur.fetchone()
        
        return result[0] if result else None

    # ==================== EMBEDDING STATUS METHODS ====================

    def is_embedded(self, message_id):
        """Check if a message has been embedded"""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT 1 FROM embedding_status WHERE message_id=?
        """, (message_id,))
        return cur.fetchone() is not None

    def mark_as_embedded(self, message_id, total_embeddings, embedding_model):
        """Mark a message as embedded"""
        cur = self.conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO embedding_status 
            (message_id, embedded_at, total_embeddings, embedding_model)
            VALUES (?, ?, ?, ?)
        """, (message_id, datetime.now().isoformat(), total_embeddings, embedding_model))
        self.conn.commit()

    def get_unembedded_messages(self):
        """Get list of message IDs that are chunked but not yet embedded"""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT c.message_id 
            FROM chunking_status c
            LEFT JOIN embedding_status e ON c.message_id = e.message_id
            WHERE e.message_id IS NULL
        """)
        return [row[0] for row in cur.fetchall()]

    def get_embedding_stats(self):
        """Get statistics about embedding status"""
        cur = self.conn.cursor()
        
        # Total chunked emails
        cur.execute("SELECT COUNT(*) FROM chunking_status")
        chunked_emails = cur.fetchone()[0]
        
        # Embedded emails
        cur.execute("SELECT COUNT(*) FROM embedding_status")
        embedded_emails = cur.fetchone()[0]
        
        # Total embeddings
        cur.execute("SELECT SUM(total_embeddings) FROM embedding_status")
        total_embeddings = cur.fetchone()[0] or 0
        
        return {
            "chunked_emails": chunked_emails,
            "embedded_emails": embedded_emails,
            "unembedded_emails": chunked_emails - embedded_emails,
            "total_embeddings": total_embeddings,
            "avg_embeddings_per_email": round(total_embeddings / embedded_emails, 2) if embedded_emails > 0 else 0
        }

    def close(self):
        self.conn.close()
