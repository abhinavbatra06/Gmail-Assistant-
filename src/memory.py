"""
Memory Module - User Preferences and Context

Stores and retrieves user preferences, query history, and context for personalized RAG.
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional


class Memory:
    
    def __init__(self, db_path: str = "db/memory.db"):
        """
        Initialize Memory module.
        
        Args:
            db_path: Path to SQLite database for memory
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables for memory."""
        cur = self.conn.cursor()
        
        # user preferences
        cur.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE,
                value TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # query history
        cur.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                intent TEXT,
                module_used TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                result_count INTEGER
            )
        """)
        
        # user context (conversation state)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                key TEXT,
                value TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_id, key)
            )
        """)
        
        # indexes
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_query_history_timestamp ON query_history(timestamp)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_context_session ON context(session_id)
        """)
        
        self.conn.commit()
    
    def set_preference(self, key: str, value: any):
        """
        Set a user preference.
        
        Args:
            key: Preference key (e.g., "preferred_senders", "default_date_range")
            value: Preference value (will be JSON-encoded if not string)
        """
        cur = self.conn.cursor()
        
        if not isinstance(value, str):
            value = json.dumps(value)
        
        cur.execute("""
            INSERT OR REPLACE INTO preferences (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, (key, value))
        
        self.conn.commit()
    
    def get_preference(self, key: str, default: any = None) -> any:
        """
        Get a user preference.
        
        Args:
            key: Preference key
            default: Default value if not found
            
        Returns:
            Preference value (JSON-decoded if applicable)
        """
        cur = self.conn.cursor()
        cur.execute("SELECT value FROM preferences WHERE key = ?", (key,))
        row = cur.fetchone()
        
        if row:
            value = row[0]
            # try to decode JSON
            try:
                return json.loads(value)
            except:
                return value
        
        return default
    
    def get_preferred_senders(self) -> List[str]:
        """
        Get list of preferred senders.
        
        Returns:
            List of sender email addresses or domains
        """
        return self.get_preference("preferred_senders", [])
    
    def set_preferred_senders(self, senders: List[str]):
        """
        Set preferred senders.
        
        Args:
            senders: List of sender email addresses or domains
        """
        self.set_preference("preferred_senders", senders)
    
    def get_default_filters(self) -> Dict:
        """
        Get default metadata filters based on preferences.
        
        Returns:
            Dict with default filters
        """
        filters = {}
        
        preferred_senders = self.get_preferred_senders()
        if preferred_senders:
            filters["from"] = {"$in": [{"$contains": sender} for sender in preferred_senders]}
        
        return filters
    
    def log_query(self, query: str, intent: str, module_used: str, result_count: int) -> int:
        """
        Log a query to history.
        
        Args:
            query: User query
            intent: Detected intent
            module_used: Module that handled the query
            result_count: Number of results returned
            
        Returns:
            query_id: ID of inserted query (for linking to query_responses)
        """
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO query_history (query, intent, module_used, result_count)
            VALUES (?, ?, ?, ?)
        """, (query, intent, module_used, result_count))
        self.conn.commit()
        return cur.lastrowid
    
    def get_recent_queries(self, limit: int = 10) -> List[Dict]:
        """
        Get recent query history.
        
        Args:
            limit: Number of recent queries to return
            
        Returns:
            List of query dictionaries
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT query, intent, module_used, timestamp, result_count
            FROM query_history
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        rows = cur.fetchall()
        return [
            {
                "query": row[0],
                "intent": row[1],
                "module_used": row[2],
                "timestamp": row[3],
                "result_count": row[4]
            }
            for row in rows
        ]
    
    def get_similar_queries(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Get similar queries from history.
        
        Args:
            query: Current query
            limit: Number of similar queries to return
            
        Returns:
            List of similar query dictionaries
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT query, intent, module_used, timestamp, result_count
            FROM query_history
            WHERE query LIKE ? OR intent IN (
                SELECT intent FROM query_history 
                WHERE query LIKE ?
                ORDER BY timestamp DESC
                LIMIT 1
            )
            ORDER BY timestamp DESC
            LIMIT ?
        """, (f"%{query[:20]}%", f"%{query[:20]}%", limit))
        
        rows = cur.fetchall()
        return [
            {
                "query": row[0],
                "intent": row[1],
                "module_used": row[2],
                "timestamp": row[3],
                "result_count": row[4]
            }
            for row in rows
        ]
    
    def set_context(self, session_id: str, key: str, value: any):
        """
        Set context for a session.
        
        Args:
            session_id: Session identifier
            key: Context key
            value: Context value
        """
        cur = self.conn.cursor()
        
        if not isinstance(value, str):
            value = json.dumps(value)
        
        cur.execute("""
            INSERT OR REPLACE INTO context (session_id, key, value, timestamp)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """, (session_id, key, value))
        
        self.conn.commit()
    
    def get_context(self, session_id: str, key: str, default: any = None) -> any:
        """
        Get context for a session.
        
        Args:
            session_id: Session identifier
            key: Context key
            default: Default value if not found
            
        Returns:
            Context value
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT value FROM context 
            WHERE session_id = ? AND key = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (session_id, key))
        
        row = cur.fetchone()
        if row:
            value = row[0]
            try:
                return json.loads(value)
            except:
                return value
        
        return default
    
    def get_all_context(self, session_id: str) -> Dict:
        """
        Get all context for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict of all context key-value pairs
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT key, value FROM context 
            WHERE session_id = ?
            ORDER BY timestamp DESC
        """, (session_id,))
        
        context = {}
        for row in cur.fetchall():
            key, value = row
            try:
                context[key] = json.loads(value)
            except:
                context[key] = value
        
        return context
    
    def clear_context(self, session_id: str):
        """
        Clear all context for a session.
        
        Args:
            session_id: Session identifier
        """
        cur = self.conn.cursor()
        cur.execute("DELETE FROM context WHERE session_id = ?", (session_id,))
        self.conn.commit()
    
    def close(self):
        """Close database connection."""
        self.conn.close()

