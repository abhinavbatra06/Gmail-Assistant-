"""
Evaluation Logger - Logs detailed query responses for RAG evaluation

Stores complete query responses with all retrieval details for building golden datasets
and computing evaluation metrics (precision, recall, MRR, etc.)
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional


class EvalLogger:
    
    def __init__(self, db_path: str = "db/memory.db"):
        """
        Initialize evaluation logger.
        
        Args:
            db_path: Path to SQLite database (same as Memory module)
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_table()
    
    def _create_table(self):
        """Create evaluation logging table."""
        cur = self.conn.cursor()
        
        # query responses table for evaluation
        cur.execute("""
            CREATE TABLE IF NOT EXISTS query_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER,
                query TEXT NOT NULL,
                answer TEXT,
                intent TEXT,
                intent_confidence REAL,
                module_used TEXT,
                routing_reason TEXT,
                num_chunks_retrieved INTEGER,
                retrieval_method TEXT,
                sources JSON,
                retrieved_chunks JSON,
                chunk_scores JSON,
                reranked BOOLEAN DEFAULT 0,
                query_optimized BOOLEAN DEFAULT 0,
                optimization_method TEXT,
                latency_ms INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (query_id) REFERENCES query_history(id)
            )
        """)
        
        # index for fast lookups
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_query_responses_query_id 
            ON query_responses(query_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_query_responses_timestamp 
            ON query_responses(timestamp)
        """)
        
        self.conn.commit()
    
    def log_response(self, 
                     query_id: int,
                     query: str,
                     result: Dict,
                     latency_ms: Optional[int] = None) -> int:
        """
        Log a complete query response for evaluation.
        
        Args:
            query_id: ID from query_history table (for joining)
            query: Original query string
            result: Full result dict from RAG query
            latency_ms: Query latency in milliseconds
            
        Returns:
            Response ID
        """
        cur = self.conn.cursor()
        
        # extract fields from result
        answer = result.get('answer', '')
        sources = json.dumps(result.get('sources', []))
        retrieved_chunks = json.dumps(result.get('retrieved_chunks', []))
        num_chunks = result.get('num_chunks_retrieved', 0)
        
        # routing/intent info
        intent = result.get('intent', 'general')
        intent_confidence = result.get('intent_confidence', 0.0)
        
        routing = result.get('routing', {})
        module_used = routing.get('module', result.get('module_used', 'retriever'))
        routing_reason = routing.get('reason', '')
        
        # retrieval details
        retrieval_method = result.get('retrieval_method', 'unknown')
        reranked = result.get('reranked', False)
        query_optimized = result.get('query_optimized', False)
        optimization_method = result.get('optimization_method', '')
        
        # chunk scores (distances/similarities)
        chunk_scores = json.dumps(result.get('chunk_scores', []))
        
        cur.execute("""
            INSERT INTO query_responses (
                query_id, query, answer, intent, intent_confidence,
                module_used, routing_reason, num_chunks_retrieved,
                retrieval_method, sources, retrieved_chunks, chunk_scores,
                reranked, query_optimized, optimization_method, latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            query_id, query, answer, intent, intent_confidence,
            module_used, routing_reason, num_chunks,
            retrieval_method, sources, retrieved_chunks, chunk_scores,
            reranked, query_optimized, optimization_method, latency_ms
        ))
        
        self.conn.commit()
        return cur.lastrowid
    
    def get_response(self, query_id: int) -> Optional[Dict]:
        """
        Get logged response by query_id.
        
        Args:
            query_id: Query ID from query_history
            
        Returns:
            Response dict or None
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT 
                id, query, answer, intent, intent_confidence,
                module_used, routing_reason, num_chunks_retrieved,
                retrieval_method, sources, retrieved_chunks, chunk_scores,
                reranked, query_optimized, optimization_method, latency_ms,
                timestamp
            FROM query_responses
            WHERE query_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (query_id,))
        
        row = cur.fetchone()
        if not row:
            return None
        
        return {
            'id': row[0],
            'query': row[1],
            'answer': row[2],
            'intent': row[3],
            'intent_confidence': row[4],
            'module_used': row[5],
            'routing_reason': row[6],
            'num_chunks_retrieved': row[7],
            'retrieval_method': row[8],
            'sources': json.loads(row[9]) if row[9] else [],
            'retrieved_chunks': json.loads(row[10]) if row[10] else [],
            'chunk_scores': json.loads(row[11]) if row[11] else [],
            'reranked': bool(row[12]),
            'query_optimized': bool(row[13]),
            'optimization_method': row[14],
            'latency_ms': row[15],
            'timestamp': row[16]
        }
    
    def get_all_responses(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get all logged responses.
        
        Args:
            limit: Max number of responses to return
            
        Returns:
            List of response dicts
        """
        cur = self.conn.cursor()
        
        query = """
            SELECT 
                id, query_id, query, answer, intent, intent_confidence,
                module_used, routing_reason, num_chunks_retrieved,
                retrieval_method, sources, retrieved_chunks, chunk_scores,
                reranked, query_optimized, optimization_method, latency_ms,
                timestamp
            FROM query_responses
            ORDER BY timestamp DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cur.execute(query)
        
        responses = []
        for row in cur.fetchall():
            responses.append({
                'id': row[0],
                'query_id': row[1],
                'query': row[2],
                'answer': row[3],
                'intent': row[4],
                'intent_confidence': row[5],
                'module_used': row[6],
                'routing_reason': row[7],
                'num_chunks_retrieved': row[8],
                'retrieval_method': row[9],
                'sources': json.loads(row[10]) if row[10] else [],
                'retrieved_chunks': json.loads(row[11]) if row[11] else [],
                'chunk_scores': json.loads(row[12]) if row[12] else [],
                'reranked': bool(row[13]),
                'query_optimized': bool(row[14]),
                'optimization_method': row[15],
                'latency_ms': row[16],
                'timestamp': row[17]
            })
        
        return responses
    
    def export_for_evaluation(self, output_file: str):
        """
        Export all logged responses to JSON for evaluation.
        
        Args:
            output_file: Path to output JSON file
        """
        responses = self.get_all_responses()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(responses, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Exported {len(responses)} responses to {output_file}")
    
    def close(self):
        """Close database connection."""
        self.conn.close()


if __name__ == "__main__":
    # Test the logger
    logger = EvalLogger()
    
    # Example: log a test response
    test_result = {
        'query': 'What events are happening this week?',
        'answer': 'There are 3 events this week...',
        'sources': [
            {'from': 'test@nyu.edu', 'subject': 'Event invitation', 'date': '2024-12-03'}
        ],
        'retrieved_chunks': [
            {'chunk_id': 'msg123_0', 'text': 'Sample chunk...'}
        ],
        'num_chunks_retrieved': 1,
        'intent': 'calendar',
        'intent_confidence': 0.95,
        'routing': {'module': 'predict', 'reason': 'Calendar query'},
        'retrieval_method': 'predict',
        'reranked': False,
        'query_optimized': True,
        'optimization_method': 'rewrite'
    }
    
    response_id = logger.log_response(1, test_result['query'], test_result, latency_ms=250)
    print(f"Logged test response with ID: {response_id}")
    
    logger.close()
