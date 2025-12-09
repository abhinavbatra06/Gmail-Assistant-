"""
Export evaluation logs to CSV format
"""
import sqlite3
import csv
import json
from pathlib import Path

def export_to_csv(output_file="Eval/eval_logs.csv"):
    """Export query history and responses to CSV"""
    
    # Use Path to handle relative paths from project root
    script_dir = Path(__file__).parent.parent
    db_path = script_dir / "db" / "memory.db"
    output_path = script_dir / output_file
    
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    
    # Join query_history and query_responses tables
    query = """
    SELECT 
        qh.id,
        qh.query,
        qh.intent,
        qh.timestamp as query_timestamp,
        qr.answer,
        qr.module_used,
        qr.retrieval_method,
        qr.num_chunks_retrieved,
        qr.latency_ms,
        qr.chunk_scores,
        qr.sources,
        qr.timestamp as response_timestamp
    FROM query_history qh
    LEFT JOIN query_responses qr ON qh.id = qr.query_id
    ORDER BY qh.id DESC
    """
    
    cur.execute(query)
    rows = cur.fetchall()
    
    print(f"Debug: Fetched {len(rows)} rows from database")
    
    # Write to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'query_id',
            'query',
            'intent',
            'query_timestamp',
            'answer',
            'routing_module',
            'retrieval_method',
            'num_chunks_retrieved',
            'latency_ms',
            'top_chunk_ids',
            'top_scores',
            'sources',
            'response_timestamp'
        ])
        
        # Data rows
        for row in rows:
            query_id, query, intent, query_ts, answer, routing, retrieval, num_chunks, latency, chunk_scores_json, sources_json, response_ts = row
            
            # Parse chunk scores
            top_chunk_ids = ""
            top_scores = ""
            if chunk_scores_json:
                try:
                    chunk_scores = json.loads(chunk_scores_json)
                    if chunk_scores:
                        top_chunk_ids = " | ".join([cs.get('chunk_id', '') for cs in chunk_scores[:5]])
                        top_scores = " | ".join([f"{cs.get('similarity', 0):.3f}" for cs in chunk_scores[:5]])
                except:
                    pass
            
            # Parse sources
            sources_str = ""
            if sources_json:
                try:
                    sources = json.loads(sources_json)
                    sources_str = " | ".join(sources[:5]) if sources else ""
                except:
                    pass
            
            writer.writerow([
                query_id,
                query,
                intent,
                query_ts,
                answer if answer else "",
                routing if routing else "",
                retrieval if retrieval else "",
                num_chunks if num_chunks else 0,
                latency if latency else 0,
                top_chunk_ids,
                top_scores,
                sources_str,
                response_ts if response_ts else ""
            ])
    
    conn.close()
    
    print(f"✅ Exported {len(rows)} queries to {output_path}")
    print(f"📄 File size: {output_path.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    export_to_csv()
