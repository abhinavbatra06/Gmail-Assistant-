"""
Batch Test Query Runner
Runs a set of test queries and saves results
"""
import sys
import time
from src.rag_query import RAGQuery

# Test queries organized by category
test_queries = {
    "Vague/Ambiguous": [
        "What's happening next week?",
        "Tell me about the party",
        "Who do I need to talk to about registration?",
        "What did Mohammad send me?",
        "Anything about cedar project lately?"
    ],
    "Typos/Casual": [
        "when is the capston thing due",
        "quiz deadlines pls",
        "whos hosting the holiday thing in december",
        "gpu stuff for project?",
        "any meetups or networking events?"
    ],
    "Multi-intent/Complex": [
        "Did I miss any important deadlines this week and what's coming up?",
        "Compare the homework and quiz schedules for my courses",
        "What buildings or floors are closed and why?",
        "Show me everything about presentations - both capstone and course related"
    ],
    "Implicit/Contextual": [
        "What do I need to prepare for the end of semester?",
        "Who's been emailing me most about courses?",
        "Any jobs or research positions I can apply to?",
        "What's the deal with AI tools for assignments?",
        "Tell me about Yann LeCun latest presentation",
        "Anything from WiDS board lately?"
    ],
    "Negation/Edge cases": [
        "What courses are NOT being offered in spring?"
    ]
}

def main():
    print("=" * 80)
    print("BATCH TEST QUERY RUNNER")
    print("=" * 80)
    
    rag = RAGQuery()
    
    total_queries = sum(len(queries) for queries in test_queries.values())
    query_num = 0
    
    for category, queries in test_queries.items():
        print(f"\n{'=' * 80}")
        print(f"CATEGORY: {category}")
        print(f"{'=' * 80}\n")
        
        for query in queries:
            query_num += 1
            print(f"\n[{query_num}/{total_queries}] Query: {query}")
            print("-" * 80)
            
            try:
                start = time.time()
                result = rag.query(query)
                elapsed = time.time() - start
                
                print(f"\nAnswer: {result['answer'][:500]}{'...' if len(result['answer']) > 500 else ''}")
                print(f"\nTime: {elapsed:.2f}s | Chunks: {result.get('num_chunks_retrieved', 0)} | Intent: {result.get('intent', 'N/A')}")
                
            except Exception as e:
                print(f"ERROR: {e}")
            
            print("-" * 80)
    
    print(f"\n{'=' * 80}")
    print(f"COMPLETED: {total_queries} queries tested")
    print(f"{'=' * 80}")
    print("\nRun 'python check_eval_logs.py' to see detailed results in memory.db")

if __name__ == "__main__":
    main()
