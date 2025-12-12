"""
Batch Test Query Runner
Runs a set of test queries and saves results
"""
import sys
import time
from src.rag_query import RAGQuery

# Test queries organized by category
# Test queries organized by category
test_queries = {
    "Vague/Ambiguous": [
        "Tell me about the party",
        "Who do I need to talk to about registration?",
        "What did Mohammad send me?",
        "Anything about cedar project lately?",
        "And any other details I should know about this",
        "Anything from WiDS board lately?",
        "Tell me about Yann LeCun latest presentation"
    ],

    "Typos/Casual": [
        "when is the capston thing due",
        "quiz deadlines pls",
        "whos hosting the holiday thing in december events calendar",
        "gpu stuff for project?",
        "any meetups or networking events?",
        "Any otjher details I should know about Spring 2026 course registration",
        "What is deadline for the deep learniung course quiz",
        "What is deadline for the deep learniung course"
    ],

    "Multi-intent/Complex": [
        "What's my schedule like and what job applications should I prioritize? events calendar",
        "Show me everything related to capstone projects and internships",
        "Which jobs require NYU enrollment and which don't?",
        "What opportunities involve LLMs or machine learning?",
        "Compare salaries across all the job postings",
        "What meetings and job opportunities do I have this week? events calendar",
        "Which positions have the earliest application deadlines?",
        "What job opportunities match my data science background?",
        "Compare the Lepercq internship with the Avra Janz RA position",
        "Show me all full-time vs internship positions",
        "What research assistant roles are available?",
        "Tell me about spring 2026 positions",
        "What opportunities have December deadlines?",
        "Show me ML and AI related jobs",
        "What's the highest paying opportunity available?",
        "Are there any virtual/remote opportunities?",
        "What positions require work authorization?",
        "What internships can start right away?",
        "Show me all communications from Zoë Levin about jobs",
        "Show me all job opportunities from Zoë Levin",
        "Show me all capstone touchbase meetings events calendar",
        "What meetings are scheduled for November 24th? events calendar",
        "Show me all meetings with Mark Freeman events calendar",
        "Who's attending the Matt + NYU Capstone Touchbase? events calendar",
        "Compare the homework and quiz schedules for my courses",
        "Did I miss any important deadlines this week and what's coming up?",
        "Summarize top research initiatives by CDS in last few emails"
    ],

    "Implicit/Contextual": [
        "Any jobs or research positions I can apply to?",
        "Who's been emailing me most about courses?",
        "What do I need to prepare for the end of semester?",
        "Show me everything about presentations - both capstone and course related",
        "What buildings or floors are closed and why?",
        "What are the NYU data science flyer opportunities?",
        "Show me the RL humanoid soccer competition details",
        "What's in the NYC tech interview presentation?",
        "Tell me about the DSC teaching assistant position",
        "What's the IPPE AI investment internship about?",
        "Show me the BME-CAIR summer program details",
        "What's in the quantitative analyst job description?",
        "Tell me about the molecular design platform job",
        "What's the salary for the sociology research assistant role?",
        "What ML scientist positions are available at FL117?",
        "What are the requirements for the autobiography dataset project?",
        "Show me details about the graduate research assistant job",
        "What's the deadline for the Paul Sonkin internship?",
        "Tell me about the NLP project hiring from Zoë",
        "What's the Lepercq internship about?",
        "When is quiz 5 due",
        "Are any grades released for my decision model class",
        "Tell me about the internship panel ?",
        "When does spring 2026 course registration start ?",
        "Give me list of courses being offered in the spring ?",
        "Give me link to spring 2026 registration",
        "Give me link to spring 2026 registration for cds"
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
