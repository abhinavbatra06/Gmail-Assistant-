"""
RAG Query Module

Implements Retrieval-Augmented Generation for querying emails.
Handles RAG queries over the email corpus.
Takes a user query, retrieves relevant chunks, and generates an answer using an LLM.
"""

import os
import json
import yaml
import re
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from openai import OpenAI
from src.vector_db import EmailVectorDB
from src.embedder import EmailEmbedder
from dotenv import load_dotenv

load_dotenv()


class RAGQuery:
    
    def __init__(self, config_path="config.yaml"):
        """
        Initialize RAG query handler.
        
        Args:
            config_path: Path to config file
        """
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        
        rag_cfg = self.cfg.get("rag", {})
        embedding_cfg = self.cfg.get("embedding", {})
        
        # initialize vector db
        self.vector_db = EmailVectorDB(config_path=config_path)
        
        # initialize embedder for query embeddings
        self.embedder = EmailEmbedder(config_path=config_path)
        
        # initialize Router module
        self.router = None
        if rag_cfg.get("enable_routing", True):
            try:
                from src.router import Router
                self.router = Router(config_path=config_path)
                print(f"   Router module: enabled")
            except Exception as e:
                print(f"   Warning: Router module not available: {str(e)}")
        
        # initialize Predict module for calendar queries
        self.predict = None
        predict_db_path = rag_cfg.get("predict_db_path", "db/events.db")
        if rag_cfg.get("enable_predict", True):
            try:
                from src.predict import Predict
                self.predict = Predict(predict_db_path)
                event_count = self.predict.count()
                stats = self.predict.get_deduplication_stats()
                print(f"   Predict module: {event_count} unique events")
                if stats["events_with_multiple_emails"] > 0:
                    print(f"   Deduplication: {stats['events_with_multiple_emails']} events found in multiple emails")
            except Exception as e:
                print(f"   Warning: Predict module not available: {str(e)}")
        
        # initialize Memory module for user preferences
        self.memory = None
        memory_db_path = rag_cfg.get("memory_db_path", "db/memory.db")
        if rag_cfg.get("enable_memory", True):
            try:
                from src.memory import Memory
                self.memory = Memory(memory_db_path)
                print(f"   Memory module: enabled")
            except Exception as e:
                print(f"   Warning: Memory module not available: {str(e)}")
        
        # Small2big retrieval settings
        self.enable_small2big = rag_cfg.get("enable_small2big", False)
        self.small2big_expansion_k = rag_cfg.get("small2big_expansion_k", 3)  # expand to 3x chunks
        
        # hybrid retrieval settings (bm25 + dense)
        self.enable_hybrid_retrieval = rag_cfg.get("enable_hybrid_retrieval", False)
        self.hybrid_alpha = rag_cfg.get("hybrid_alpha", 0.5)  # 0.0 = only bm25, 1.0 = only dense, 0.5 = balanced
        
        # iterative retrieval settings
        self.enable_iterative_retrieval = rag_cfg.get("enable_iterative_retrieval", False)
        self.max_iterations = rag_cfg.get("max_iterations", 3)  # max retrieval iterations
        
        # reranking settings
        self.enable_reranking = rag_cfg.get("enable_reranking", False)
        self.rerank_top_k = rag_cfg.get("rerank_top_k", 20)  # retrieve more, then rerank to top_k
        self.rerank_method = rag_cfg.get("rerank_method", "llm")  # "llm" or "cross_encoder"
        
        # RAG settings
        self.top_k = rag_cfg.get("top_k", 5)
        self.context_window = rag_cfg.get("context_window", 2000)
        self.response_model = rag_cfg.get("response_model", "gpt-4o-mini")
        self.query_optimization = rag_cfg.get("query_optimization", "rewrite")  # options: "rewrite", "hyde", "none"
        self.enable_query_expansion = rag_cfg.get("enable_query_expansion", False)  # explicit query expansion
        self.enable_subquery_decomposition = rag_cfg.get("enable_subquery_decomposition", False)  # break complex queries into sub-queries
        self.enable_intent_routing = rag_cfg.get("enable_intent_routing", False)  # intent-based query routing
        
        # openai client
        api_key_env = embedding_cfg.get("openai_api_key_env", "OPENAI_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"OpenAI API key not found in environment variable: {api_key_env}")
        
        self.client = OpenAI(api_key=api_key)
        
        # current date for temporal context
        self.current_date = datetime.now()
        
        print(f"RAG Query initialized:")
        print(f"   Top K: {self.top_k}")
        print(f"   Context window: {self.context_window} tokens")
        print(f"   Response model: {self.response_model}")
        print(f"   Query optimization: {self.query_optimization}")
        print(f"   Query expansion: {self.enable_query_expansion}")
        print(f"   Sub-query decomposition: {self.enable_subquery_decomposition}")
        print(f"   Intent routing: {self.enable_intent_routing}")
        print(f"   Hybrid retrieval (BM25+Dense): {self.enable_hybrid_retrieval}")
        print(f"   Iterative retrieval: {self.enable_iterative_retrieval}")
        print(f"   Reranking: {self.enable_reranking} (method: {self.rerank_method})")
        print(f"   Current date: {self.current_date.strftime('%A, %B %d, %Y')}")
        print(f"   Vector DB chunks: {self.vector_db.count()}")
    
    def _rewrite_query(self, query: str) -> str:
        """
        Rewrite the user's query to be more specific and searchable while preserving the original intent.
        
        Args:
            query: Original user query
            
        Returns:
            Rewritten query string
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "system",
                    "content": """You are a query rewriter for an email search system.
                    Rewrite the user's query to be more specific and searchable while preserving the original intent.
                    - Expand abbreviations (AI → Artificial Intelligence, ML → Machine Learning)
                    - Add relevant synonyms and related terms
                    - Preserve temporal intent (this week, upcoming, recent, etc.)
                    - Keep important keywords and phrases
                    - Make the query more suitable for semantic search in email content
                    Output ONLY the rewritten query, nothing else. Do not add explanations or additional text."""
                }, {
                    "role": "user",
                    "content": query
                }],
                temperature=0.3
            )
            rewritten = response.choices[0].message.content.strip()
            return rewritten if rewritten else query 
        except Exception as e:
            print(f"   Warning: Query rewriting failed: {str(e)}")
            return query 

    def _generate_hypothetical_answer(self, query: str) -> str:
        """
        Generate a hypothetical answer/document using HyDE method.
        
        Args:
            query: User's original query
            
        Returns:
            Hypothetical answer/document text
        """
        current_date_str = self.current_date.strftime("%A, %B %d, %Y")
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": f"""You are generating a hypothetical answer document for a query.
Today's date is {current_date_str}.
Generate a realistic, detailed answer that would be found in an email that answers this query.
Write it as if it were an actual email response - be specific, include relevant details, and use natural language.
Do NOT include meta-commentary like "This email would contain..." - just write the answer directly.
Keep it concise but informative (2-4 sentences typically)."""
            }, {
                "role": "user",
                "content": f"Generate a hypothetical answer for this query: {query}"
            }],
            temperature=0.7,
            max_tokens=300
        )
        hypothetical = response.choices[0].message.content.strip()
        return hypothetical if hypothetical else query
    
    def _expand_query_conservative(self, query: str) -> str:
        """
        Conservative query expansion for event queries - focuses on exact location matches.
        
        Args:
            query: Original query string
            
        Returns:
            Expanded query string with minimal, focused expansion
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": """You are a query expansion system for event/calendar queries.
Expand the query conservatively - only add:
- Exact location name variations (e.g., "MakerSpace" → "MakerSpace, Maker Space, NYU MakerSpace")
- Common abbreviations (e.g., "NYU" → "NYU, New York University")
- Preserve all original terms, especially location names
- Do NOT add broad synonyms like "activities, occurrences, happenings"
- Do NOT add unrelated location synonyms
- Keep temporal modifiers intact
Output ONLY the expanded query, nothing else."""
            }, {
                "role": "user",
                "content": f"Expand this event query conservatively: {query}"
            }],
            temperature=0.2,
            max_tokens=150
        )
        expanded = response.choices[0].message.content.strip()
        return expanded if expanded else query
    
    def _expand_query(self, query: str) -> str:
        """
        Explicitly expand query with synonyms, related terms, and domain-specific mappings.
        
        Args:
            query: Original query string
            
        Returns:
            Expanded query string with additional terms
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": """You are a query expansion system for email search.
Expand the query by adding relevant synonyms, related terms, and domain-specific variations.
- Add synonyms and related terms (e.g., "meeting" → "meeting, appointment, conference, call")
- Expand abbreviations and acronyms (e.g., "CS" → "Computer Science, CS", "ML" → "Machine Learning, ML")
- Add common variations (e.g., "deadline" → "deadline, due date, submission date")
- Include related concepts (e.g., "assignment" → "assignment, homework, project, task")
- Preserve all original terms
- Keep temporal and other modifiers intact
Output ONLY the expanded query with all terms, nothing else."""
            }, {
                "role": "user",
                "content": f"Expand this query: {query}"
            }],
            temperature=0.4,
            max_tokens=200
        )
        expanded = response.choices[0].message.content.strip()
        return expanded if expanded else query
    
    def _decompose_query(self, query: str) -> List[str]:
        """
        Decompose complex queries into sub-queries for better retrieval.
        Detects multi-part questions, AND/OR logic, and breaks them into separate queries.
        
        Args:
            query: Original complex query
            
        Returns:
            List of sub-queries, or [query] if decomposition not needed
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": """You are a query decomposition system for email search.
Analyze the query and determine if it contains multiple distinct questions or parts.
If the query is simple or single-faceted, return it as-is.
If it contains multiple parts (connected by AND, OR, "and also", "as well as", etc.), 
break it into separate sub-queries.

Examples:
- "What events are happening this week AND what deadlines do I have?" → ["What events are happening this week?", "What deadlines do I have?"]
- "Show me emails about AI courses OR machine learning seminars" → ["Show me emails about AI courses", "Show me emails about machine learning seminars"]
- "Find emails from NYU about assignments" → ["Find emails from NYU about assignments"] (single query)

Output format: One sub-query per line, no numbering, no explanations.
If only one query, output just that query."""
            }, {
                "role": "user",
                "content": f"Decompose this query: {query}"
            }],
            temperature=0.3,
            max_tokens=300
        )
        
        decomposed_text = response.choices[0].message.content.strip()
        
        # parse response - split by newlines and clean
        sub_queries = [q.strip() for q in decomposed_text.split('\n') if q.strip()]
        
        # remove any numbering or bullets if present
        sub_queries = [q.lstrip('0123456789.-) ').strip() for q in sub_queries]
        
        # filter out empty strings
        sub_queries = [q for q in sub_queries if q]
        
        # if decomposition failed or only one query, return original
        if not sub_queries or len(sub_queries) == 1:
            return [query]
        
        return sub_queries
    
    def _merge_search_results(self, all_results: List[Dict], top_k: int) -> Dict:
        """
        Merge search results from multiple sub-queries.
        Combines results, deduplicates by chunk ID, and ranks by relevance.
        
        Args:
            all_results: List of result dicts from multiple searches
            top_k: Target number of results to return
            
        Returns:
            Merged results dict with deduplicated and ranked chunks
        """
        if not all_results:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        # if only one result set, return it
        if len(all_results) == 1:
            return all_results[0]
        
        # collect all chunks with their metadata
        chunk_map = {}  # chunk_id -> (doc, metadata, distance, source_query_idx)
        
        for query_idx, results in enumerate(all_results):
            if not results.get("ids") or not results["ids"][0]:
                continue
            
            ids = results["ids"][0] if isinstance(results["ids"][0], list) else results["ids"]
            documents = results["documents"][0] if isinstance(results["documents"][0], list) else results["documents"]
            metadatas = results["metadatas"][0] if isinstance(results["metadatas"][0], list) else results["metadatas"]
            distances = results["distances"][0] if isinstance(results["distances"][0], list) else results["distances"]
            
            for chunk_id, doc, metadata, distance in zip(ids, documents, metadatas, distances):
                # if chunk already seen, keep the one with lower distance
                if chunk_id in chunk_map:
                    existing_distance = chunk_map[chunk_id][2]
                    if distance < existing_distance:
                        chunk_map[chunk_id] = (doc, metadata, distance, query_idx)
                else:
                    chunk_map[chunk_id] = (doc, metadata, distance, query_idx)
        
        # sort by distance (lower is better) and take top_k
        sorted_chunks = sorted(chunk_map.items(), key=lambda x: x[1][2])[:top_k]
        
        # reconstruct results format
        merged_ids = []
        merged_docs = []
        merged_metadatas = []
        merged_distances = []
        
        for chunk_id, (doc, metadata, distance, _) in sorted_chunks:
            merged_ids.append(chunk_id)
            merged_docs.append(doc)
            merged_metadatas.append(metadata)
            merged_distances.append(distance)
        
        return {
            "ids": [merged_ids],
            "documents": [merged_docs],
            "metadatas": [merged_metadatas],
            "distances": [merged_distances]
        }
    
    def _classify_intent(self, query: str) -> Dict[str, any]:
        """
        Classify query intent to route to appropriate retrieval strategy.
        
        Args:
            query: User query string
            
        Returns:
            Dict with intent type, confidence, and intent-specific parameters
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": """You are an intent classifier for email queries.
Classify the user's query into one of these intents:
1. "calendar" - Questions about events, meetings, calendar invites, schedules
2. "deadline" - Questions about due dates, assignments, deadlines, submissions
3. "sender" - Questions about emails from specific senders or organizations
4. "information" - Questions seeking specific information or facts from emails
5. "summarization" - Requests to summarize threads, conversations, or topics
6. "general" - General email search queries

Output ONLY a JSON object with this exact format:
{"intent": "calendar|deadline|sender|information|summarization|general", "confidence": 0.0-1.0}

Be specific - if a query mentions events, meetings, or calendar, classify as "calendar".
If it mentions due dates, assignments, or deadlines, classify as "deadline".
If it asks about a specific sender, classify as "sender".
If it asks to summarize, classify as "summarization".
Otherwise, classify as "general"."""
            }, {
                "role": "user",
                "content": f"Classify this query: {query}"
            }],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            intent = result.get("intent", "general").lower()
            confidence = result.get("confidence", 0.5)
            
            # validate intent
            valid_intents = ["calendar", "deadline", "sender", "information", "summarization", "general"]
            if intent not in valid_intents:
                intent = "general"
            
            return {
                "intent": intent,
                "confidence": confidence,
                "original_query": query
            }
        except (json.JSONDecodeError, KeyError) as e:
            # fallback to general intent
            return {
                "intent": "general",
                "confidence": 0.5,
                "original_query": query
            }
    
    def _get_intent_routing_strategy(self, intent_info: Dict) -> Dict:
        """
        Get retrieval strategy parameters based on intent.
        
        Args:
            intent_info: Intent classification result
            
        Returns:
            Dict with intent-specific retrieval parameters
        """
        intent = intent_info["intent"]
        
        strategies = {
            "calendar": {
                "top_k_multiplier": 1.5,  # get more results for calendar queries
                "preferred_source_types": ["calendar", "attachment"],
                "metadata_filters": {"source_type": {"$in": ["calendar", "attachment"]}},
                "query_optimization": "hyde",  # HyDE works well for calendar queries
                "prioritize_attachments": True,
                "description": "Calendar/event queries - prioritizing calendar attachments"
            },
            "deadline": {
                "top_k_multiplier": 1.2,
                "preferred_source_types": ["email", "attachment"],
                "metadata_filters": None,  # no specific filter, but look for assignment language
                "query_optimization": "rewrite",  # query rewriting helps with deadline queries
                "prioritize_attachments": False,
                "description": "Deadline/assignment queries - looking for due dates and assignments"
            },
            "sender": {
                "top_k_multiplier": 1.0,
                "preferred_source_types": ["email"],
                "metadata_filters": None,  # will be set based on query analysis
                "query_optimization": "rewrite",
                "prioritize_attachments": False,
                "description": "Sender-based queries - filtering by sender metadata"
            },
            "information": {
                "top_k_multiplier": 1.0,
                "preferred_source_types": ["email", "attachment"],
                "metadata_filters": None,
                "query_optimization": "rewrite",
                "prioritize_attachments": False,
                "description": "Information extraction queries - focused retrieval"
            },
            "summarization": {
                "top_k_multiplier": 2.0,  # get more chunks for summarization
                "preferred_source_types": ["email"],
                "metadata_filters": None,
                "query_optimization": "rewrite",
                "prioritize_attachments": False,
                "description": "Summarization queries - retrieving more context"
            },
            "general": {
                "top_k_multiplier": 1.0,
                "preferred_source_types": ["email", "attachment"],
                "metadata_filters": None,
                "query_optimization": None,  # use default from config
                "prioritize_attachments": False,
                "description": "General email search queries"
            }
        }
        
        return strategies.get(intent, strategies["general"])
    
    def _extract_sender_from_query(self, query: str) -> Optional[str]:
        """
        Extract sender information from query if present.
        
        Args:
            query: User query string
            
        Returns:
            Sender email or domain if found, None otherwise
        """
        # look for email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, query)
        if emails:
            return emails[0]
        
        # look for domain mentions (e.g., "from NYU", "nyu.edu")
        domain_keywords = ["from", "by", "sent by"]
        query_lower = query.lower()
        for keyword in domain_keywords:
            if keyword in query_lower:
                # try to extract what comes after
                parts = query_lower.split(keyword, 1)
                if len(parts) > 1:
                    after = parts[1].strip().split()[0] if parts[1].strip() else None
                    if after:
                        # check if it's a domain-like string
                        if "." in after or after.endswith("edu") or after.endswith("com"):
                            return after
        
        return None
    
    def _query_with_predict(self, query: str, top_k: int) -> Dict:
        """
        Query using Predict module for calendar events.
        
        Args:
            query: User query
            top_k: Number of results
            
        Returns:
            Dict with answer and events
        """
        if not self.predict:
            return {
                "query": query,
                "answer": "Predict module not available.",
                "sources": [],
                "retrieved_chunks": []
            }
        
        # parse query for date ranges and location
        query_lower = query.lower()
        events = []
        
        # extract location keywords from query (common patterns)
        location_keywords = None
        location_patterns = [
            "at the", "at", "in", "location", "place", "venue"
        ]
        query_words = query_lower.split()
        # filter out common stop words
        stop_words = {"the", "a", "an", "and", "or", "for", "to", "of", "on", "with"}
        # try to find location after "at" or "in"
        for i, word in enumerate(query_words):
            if word in ["at", "in"] and i + 1 < len(query_words):
                # take next 2-3 words as potential location
                location_parts = query_words[i+1:min(i+4, len(query_words))]
                
                location_parts = [w for w in location_parts if w not in stop_words]
                if location_parts:
                    location_keywords = " ".join(location_parts)
                    break
        
        # if no explicit location found, try searching for common location terms in query
        if not location_keywords:
            # look for location-related terms in the query itself
            location_terms = []
            for word in query_words:
                if word not in stop_words and len(word) > 3:  # skip short words
                    location_terms.append(word)
            if location_terms:
                location_keywords = " ".join(location_terms[-3:])  # take last 3 words as potential location
        
        # query events with date and location filters
        start_date = None
        end_date = None
        
        if "this week" in query_lower:
            events = self.predict.get_events_this_week()
        elif "upcoming" in query_lower or "coming" in query_lower or "recent" in query_lower:
            if "recent" in query_lower:
                # recent events: last 30 days
                from datetime import timedelta
                start_date = self.current_date - timedelta(days=30)
            else:
                # upcoming events: next 7 days
                events = self.predict.get_upcoming_events(days=7)
        else:
            # general search - try with location filter first
            if location_keywords:
                # use query_events with location filter
                events = self.predict.query_events(
                    location=location_keywords,
                    start_date=start_date,
                    limit=top_k * 2  # get more to account for filtering
                )
                # if no results with location, try without location
                if not events:
                    events = self.predict.search_events(query, limit=top_k)
            else:
                # general search
                events = self.predict.search_events(query, limit=top_k)
        
        # if no events found in Predict module, return None to trigger fallback
        if not events:
            print("⚠️  No events found in Predict module, will fall back to RAG retrieval...")
            return None
        
        events_text = self.predict.format_events_for_answer(events[:top_k])
        
        # generate answer with LLM
        current_date_str = self.current_date.strftime("%A, %B %d, %Y")
        system_prompt = f"""You are a helpful assistant that answers questions about calendar events.
Today's date is {current_date_str}.
Use the provided event information to answer the user's question.
Be concise and accurate."""
        
        user_prompt = f"""Based on the following calendar events, answer this question: {query}

Events:
{events_text}

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.response_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.context_window,
                temperature=0.7
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
        
        # format sources
        sources = []
        for event in events[:top_k]:
            sources.append({
                "from": event.get("organizer", "Unknown"),
                "subject": event.get("summary", "Untitled Event"),
                "date": event.get("start_time_str", "Unknown"),
                "source_type": "calendar",
                "message_id": event.get("message_ids", ["unknown"])[0] if event.get("message_ids") else "unknown"
            })
        
        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": [],
            "num_chunks_retrieved": len(sources),
            "used_predict": True
        }
    
    def _expand_to_full_context(self, small_chunks: Dict, expansion_k: int = 3) -> Dict:
        """
        Small2big retrieval: expand small chunks to full email context.
        
        Args:
            small_chunks: Results from initial small chunk search
            expansion_k: Multiplier for how many chunks to retrieve per message
            
        Returns:
            Expanded results with full context
        """
        if not small_chunks.get("ids") or not small_chunks["ids"][0]:
            return small_chunks
        
        # extract message ids from small chunks
        ids = small_chunks["ids"][0] if isinstance(small_chunks["ids"][0], list) else small_chunks["ids"]
        metadatas = small_chunks["metadatas"][0] if isinstance(small_chunks["metadatas"][0], list) else small_chunks["metadatas"]
        
        # group chunks by message_id
        message_chunks = {}  # message_id -> list of chunk indices
        for i, metadata in enumerate(metadatas):
            msg_id = metadata.get("message_id", "")
            if msg_id:
                if msg_id not in message_chunks:
                    message_chunks[msg_id] = []
                message_chunks[msg_id].append(i)
        
        # for each message, retrieve more chunks
        expanded_ids = []
        expanded_docs = []
        expanded_metadatas = []
        expanded_distances = []
        
        distances = small_chunks["distances"][0] if isinstance(small_chunks["distances"][0], list) else small_chunks["distances"]
        documents = small_chunks["documents"][0] if isinstance(small_chunks["documents"][0], list) else small_chunks["documents"]
        
        for msg_id, chunk_indices in message_chunks.items():
            # get the best distance from this message's chunks
            msg_distances = [distances[i] for i in chunk_indices]
            best_distance = min(msg_distances)
            
            # retrieve more chunks from this message
            # use metadata filter to get all chunks from this message
            msg_filter = {"message_id": {"$eq": msg_id}}
            
            # get a sample embedding (use first chunk's embedding if available)
            # current implementation uses the vector DB's search with message filter
            # this is a simplified approach; to extend this the approach might be to
            # retrieve all chunks for the message and re-rank
            
            # add original chunks
            for idx in chunk_indices:
                expanded_ids.append(ids[idx])
                expanded_docs.append(documents[idx])
                expanded_metadatas.append(metadatas[idx])
                expanded_distances.append(distances[idx])
        
        return {
            "ids": [expanded_ids],
            "documents": [expanded_docs],
            "metadatas": [expanded_metadatas],
            "distances": [expanded_distances]
        }
    
    def _hybrid_search(self, query: str, query_embedding: List[float], 
                       top_k: int, filter_metadata: Optional[Dict] = None) -> Dict:
        """
        Hybrid retrieval combining BM25 (keyword) and dense (semantic) search.
        
        Args:
            query: Original query string (for BM25)
            query_embedding: Query embedding (for dense search)
            top_k: Number of results
            filter_metadata: Optional metadata filter
            
        Returns:
            Combined search results
        """
        # get dense search results
        dense_results = self.vector_db.search(
            query_embedding=query_embedding,
            n_results=top_k * 2,  # get more for reranking
            where=filter_metadata
        )
        
        # get all chunks for bm25 (need to load them)
        # using the chunks from dense results for efficiency
        if not dense_results.get("ids") or not dense_results["ids"][0]:
            return dense_results
        
        ids = dense_results["ids"][0] if isinstance(dense_results["ids"][0], list) else dense_results["ids"]
        documents = dense_results["documents"][0] if isinstance(dense_results["documents"][0], list) else dense_results["documents"]
        metadatas = dense_results["metadatas"][0] if isinstance(dense_results["metadatas"][0], list) else dense_results["metadatas"]
        distances = dense_results["distances"][0] if isinstance(dense_results["distances"][0], list) else dense_results["distances"]
        
        # build chunks for bm25
        chunks = []
        for chunk_id, doc, metadata in zip(ids, documents, metadatas):
            chunks.append({
                "chunk_id": chunk_id,
                "text": doc,
                "metadata": metadata
            })
        
        # init bm25 retriever
        try:
            from src.bm25_retriever import BM25Retriever
            bm25_retriever = BM25Retriever(chunks)
            bm25_results = bm25_retriever.search(query, top_k=len(chunks))
            
            # create score maps
            bm25_scores = {r["chunk_id"]: r["score"] for r in bm25_results}
            
            # normalize scores (0-1 range)
            max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
            max_dense = max(distances) if distances else 1.0
            
            # combine scores: alpha * dense + (1 - alpha) * bm25
            combined_scores = []
            for i, chunk_id in enumerate(ids):
                # convert distance to similarity (lower distance = higher similarity)
                dense_sim = 1.0 - (distances[i] / max_dense) if max_dense > 0 else 0.0
                bm25_sim = (bm25_scores.get(chunk_id, 0.0) / max_bm25) if max_bm25 > 0 else 0.0
                
                # hybrid score
                hybrid_score = self.hybrid_alpha * dense_sim + (1 - self.hybrid_alpha) * bm25_sim
                combined_scores.append((i, hybrid_score))
            
            # sort by combined score
            combined_scores.sort(key=lambda x: x[1], reverse=True)
            
            # rebuild results with top_k
            top_indices = [idx for idx, _ in combined_scores[:top_k]]
            
            return {
                "ids": [[ids[i] for i in top_indices]],
                "documents": [[documents[i] for i in top_indices]],
                "metadatas": [[metadatas[i] for i in top_indices]],
                "distances": [[1.0 - combined_scores[i][1] for i in range(len(top_indices))]]  # convert back to distance
            }
        except ImportError:
            print("⚠️  rank-bm25 not installed, falling back to dense search only")
            return dense_results
    
    def _rerank_results(self, query: str, results: Dict, top_k: int) -> Dict:
        """
        Rerank retrieval results using LLM or cross-encoder for better relevance.
        
        Args:
            query: Original user query
            results: Initial retrieval results
            top_k: Final number of results to return
            
        Returns:
            Reranked results
        """
        if not results.get("ids") or not results["ids"][0]:
            return results
        
        ids = results["ids"][0] if isinstance(results["ids"][0], list) else results["ids"]
        documents = results["documents"][0] if isinstance(results["documents"][0], list) else results["documents"]
        metadatas = results["metadatas"][0] if isinstance(results["metadatas"][0], list) else results["metadatas"]
        distances = results["distances"][0] if isinstance(results["distances"][0], list) else results["distances"]
        
        if len(ids) <= top_k:
            # already have fewer results than top_k, no need to rerank
            return results
        
        print(f"🔄 Reranking {len(ids)} results to top {top_k}...")
        
        if self.rerank_method == "llm":
            return self._llm_rerank(query, ids, documents, metadatas, distances, top_k)
        elif self.rerank_method == "cross_encoder":
            return self._cross_encoder_rerank(query, ids, documents, metadatas, distances, top_k)
        else:
            # fallback: simple distance-based reranking
            return self._simple_rerank(ids, documents, metadatas, distances, top_k)
    
    def _llm_rerank(self, query: str, ids: List, documents: List, 
                    metadatas: List, distances: List, top_k: int) -> Dict:
        """
        Rerank using LLM to score relevance.
        
        Args:
            query: User query
            ids: Chunk IDs
            documents: Chunk texts
            metadatas: Chunk metadata
            distances: Original distances
            top_k: Number of results to return
            
        Returns:
            Reranked results
        """
        # prepare chunks for LLM scoring
        chunks_with_context = []
        for i, (chunk_id, doc, metadata) in enumerate(zip(ids, documents, metadatas)):
            # add metadata context
            from_addr = metadata.get("from", "")
            subject = metadata.get("subject", "")
            date = metadata.get("date", "")
            
            context = f"From: {from_addr}\nSubject: {subject}\nDate: {date}\nContent: {doc[:500]}"
            chunks_with_context.append({
                "index": i,
                "chunk_id": chunk_id,
                "context": context,
                "score": None
            })
        
        # score chunks in batches
        batch_size = 10  # score 10 at a time to avoid token limits
        scored_chunks = []
        
        for batch_start in range(0, len(chunks_with_context), batch_size):
            batch = chunks_with_context[batch_start:batch_start + batch_size]
            
            # prompt for scoring
            chunks_text = "\n\n".join([
                f"[Chunk {i+1}]\n{chunk['context']}"
                for i, chunk in enumerate(batch)
            ])
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "system",
                        "content": """You are a relevance scorer for email search results.
Score each chunk from 0.0 to 1.0 based on how relevant it is to the query.
Output ONLY a JSON object with a "scores" array in the same order as the chunks.
Example: {"scores": [0.9, 0.3, 0.7]}"""
                    }, {
                        "role": "user",
                        "content": f"Query: {query}\n\nChunks:\n{chunks_text}\n\nScores (JSON):"
                    }],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                scores = result.get("scores", [])
                
                # assign scores
                if isinstance(scores, list):
                    for i, score in enumerate(scores):
                        if batch_start + i < len(chunks_with_context):
                            chunks_with_context[batch_start + i]["score"] = float(score)
            except Exception as e:
                print(f"   Warning: LLM reranking failed for batch: {str(e)}")
                # fallback to distance-based scoring
                for chunk in batch:
                    chunk["score"] = 1.0 - distances[chunk["index"]] if distances else 0.5
        
        # sort by score and take top_k
        scored_chunks = sorted(chunks_with_context, key=lambda x: x["score"] or 0.0, reverse=True)[:top_k]
        
        # rebuild results
        reranked_ids = [chunk["chunk_id"] for chunk in scored_chunks]
        reranked_docs = []
        reranked_metas = []
        reranked_dists = []
        
        id_to_index = {chunk_id: i for i, chunk_id in enumerate(ids)}
        for chunk_id in reranked_ids:
            idx = id_to_index[chunk_id]
            reranked_docs.append(documents[idx])
            reranked_metas.append(metadatas[idx])
            # use original distance or convert score to distance
            chunk_score = next((c["score"] for c in scored_chunks if c["chunk_id"] == chunk_id), None)
            reranked_dists.append(1.0 - chunk_score if chunk_score is not None else distances[idx])
        
        return {
            "ids": [reranked_ids],
            "documents": [reranked_docs],
            "metadatas": [reranked_metas],
            "distances": [reranked_dists]
        }
    
    def _cross_encoder_rerank(self, query: str, ids: List, documents: List,
                              metadatas: List, distances: List, top_k: int) -> Dict:
        """
        Rerank using cross-encoder model (if available).
        
        Args:
            query: User query
            ids: Chunk IDs
            documents: Chunk texts
            metadatas: Chunk metadata
            distances: Original distances
            top_k: Number of results to return
            
        Returns:
            Reranked results
        """
        try:
            from sentence_transformers import CrossEncoder
            # init cross-encoder (lazy loading)
            if not hasattr(self, '_cross_encoder'):
                self._cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            #prepare query-document pairs
            pairs = [[query, doc[:512]] for doc in documents]  # limit doc length
            
            # get scores
            scores = self._cross_encoder.predict(pairs)
            
            # sort by score
            scored_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            top_indices = scored_indices[:top_k]
            
            # rebuild results
            return {
                "ids": [[ids[i] for i in top_indices]],
                "documents": [[documents[i] for i in top_indices]],
                "metadatas": [[metadatas[i] for i in top_indices]],
                "distances": [[1.0 - float(scores[i]) for i in top_indices]]  # convert score to distance
            }
        except ImportError:
            print("   Warning: sentence-transformers not installed, falling back to LLM reranking")
            return self._llm_rerank(query, ids, documents, metadatas, distances, top_k)
        except Exception as e:
            print(f"   Warning: Cross-encoder reranking failed: {str(e)}, falling back to LLM")
            return self._llm_rerank(query, ids, documents, metadatas, distances, top_k)
    
    def _simple_rerank(self, ids: List, documents: List, metadatas: List,
                      distances: List, top_k: int) -> Dict:
        """
        Simple reranking based on distance (already sorted, just take top_k).
        
        Args:
            ids: Chunk IDs
            documents: Chunk texts
            metadatas: Chunk metadata
            distances: Distances
            top_k: Number of results
            
        Returns:
            Top-k results
        """
        # results are already sorted by distance, just take top_k
        return {
            "ids": [ids[:top_k]],
            "documents": [documents[:top_k]],
            "metadatas": [metadatas[:top_k]],
            "distances": [distances[:top_k]]
        }
    
    def _iterative_retrieval(self, query: str, initial_results: Dict, 
                            top_k: int, max_iterations: int = 3) -> Dict:
        """
        Iterative retrieval: refine query based on initial results.
        
        Args:
            query: Original query
            initial_results: Initial retrieval results
            top_k: Target number of results
            max_iterations: Maximum number of iterations
            
        Returns:
            Refined search results
        """
        if not initial_results.get("ids") or not initial_results["ids"][0]:
            return initial_results
        
        current_results = initial_results
        current_query = query
        
        for iteration in range(1, max_iterations):
            # analyze current results to refine query
            documents = current_results["documents"][0] if isinstance(current_results["documents"][0], list) else current_results["documents"]
            metadatas = current_results["metadatas"][0] if isinstance(current_results["metadatas"][0], list) else current_results["metadatas"]
            
            # extract key terms from top results
            top_docs = documents[:min(3, len(documents))]
            context = "\n".join(top_docs[:2])  # Use top 2 for context
            
            # generate refined query using LLM
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "system",
                        "content": """You are a query refinement system.
Based on the initial query and retrieved results, generate a more specific query that will help find better results.
Focus on key terms, entities, and concepts from the results.
Output ONLY the refined query, nothing else."""
                    }, {
                        "role": "user",
                        "content": f"Original query: {query}\n\nRetrieved context:\n{context}\n\nGenerate a refined query:"
                    }],
                    temperature=0.3,
                    max_tokens=100
                )
                
                refined_query = response.choices[0].message.content.strip()
                
                if refined_query != current_query and len(refined_query) > 0:
                    print(f"   Iteration {iteration + 1}: Refined query: {refined_query}")
                    current_query = refined_query
                    
                    # search with refined query
                    refined_embedding = self._embed_query(refined_query)
                    refined_results = self.vector_db.search(
                        query_embedding=refined_embedding,
                        n_results=top_k * 2
                    )
                    
                    # merge with previous results (keep best from both)
                    current_results = self._merge_search_results([current_results, refined_results], top_k)
                else:
                    # no improvement; stop iterating
                    break
            except Exception as e:
                print(f"   Warning: Iterative refinement failed: {str(e)}")
                break
        
        return current_results
    
    def _embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query.
        
        Args:
            query: User query string
            
        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            model=self.embedder.model,
            input=[query]
        )
        return response.data[0].embedding
    
    def _parse_date_from_metadata(self, date_string: str) -> Optional[datetime]:
        """
        Parse date string from metadata to datetime object.
        
        Args:
            date_string: Date string from metadata
            
        Returns:
            datetime object or None if parsing fails
        """
        if not date_string or date_string == "Unknown date":
            return None
        
        try:
            # try parsing email date format
            dt = parsedate_to_datetime(date_string)
            return dt
        except:
            try:
                # try ISO format
                return datetime.fromisoformat(date_string.replace('Z', '+00:00'))
            except:
                return None
    
    def _is_temporal_query(self, query: str) -> bool:
        """
        Check if query contains temporal terms that require date filtering.
        
        Args:
            query: User query string
            
        Returns:
            True if query mentions temporal terms
        """
        temporal_terms = [
            "this week", "next week", "upcoming", "recent", "latest",
            "today", "tomorrow", "this month", "next month",
            "soon", "future", "coming", "scheduled"
        ]
        query_lower = query.lower()
        return any(term in query_lower for term in temporal_terms)
    
    def _get_date_filter_for_query(self, query: str) -> Optional[Dict]:
        """
        Generate date filter based on query terms.
        
        Args:
            query: User query string
            
        Returns:
            ChromaDB date filter dict or None
        """
        if not self._is_temporal_query(query):
            return None
        
        query_lower = query.lower()
        
        # for "this week" or "upcoming" queries, filter to current week and future
        if "this week" in query_lower or "upcoming" in query_lower or "coming" in query_lower:
            # get start of current week (Sunday)
            # weekday() returns: Monday=0, Tuesday=1, ..., Sunday=6
            # for Sunday-based week: if today is Sunday (6), days_since_sunday = 0
            # if today is Monday (0), days_since_sunday = 1, etc.
            days_since_sunday = (self.current_date.weekday() + 1) % 7
            week_start = self.current_date - timedelta(days=days_since_sunday)
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # filter to current week onwards (include emails from start of this week)
            # use timestamp for filtering
            min_timestamp = int(week_start.timestamp())
            
            return {"timestamp": {"$gte": min_timestamp}}
        
        # for "recent" queries, filter to last month
        if "recent" in query_lower or "latest" in query_lower:
            month_ago = self.current_date - timedelta(days=30)
            min_timestamp = int(month_ago.timestamp())
            return {"timestamp": {"$gte": min_timestamp}}
        
        return None
    
    def _apply_date_filter_post_retrieval(self, results: Dict, date_filter: Dict) -> Dict:
        """
        Apply date filter to results after retrieval (post-filtering).
        This is needed because ChromaDB may not support numeric comparisons on metadata.
        
        Args:
            results: Search results from vector DB
            date_filter: Date filter dict (e.g., {"timestamp": {"$gte": min_timestamp}})
            
        Returns:
            Filtered results dict
        """
        if not results.get("ids") or not results["ids"][0]:
            return results
        
        # extract min_timestamp from date_filter
        min_timestamp = None
        if "timestamp" in date_filter and "$gte" in date_filter["timestamp"]:
            min_timestamp = date_filter["timestamp"]["$gte"]
        
        if min_timestamp is None:
            return results
        
        # extract results (chromadb returns nested lists)
        ids = results["ids"][0] if isinstance(results["ids"][0], list) else results["ids"]
        documents = results["documents"][0] if isinstance(results["documents"][0], list) else results["documents"]
        metadatas = results["metadatas"][0] if isinstance(results["metadatas"][0], list) else results["metadatas"]
        distances = results["distances"][0] if isinstance(results["distances"][0], list) else results["distances"]
        
        # filter by timestamp
        filtered_indices = []
        for i, metadata in enumerate(metadatas):
            # check timestamp in metadata
            timestamp = metadata.get("timestamp")
            if timestamp:
                try:
                    ts = int(timestamp) if isinstance(timestamp, str) else timestamp
                    if ts >= min_timestamp:
                        filtered_indices.append(i)
                except:
                    # if timestamp parsing fails, include it (better to show than hide)
                    filtered_indices.append(i)
            else:
                # if no timestamp, parse from date string
                date_str = metadata.get("date", "")
                date_obj = self._parse_date_from_metadata(date_str)
                if date_obj:
                    ts = int(date_obj.timestamp())
                    if ts >= min_timestamp:
                        filtered_indices.append(i)
                else:
                    # if date cannot be parsed, include it
                    filtered_indices.append(i)
        
        if not filtered_indices:
            # no results after filtering - return empty results
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        # filter results
        filtered_ids = [ids[i] for i in filtered_indices]
        filtered_documents = [documents[i] for i in filtered_indices]
        filtered_metadatas = [metadatas[i] for i in filtered_indices]
        filtered_distances = [distances[i] for i in filtered_indices]
        
        # return in chromadb format (nested lists)
        return {
            "ids": [filtered_ids],
            "documents": [filtered_documents],
            "metadatas": [filtered_metadatas],
            "distances": [filtered_distances]
        }
    
    def _post_filter_event_results(self, query: str, results: Dict) -> Dict:
        """
        Post-filter results for event queries to remove clearly irrelevant chunks.
        
        Args:
            query: Original user query
            results: Search results from vector DB
            
        Returns:
            Filtered results dict
        """
        if not results.get("ids") or not results["ids"][0]:
            return results
        
        # extract location keywords from query
        query_lower = query.lower()
        location_keywords = []
        
        # extract location after "at" or "in"
        query_words = query_lower.split()
        for i, word in enumerate(query_words):
            if word in ["at", "in"] and i + 1 < len(query_words):
                # take next 2-4 words as potential location
                location_parts = query_words[i+1:min(i+5, len(query_words))]
                # filter out stop words
                stop_words = {"the", "a", "an", "and", "or", "for", "to", "of", "on", "with", "past", "days"}
                location_parts = [w for w in location_parts if w not in stop_words]
                if location_parts:
                    location_keywords.extend(location_parts)
                    break
        
        # if no explicit location found, look for common location patterns
        if not location_keywords:
            # look for capitalized words that might be locations
            import re
            capitalized = re.findall(r'\b[A-Z][a-z]+\b', query)
            if capitalized:
                location_keywords = [w.lower() for w in capitalized[-3:]]  # take last 3 capitalized words
        
        # extract results
        ids = results["ids"][0] if isinstance(results["ids"][0], list) else results["ids"]
        documents = results["documents"][0] if isinstance(results["documents"][0], list) else results["documents"]
        metadatas = results["metadatas"][0] if isinstance(results["metadatas"][0], list) else results["metadatas"]
        distances = results["distances"][0] if isinstance(results["distances"][0], list) else results["distances"]
        
        # filter chunks that dont contain location keywords
        filtered_indices = []
        for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
            doc_lower = doc.lower()
            metadata_text = " ".join([str(v) for v in metadata.values()]).lower()
            combined_text = f"{doc_lower} {metadata_text}"
            
            # check if chunk contains location keywords or is clearly event-related
            has_location = False
            if location_keywords:
                # check if at least one location keyword appears
                for keyword in location_keywords:
                    if keyword in combined_text:
                        has_location = True
                        break
            else:
                # if no location keywords, check for event-related terms
                event_terms = ["event", "workshop", "session", "meeting", "calendar", "schedule"]
                has_location = any(term in combined_text for term in event_terms)
            
            # also check subject for event-related terms
            subject = metadata.get("subject", "").lower()
            if "event" in subject or "workshop" in subject or "meeting" in subject:
                has_location = True
            
            # exclude clearly irrelevant chunks
            exclude_terms = ["booking", "reserved", "study room", "newsletter", "digest"]
            is_excluded = any(term in combined_text for term in exclude_terms) and not has_location
            
            if has_location and not is_excluded:
                filtered_indices.append(i)
        
        if not filtered_indices:
            # if filtering removed everything, return original results (better than nothing)
            print("   ⚠️  Post-filtering removed all results, keeping original")
            return results
        
        # filter results
        filtered_ids = [ids[i] for i in filtered_indices]
        filtered_documents = [documents[i] for i in filtered_indices]
        filtered_metadatas = [metadatas[i] for i in filtered_indices]
        filtered_distances = [distances[i] for i in filtered_indices]
        
        print(f"    Post-filtered: {len(ids)} → {len(filtered_ids)} chunks (removed {len(ids) - len(filtered_indices)} irrelevant)")
        
        # return in chromadb format
        return {
            "ids": [filtered_ids],
            "documents": [filtered_documents],
            "metadatas": [filtered_metadatas],
            "distances": [filtered_distances]
        }
    
    def _format_context(self, results: Dict) -> str:
        """
        Format retrieved chunks into context for LLM.
        
        Args:
            results: Search results from vector DB
            
        Returns:
            Formatted context string
        """
        if not results.get("ids") or not results["ids"][0]:
            return ""
        
        contexts = []
        
        # extract results 
        ids = results["ids"][0] if isinstance(results["ids"][0], list) else results["ids"]
        documents = results["documents"][0] if isinstance(results["documents"][0], list) else results["documents"]
        metadatas = results["metadatas"][0] if isinstance(results["metadatas"][0], list) else results["metadatas"]
        distances = results["distances"][0] if isinstance(results["distances"][0], list) else results["distances"]
        
        for i, (chunk_id, doc, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances), 1):
            source_type = metadata.get("source_type", "unknown")
            from_addr = metadata.get("from", "Unknown")
            subject = metadata.get("subject", "No subject")
            date = metadata.get("date", "Unknown date")
            
            # parse date and show relative time if possible
            date_obj = self._parse_date_from_metadata(date)
            date_display = date
            if date_obj:
                days_ago = (self.current_date - date_obj.replace(tzinfo=None)).days
                if days_ago == 0:
                    date_display = f"{date} (Today)"
                elif days_ago == 1:
                    date_display = f"{date} (Yesterday)"
                elif days_ago > 0 and days_ago < 7:
                    date_display = f"{date} ({days_ago} days ago)"
                elif days_ago < 0:
                    date_display = f"{date} (In {abs(days_ago)} days)"
            
            context_parts = [
                f"[Chunk {i}]",
                f"Source: {source_type}",
                f"From: {from_addr}",
                f"Subject: {subject}",
                f"Date: {date_display}",
                f"Relevance: {1 - distance:.3f}",  # convert distance to similarity
                f"Content: {doc}"
            ]
            
            contexts.append("\n".join(context_parts))
        
        return "\n\n".join(contexts)
    
    def query(self, 
              user_query: str,
              top_k: Optional[int] = None,
              filter_metadata: Optional[Dict] = None,
              session_id: Optional[str] = None) -> Dict:
        """
        Perform a RAG query using modular architecture.
        
        Args:
            user_query: User's question/query
            top_k: Number of chunks to retrieve (overrides config)
            filter_metadata: Optional metadata filter (e.g., {"from": {"$contains": "nyu.edu"}})
            session_id: Optional session ID for memory/context
            
        Returns:
            Dict with answer, sources, and retrieved chunks
        """
        if top_k is None:
            top_k = self.top_k
        
        print(f"\n{'='*60}")
        print(f"RAG QUERY: {user_query}")
        print(f"{'='*60}")
        
        # step 0a: get user preferences from Memory (if enabled)
        if self.memory:
            default_filters = self.memory.get_default_filters()
            if default_filters:
                # merge with user-provided filters
                if filter_metadata:
                    if "$and" in filter_metadata:
                        filter_metadata["$and"].append(default_filters)
                    else:
                        filter_metadata = {"$and": [filter_metadata, default_filters]}
                else:
                    filter_metadata = default_filters
                print(f"💾 Applied user preferences from Memory")
        
        # step 0b: route query using Router module (if enabled)
        routing_decision = None
        if self.router:
            print("🔄 Routing query...")
            routing_decision = self.router.route_query(user_query)
            print(f"    Module: {routing_decision['module']}")
            print(f"    Intent: {routing_decision['intent']} (confidence: {routing_decision['confidence']:.2f})")
            print(f"    Reason: {routing_decision['reason']}")
            
            # route to Predict module if router says so
            is_event_query_fallback = False
            if routing_decision["module"] == "predict" and self.predict:
                print("📅 Router: Using Predict module for calendar query...")
                result = self._query_with_predict(user_query, top_k)
                # if Predict module found no events, fall back to regular RAG with enhanced retrieval
                if result is None:
                    print("    Falling back to RAG retrieval (no events in Predict DB)...")
                    print("    Using enhanced retrieval for event queries...")
                    # increase top_k for event queries (will also be filtered later)
                    top_k = top_k * 2  # get more chunks for event queries (reduced from 3x)
                    is_event_query_fallback = True
                    # continue with regular RAG flow below
                else:
                    # log to memory
                    if self.memory:
                        self.memory.log_query(user_query, routing_decision["intent"], "predict", result.get("num_chunks_retrieved", 0))
                    return result
            
            # apply routing filters
            if "metadata_filters" in routing_decision:
                if filter_metadata:
                    if "$and" in filter_metadata:
                        filter_metadata["$and"].append(routing_decision["metadata_filters"])
                    else:
                        filter_metadata = {"$and": [filter_metadata, routing_decision["metadata_filters"]]}
                else:
                    filter_metadata = routing_decision["metadata_filters"]
        
        # step 0c: fallback to legacy routing (if router not enabled)
        if not self.router:
            # check if this is a calendar query that should use Predict
            if self.predict:
                calendar_keywords = ["event", "meeting", "calendar", "schedule", "appointment", 
                                   "this week", "upcoming", "today", "tomorrow"]
                query_lower = user_query.lower()
                is_calendar_query = any(keyword in query_lower for keyword in calendar_keywords)
                
                if is_calendar_query:
                    print("📅 Detected calendar query - using Predict module...")
                    result = self._query_with_predict(user_query, top_k)
                    if self.memory:
                        self.memory.log_query(user_query, "calendar", "predict", result.get("num_chunks_retrieved", 0))
                    return result
        
        # step 0d: classify intent and get routing strategy (legacy, if router not used)
        intent_info = None
        intent_strategy = None
        if self.enable_intent_routing and not self.router:
            print("🎯 Classifying query intent...")
            intent_info = self._classify_intent(user_query)
            intent_strategy = self._get_intent_routing_strategy(intent_info)
            print(f"   Intent: {intent_info['intent']} (confidence: {intent_info['confidence']:.2f})")
            print(f"   Strategy: {intent_strategy['description']}")
            
            # for calendar intent, try Predict first
            if intent_info['intent'] == 'calendar' and self.predict:
                print("📅 Calendar intent detected - using Predict module...")
                result = self._query_with_predict(user_query, top_k)
                if self.memory:
                    self.memory.log_query(user_query, intent_info['intent'], "predict", result.get("num_chunks_retrieved", 0))
                return result
            
            #adjust top_k based on intent if needed
            if intent_strategy.get("top_k_multiplier") != 1.0:
                adjusted_top_k = int(top_k * intent_strategy["top_k_multiplier"])
                print(f"   Adjusted top_k: {top_k} → {adjusted_top_k}")
                top_k = adjusted_top_k
        
        # step 1: enhance query for event queries that fell back from Predict
        if is_event_query_fallback:
            enhanced_query = user_query
            if "event" not in user_query.lower():
                enhanced_query = f"{user_query} events calendar"
            if "recent" in user_query.lower() or "upcoming" in user_query.lower():
                enhanced_query = f"{enhanced_query} this week upcoming"
            user_query = enhanced_query
            print(f"   → Enhanced event query: {user_query}")
        
        # step 2: decompose query into sub-queries if enabled
        sub_queries = [user_query]  # default to single query
        if self.enable_subquery_decomposition:
            print("🔀 Decomposing query into sub-queries...")
            sub_queries = self._decompose_query(user_query)
            if len(sub_queries) > 1:
                print(f"   Decomposed into {len(sub_queries)} sub-queries:")
                for i, sq in enumerate(sub_queries, 1):
                    print(f"   {i}. {sq}")
            else:
                print("   Query is simple, no decomposition needed")
        
        # step 3: process each sub-query and collect results
        all_search_results = []
        
        for sub_idx, sub_query in enumerate(sub_queries):
            if len(sub_queries) > 1:
                print(f"\n--- Processing sub-query {sub_idx + 1}/{len(sub_queries)}: {sub_query} ---")
            
            # step 3a: optimize query for retrieval based on configured method or intent strategy
            retrieval_query = sub_query
            
            # determine optimization method: use intent strategy if available, otherwise use config
            optimization_method = self.query_optimization
            if intent_strategy and intent_strategy.get("query_optimization"):
                optimization_method = intent_strategy["query_optimization"]
                if len(sub_queries) == 1:
                    print(f"🎯 Using intent-specific optimization: {optimization_method}")
            
            if optimization_method == "rewrite":
                print("✏️  Rewriting query for better search...")
                retrieval_query = self._rewrite_query(sub_query)
                if retrieval_query != sub_query:
                    print(f"   Original: {sub_query}")
                    print(f"   Rewritten: {retrieval_query}")
                else:
                    print("   Query unchanged")
            elif optimization_method == "hyde":
                print("🎭 Generating hypothetical answer (HyDE)...")
                hypothetical_answer = self._generate_hypothetical_answer(sub_query)
                print(f"   Hypothetical answer: {hypothetical_answer[:200]}..." if len(hypothetical_answer) > 200 else f"   Hypothetical answer: {hypothetical_answer}")
                retrieval_query = hypothetical_answer
            else:  # "none" or invalid
                if len(sub_queries) == 1:  # only print if single query
                    print("🔍 Using original query without optimization...")
            
            # step 3b: apply query expansion if enabled (more conservative for event queries)
            if self.enable_query_expansion:
                # for event queries, use more conservative expansion
                if is_event_query_fallback:
                    print("📝 Expanding query (conservative mode for event queries)...")
                    expanded_query = self._expand_query_conservative(retrieval_query)
                else:
                    print("📝 Expanding query with synonyms and related terms...")
                    expanded_query = self._expand_query(retrieval_query)
                if expanded_query != retrieval_query:
                    print(f"   Before: {retrieval_query}")
                    print(f"   After: {expanded_query}")
                    retrieval_query = expanded_query
                else:
                    print("   No expansion needed")
            
            # step 3c:  generate query embedding
            print("🔍 Generating query embedding...")
            query_embedding = self._embed_query(retrieval_query)
            
            # step 3d: apply intent-specific and date filters
            date_filter = self._get_date_filter_for_query(sub_query)
            
            # start with user-provided filter
            filter_parts = []
            if filter_metadata:
                filter_parts.append(filter_metadata)
            
            # apply intent-specific metadata filters
            if intent_strategy and intent_strategy.get("metadata_filters"):
                intent_filters = intent_strategy["metadata_filters"]
                if intent_filters:
                    filter_parts.append(intent_filters)
                    if len(sub_queries) == 1:
                        print(f"🎯 Applying intent-specific filters: {intent_filters}")
            
            # handle sender intent - extract sender from query
            if intent_info and intent_info["intent"] == "sender":
                sender = self._extract_sender_from_query(sub_query)
                if sender:
                    sender_filter = {"from": {"$contains": sender}}
                    filter_parts.append(sender_filter)
                    if len(sub_queries) == 1:
                        print(f"📧 Extracted sender filter: {sender}")
            
            # apply date filter (but don't include in chromadb filter - do post-filtering instead)
            # chromadb may not support numeric comparisons on metadata, so filter will be done after retrieval
            use_date_filter = date_filter is not None
            if use_date_filter and len(sub_queries) == 1:
                print(f"📅 Will apply date filter after retrieval (ChromaDB compatibility)...")
            
            # combine filters (excluding date filter for chromdb)
            if len(filter_parts) == 0:
                combined_filter = None
            elif len(filter_parts) == 1:
                combined_filter = filter_parts[0]
            else:
                combined_filter = {"$and": filter_parts}
            
            # step 3e: search vector DB (hybrid or dense)
            # adjust top_k per sub-query: if multiple sub-queries, get more results per query
            per_query_k = top_k * 2 if len(sub_queries) > 1 else top_k
            # if reranking enabled, retrieve more candidates
            # if date filtering needed, retrieve more to account for post-filtering
            search_k = self.rerank_top_k if self.enable_reranking else (per_query_k * 3 if use_date_filter else (per_query_k * 2 if len(sub_queries) > 1 else per_query_k))
            
            try:
                if self.enable_hybrid_retrieval:
                    print(f"🔀 Hybrid search (BM25 + Dense, alpha={self.hybrid_alpha})...")
                    sub_results = self._hybrid_search(
                        query=retrieval_query,
            query_embedding=query_embedding,
                        top_k=search_k,
                        filter_metadata=combined_filter
                    )
                else:
                    print(f"📚 Searching vector DB (top {search_k})...")
                    sub_results = self.vector_db.search(
                        query_embedding=query_embedding,
                        n_results=search_k,
            where=combined_filter
        )
            except Exception as e:
                # if filter causes error, try without filter
                if "Invalid where clause" in str(e) or "InvalidArgumentError" in str(type(e).__name__):
                    print(f"⚠️  Filter error, retrying without metadata filter: {str(e)}")
                    if self.enable_hybrid_retrieval:
                        sub_results = self._hybrid_search(
                            query=retrieval_query,
                            query_embedding=query_embedding,
                            top_k=search_k,
                            filter_metadata=None
                        )
                    else:
                        sub_results = self.vector_db.search(
                            query_embedding=query_embedding,
                            n_results=search_k,
                            where=None
                        )
                else:
                    raise
            
            # apply date filter post-retrieval if needed
            if use_date_filter and sub_results.get("ids") and sub_results["ids"][0]:
                sub_results = self._apply_date_filter_post_retrieval(sub_results, date_filter)
            
            # step 3f: apply reranking if enabled (before iterative retrieval)
            if self.enable_reranking and sub_results.get("ids") and sub_results["ids"][0]:
                sub_results = self._rerank_results(retrieval_query, sub_results, per_query_k)
            
            # step 3g: apply iterative retrieval if enabled (only for single queries)
            if self.enable_iterative_retrieval and len(sub_queries) == 1 and sub_idx == 0:
                print(f"🔄 Iterative retrieval (max {self.max_iterations} iterations)...")
                sub_results = self._iterative_retrieval(
                    query=retrieval_query,
                    initial_results=sub_results,
                    top_k=per_query_k,
                    max_iterations=self.max_iterations
                )
            
            all_search_results.append(sub_results)
        
        # step 4a: merge results if multiple sub-queries
        if len(sub_queries) > 1:
            print(f"\n🔗 Merging results from {len(sub_queries)} sub-queries...")
            results = self._merge_search_results(all_search_results, top_k)
            print(f"   Merged to {len(results.get('ids', [[]])[0]) if results.get('ids') and results['ids'][0] else 0} unique chunks")
        else:
            results = all_search_results[0] if all_search_results else {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        # step 4b: Apply small2big expansion if enabled
        if self.enable_small2big and results.get("ids") and results["ids"][0]:
            print(f"\n🔍 Small2big: Expanding to full context (k={self.small2big_expansion_k})...")
            original_count = len(results["ids"][0]) if isinstance(results["ids"][0], list) else len(results["ids"])
            results = self._expand_to_full_context(results, expansion_k=self.small2big_expansion_k)
            expanded_count = len(results["ids"][0]) if isinstance(results["ids"][0], list) else len(results["ids"])
            print(f"   Expanded from {original_count} to {expanded_count} chunks")
        
        # step 4c: apply reranking if enabled (for merged results from multiple sub-queries)
        if self.enable_reranking and len(sub_queries) > 1 and results.get("ids") and results["ids"][0]:
            # rerank merged results from multiple sub-queries
            print(f"\n🔄 Reranking merged results from {len(sub_queries)} sub-queries...")
            results = self._rerank_results(user_query, results, top_k)
        
        # step 4d: post-filter irrelevant chunks for event queries
        if is_event_query_fallback and results.get("ids") and results["ids"][0]:
            print("\n🔍 Post-filtering event results to remove irrelevant chunks...")
            results = self._post_filter_event_results(user_query, results)
        
        # step 5: post-filter by date if needed (fallback if metadata filtering didn't work)
        # check date filter from original query for post-filtering
        date_filter = self._get_date_filter_for_query(user_query)
        if date_filter and results.get("ids") and results["ids"][0]:
            # extract and filter results by date
            ids = results["ids"][0] if isinstance(results["ids"][0], list) else results["ids"]
            documents = results["documents"][0] if isinstance(results["documents"][0], list) else results["documents"]
            metadatas = results["metadatas"][0] if isinstance(results["metadatas"][0], list) else results["metadatas"]
            distances = results["distances"][0] if isinstance(results["distances"][0], list) else results["distances"]
            
            # filter by parsing dates from metadata
            filtered_indices = []
            min_timestamp = None
            if "this week" in user_query.lower() or "upcoming" in user_query.lower():
                # get start of current week (Sunday) - same logic as _get_date_filter_for_query
                days_since_sunday = (self.current_date.weekday() + 1) % 7
                week_start = self.current_date - timedelta(days=days_since_sunday)
                week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
                min_timestamp = int(week_start.timestamp())
            
            for i, metadata in enumerate(metadatas):
                # check timestamp in metadata
                timestamp = metadata.get("timestamp")
                if timestamp:
                    try:
                        ts = int(timestamp) if isinstance(timestamp, str) else timestamp
                        if min_timestamp is None or ts >= min_timestamp:
                            filtered_indices.append(i)
                    except:
                        # if timestamp parsing fails, include it (better to show than hide)
                        filtered_indices.append(i)
                else:
                    # if no timestamp, parse from date string
                    date_str = metadata.get("date", "")
                    date_obj = self._parse_date_from_metadata(date_str)
                    if date_obj:
                        ts = int(date_obj.timestamp())
                        if min_timestamp is None or ts >= min_timestamp:
                            filtered_indices.append(i)
                    else:
                        # if date cannot be parsed, include it
                        filtered_indices.append(i)
            
            # limit to top_k
            filtered_indices = filtered_indices[:top_k]
            
            # rebuild results with filtered data
            if filtered_indices:
                results = {
                    "ids": [[ids[i] for i in filtered_indices]],
                    "documents": [[documents[i] for i in filtered_indices]],
                    "metadatas": [[metadatas[i] for i in filtered_indices]],
                    "distances": [[distances[i] for i in filtered_indices]]
                }
            else:
                # no results after filtering - try without date filter
                print("⚠️  No results with date filter, trying without...")
                results = self.vector_db.search(
                    query_embedding=query_embedding,
                    n_results=top_k,
                    where=filter_metadata
                )
        
        if not results.get("ids") or not results["ids"][0]:
            return {
                "query": user_query,
                "answer": "No relevant emails found in the corpus.",
                "sources": [],
                "retrieved_chunks": []
            }
        
        # step 6: format context
        context = self._format_context(results)
        
        # step 7: generate answer with LLM
        print("🤖 Generating answer...")
        
        # build prompt with current date context
        current_date_str = self.current_date.strftime("%A, %B %d, %Y")
        current_time_str = self.current_date.strftime("%I:%M %p")
        
        system_prompt = f"""You are a helpful assistant that answers questions based on email content.
Today's date is {current_date_str} ({current_time_str}).
You must ONLY use information from the provided email chunks. If the information is not in the chunks, say so.
When answering questions about "this week" or "upcoming" events, consider that today is {current_date_str}.
Be concise and accurate. Cite which chunks you used when relevant."""
        
        user_prompt = f"""Based on the following email chunks, answer this question: {user_query}

Note: Today is {current_date_str}. When interpreting temporal references like "this week" or "upcoming", use this date as reference.

Email Chunks:
{context}

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.response_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.context_window,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
        
        # step 8: format sources
        sources = []
        if results.get("metadatas") and results["metadatas"][0]:
            metadatas = results["metadatas"][0] if isinstance(results["metadatas"][0], list) else results["metadatas"]
            for metadata in metadatas:
                sources.append({
                    "from": metadata.get("from", "Unknown"),
                    "subject": metadata.get("subject", "No subject"),
                    "date": metadata.get("date", "Unknown"),
                    "source_type": metadata.get("source_type", "unknown"),
                    "message_id": metadata.get("message_id", "unknown")
                })
        
        # extract retrieved chunks for reference
        retrieved_chunks = []
        if results.get("documents") and results["documents"][0]:
            documents = results["documents"][0] if isinstance(results["documents"][0], list) else results["documents"]
            ids = results["ids"][0] if isinstance(results["ids"][0], list) else results["ids"]
            for chunk_id, doc in zip(ids, documents):
                retrieved_chunks.append({
                    "chunk_id": chunk_id,
                    "text": doc[:200] + "..." if len(doc) > 200 else doc  # preview
                })
        
        result = {
            "query": user_query,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks,
            "num_chunks_retrieved": len(sources)
        }
        
        # add routing information
        if routing_decision:
            result["routing"] = routing_decision
            result["intent"] = routing_decision["intent"]
            result["intent_confidence"] = routing_decision["confidence"]
        elif intent_info:
            result["intent"] = intent_info["intent"]
            result["intent_confidence"] = intent_info["confidence"]
        
        # log query to memory
        if self.memory:
            intent = routing_decision["intent"] if routing_decision else (intent_info["intent"] if intent_info else "general")
            self.memory.log_query(user_query, intent, "retriever", len(sources))
        
        return result
    
    def close(self):
        """Close connections"""
        self.embedder.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Query emails using RAG")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--query", required=True, help="Query to search for")
    parser.add_argument("--top-k", type=int, help="Number of chunks to retrieve (overrides config)")
    
    args = parser.parse_args()
    
    rag = RAGQuery(config_path=args.config)
    
    try:
        result = rag.query(args.query, top_k=args.top_k)
        
        print("\n" + "="*60)
        print("ANSWER")
        print("="*60)
        print(result["answer"])
        
        print("\n" + "="*60)
        print("SOURCES")
        print("="*60)
        for i, source in enumerate(result["sources"], 1):
            print(f"\n{i}. From: {source['from']}")
            print(f"   Subject: {source['subject']}")
            print(f"   Date: {source['date']}")
            print(f"   Type: {source['source_type']}")
        
    finally:
        rag.close()

