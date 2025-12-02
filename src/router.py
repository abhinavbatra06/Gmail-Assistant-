"""
Router Module - Modular RAG Routing

Routes queries to appropriate modules (Predict, Retriever, etc.) based on query type and intent.
"""

import json
from typing import Dict, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


class Router:
    
    def __init__(self, config_path="config.yaml"):
        """
        Initialize router.
        
        Args:
            config_path: Path to config file
        """
        import yaml
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        
        rag_cfg = self.cfg.get("rag", {})
        embedding_cfg = self.cfg.get("embedding", {})
        
        # openai client for intent classification
        api_key_env = embedding_cfg.get("openai_api_key_env", "OPENAI_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"OpenAI API key not found in environment variable: {api_key_env}")
        
        self.client = OpenAI(api_key=api_key)
        self.enable_routing = rag_cfg.get("enable_intent_routing", False)
    
    def route_query(self, query: str) -> Dict[str, any]:
        """
        Route query to appropriate module.
        
        Args:
            query: User query
            
        Returns:
            Routing decision with module type and parameters
        """
        if not self.enable_routing:
            return {
                "module": "retriever",
                "intent": "general",
                "confidence": 1.0,
                "reason": "Routing disabled, using default retriever"
            }
        
        # classify intent
        intent_info = self._classify_intent(query)
        
        # determine routing
        routing_decision = {
            "module": "retriever",  # default
            "intent": intent_info["intent"],
            "confidence": intent_info["confidence"],
            "reason": f"Intent: {intent_info['intent']}"
        }
        
        # route to Predict for calendar queries
        if intent_info["intent"] == "calendar":
            routing_decision["module"] = "predict"
            routing_decision["reason"] = "Calendar query - routing to Predict module"
        
        # route to specialized handlers for other intents
        elif intent_info["intent"] == "sender":
            routing_decision["module"] = "retriever"
            routing_decision["metadata_filters"] = self._extract_sender_filter(query)
            routing_decision["reason"] = "Sender query - applying sender filter"
        
        return routing_decision
    
    def _classify_intent(self, query: str) -> Dict[str, any]:
        """
        Classify query intent.
        
        Args:
            query: User query string
            
        Returns:
            Dict with intent type, confidence, and original query
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
    
    def _extract_sender_filter(self, query: str) -> Optional[Dict]:
        """
        Extract sender information from query for filtering.
        
        Args:
            query: User query string
            
        Returns:
            Sender filter dict or None
        """
        import re
        
        # look for email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, query)
        if emails:
            return {"from": {"$contains": emails[0]}}
        
        # look for domain mentions
        domain_keywords = ["from", "by", "sent by"]
        query_lower = query.lower()
        for keyword in domain_keywords:
            if keyword in query_lower:
                parts = query_lower.split(keyword, 1)
                if len(parts) > 1:
                    after = parts[1].strip().split()[0] if parts[1].strip() else None
                    if after and ("." in after or after.endswith("edu") or after.endswith("com")):
                        return {"from": {"$contains": after}}
        
        return None

