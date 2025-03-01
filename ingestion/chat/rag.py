"""
RAG (Retrieval-Augmented Generation) module for Cipher chatbot.

This module implements a RAG system using FAISS vector search
and the Anthropic Claude model for generating contextual responses.
"""

import os
import sys
import json
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add project root to the path if needed
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from anthropic import Anthropic
from ingestion.vector.search import VectorSearch


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for the Cipher chatbot.

    This class combines vector search with LLM generation to create
    a system that retrieves relevant context and uses it to generate
    accurate responses to user queries.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "claude-3-5-haiku-20241022",
        vector_index_dir: str = "./ingestion/vector/data",
        search_results_count: int = 5,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        history_dir: str = "./ingestion/chat/history",
    ):
        """
        Initialize the RAG engine.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model_name: Claude model to use
            vector_index_dir: Directory containing the FAISS index
            search_results_count: Number of search results to retrieve
            max_tokens: Maximum tokens in the response
            temperature: Temperature for response generation
            history_dir: Directory to store chat history
        """
        # Set up the Anthropic client
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model_name = model_name

        # Demo mode flag - true if no API key available
        self.demo_mode = False

        if not self.api_key:
            print("Warning: No Anthropic API key found. Running in demo mode.")
            self.demo_mode = True
        else:
            try:
                self.client = Anthropic(api_key=self.api_key)
            except Exception as e:
                print(f"Error initializing Anthropic client: {str(e)}")
                print("Running in demo mode.")
                self.demo_mode = True

        self.max_tokens = max_tokens
        self.temperature = temperature

        # Set up vector search
        self.search = VectorSearch(index_dir=vector_index_dir)
        self.search_results_count = search_results_count

        # Chat history setup
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history_path = self.history_dir / f"session_{self.session_id}.json"
        
        # Initialize conversation history
        self.history: List[Dict[str, str]] = []

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query using enhanced retrieval techniques.

        Args:
            query: User query to find relevant context for

        Returns:
            List of relevant context entries from the vector store
        """
        # Perform basic semantic search
        basic_results = self.search.search(query, k=self.search_results_count)
        
        # If we got fewer results than requested or confidence is low, try to broaden search
        if len(basic_results) < self.search_results_count:
            # Extract key entities from the query for expanded search
            # This is a simplified version - in a full implementation, you'd use NER
            query_words = query.lower().split()
            important_words = [word for word in query_words 
                              if len(word) > 4 and word not in 
                              ['about', 'these', 'those', 'their', 'where', 'which', 'would']]
            
            if important_words:
                # Try searching with just the important entities/keywords
                entity_query = " ".join(important_words)
                expanded_results = self.search.search(entity_query, 
                                                    k=self.search_results_count)
                
                # Add any new results that weren't in the original results
                existing_ids = {r['id'] for r in basic_results if 'id' in r}
                for result in expanded_results:
                    if 'id' in result and result['id'] not in existing_ids:
                        basic_results.append(result)
                        existing_ids.add(result['id'])
                        
                        # Stop if we've reached our desired count
                        if len(basic_results) >= self.search_results_count:
                            break
        
        # Sort results by relevance (lowest distance = most relevant)
        sorted_results = sorted(basic_results, key=lambda x: x.get('distance', float('inf')))
        
        # Limit to the top results
        return sorted_results[:self.search_results_count]

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results into a structured intelligence context for the LLM.

        Args:
            results: List of search results from the vector store

        Returns:
            Formatted context string with intelligence-style presentation
        """
        if not results:
            return "No relevant information found."

        context_parts = []

        for i, result in enumerate(results):
            # Calculate confidence score (transform distance to confidence)
            # Lower distance = higher confidence (1.0 - normalized_distance)
            confidence = 1.0 - (result.get('distance', 0.5) / 2)
            confidence_level = "HIGH" if confidence > 0.8 else "MEDIUM" if confidence > 0.6 else "LOW"
            
            # Format timestamp if available, otherwise use generic date
            report_date = result.get('date', 'N/A')
            
            # Format each result into a structured intelligence entry with appropriate metadata
            context_parts.append(
                f"[SOURCE {i+1}] {'-'*50}\n"
                f"CLASSIFICATION: UNCLASSIFIED // FOR OFFICIAL USE ONLY\n"
                f"SUBJECT: {result['title']}\n"
                f"REFERENCE ID: {result['report_id']}\n"
                f"TOPIC: {result['topic']}\n"
                f"DATE: {report_date}\n" 
                f"CONFIDENCE: {confidence_level} ({confidence:.2f})\n"
                f"CONTENT:\n{result['content']}\n"
                f"{'='*60}\n"
            )

        # Add header to the context
        header = f"""
INTELLIGENCE BRIEFING MATERIALS
CLASSIFICATION: UNCLASSIFIED // FOR OFFICIAL USE ONLY
RETRIEVED: {len(results)} RELEVANT SOURCES
{'='*60}
"""
        
        # Add footer with handling instructions
        footer = f"""
{'='*60}
HANDLING INSTRUCTIONS: Information is UNCLASSIFIED but sensitive. 
Provide appropriate context and source citation in analysis.
"""

        return header + "\n".join(context_parts) + footer

    def generate_response(
        self, query: str, context: str, with_citations: bool = True
    ) -> str:
        """
        Generate a response using the Claude API with retrieved context.

        Args:
            query: User query
            context: Retrieved context from the vector store
            with_citations: Whether to include citations in the response

        Returns:
            Generated response from Claude
        """
        # Check if we're in demo mode
        if self.demo_mode:
            return f"This is a demo response for: {query}"

        # Build the system prompt
        system_prompt = f"""You are Cipher, an elite AI analyst for The Cipher Brief, specializing in geopolitical intelligence, threat analysis, and national security insights.
Your responses should be precise, authoritative, and insightful - emulating the analytical rigor of intelligence professionals.

CORE CAPABILITIES:
- Geopolitical Analysis: Explain complex international relations, conflicts, and diplomatic developments
- National Security: Analyze threats to national security, defense strategies, and security implications
- Intelligence Tradecraft: Provide expert insights on intelligence operations, trends, and methodologies
- Military Analysis: Assess military capabilities, strategies, and emerging defense technologies
- Cyber Intelligence: Analyze cyber threats, operations, and their implications for national security

When answering questions about geopolitics, security, intelligence matters, or current events from your database,
use only the provided context information. If the context doesn't contain relevant information,
acknowledge this limitation clearly and provide a framework for understanding the issue
based on established analytical methodologies without speculating beyond your knowledge.

However, if the user is asking a general question unrelated to intelligence matters
(such as coding help, math problems, creative writing, or casual conversation),
respond normally as a helpful assistant without the intelligence analyst persona.

{f"When referencing specific intelligence from the context, include citation numbers in brackets like [1], [2], etc." if with_citations else ""}

ANALYTICAL STYLE:
- Structured Analysis: Present information in a clear, logical flow with key judgments clearly identified
- Nuanced Assessment: Highlight degrees of confidence in your analysis and present alternative viewpoints
- Contextual Understanding: Place events in their proper historical and strategic context
- Objective Analysis: Maintain analytical objectivity while presenting multiple perspectives
- Forward-Looking: Identify potential implications and future scenarios when appropriate

For intelligence-related questions, model your responses after high-quality intelligence briefings - factual, concise, and focused on insights that would be valuable to decision-makers."""

        # Check if context is empty or not useful
        context_is_empty = context == "No relevant information found."
        
        # Build the user message
        if context_is_empty:
            # For empty context, just use conversation history and answer normally
            user_message = query
            messages = self.history + [{"role": "user", "content": user_message}]
        else:
            # For queries with relevant context, include the context
            user_message = f"""INTELLIGENCE QUERY: {query}

RETRIEVED INTELLIGENCE:
{context}

ANALYTICAL GUIDANCE:
1. If this query relates to geopolitics, national security, intelligence matters, or current events, provide a structured intelligence analysis based strictly on the retrieved information.
   - Begin with key judgments or a concise executive summary
   - Analyze the implications and significance of the information
   - Identify any intelligence gaps or areas of uncertainty
   - Place the information in relevant geopolitical/historical context
   - {f"Include source citations [1], [2], etc. to maintain analytical rigor" if with_citations else "Clearly delineate facts from your assessment"}

2. If this is a general question unrelated to intelligence matters, respond as a helpful assistant without the intelligence analyst persona or references to the retrieved information.

3. Maintain appropriate classification handling - treat all information as sensitive but unclassified (SBU)."""
            messages = [{"role": "user", "content": user_message}]

        try:
            # Generate response
            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            return response.content[0].text
        except Exception as e:
            print(f"Error generating response via API: {str(e)}")
            return f"Error generating response: {str(e)}"

    def save_history(self):
        """
        Save the current conversation history to a JSON file.
        """
        history_data = {
            "session_id": self.session_id,
            "model": self.model_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "messages": self.history
        }
        
        with open(self.history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
    
    def is_general_question(self, query: str) -> bool:
        """
        Determine if a query is likely a general question not related to intelligence matters.
        
        Args:
            query: User query
            
        Returns:
            Boolean indicating if this is likely a general question
        """
        # List of common prefixes/phrases for general chit-chat
        general_prefixes = [
            "hello", "hi", "hey", "how are you", "good morning", "good afternoon", 
            "nice to meet", "thanks", "thank you", "what's up", "tell me a joke",
            "can you help", "tell me about yourself", "what can you do"
        ]
        
        # Common coding/technical question indicators
        technical_indicators = [
            "code", "program", "function", "how do i", "how to", "write", 
            "python", "javascript", "java", "c++", "algorithm"
        ]
        
        # Intelligence and geopolitical keywords that indicate RAG should be used
        intelligence_keywords = [
            "russia", "china", "iran", "north korea", "ukraine", "taiwan", 
            "military", "intelligence", "cyber", "terrorist", "terrorism", "defense",
            "security", "threat", "nuclear", "weapons", "war", "conflict", "attack",
            "strategy", "diplomatic", "foreign policy", "geopolitical", "government", 
            "national security", "classified", "covert", "espionage", "spy", "hacking",
            "election", "missile", "nato", "middle east", "africa", "european union",
            "sanctions", "terrorist", "jihad", "insurgent", "treaty", "agreement"
        ]
        
        # Check for general conversation starters
        query_lower = query.lower()
        
        # Check if it contains intelligence keywords (not a general question)
        for keyword in intelligence_keywords:
            if keyword in query_lower:
                return False  # Contains intelligence keyword, not a general question
                
        # Check for general chat prefixes
        for prefix in general_prefixes:
            if query_lower.startswith(prefix) or query_lower == prefix:
                return True
                
        # Check for technical question indicators
        for indicator in technical_indicators:
            if indicator in query_lower:
                return True
                
        # Check if query is very short (likely conversational)
        if len(query.split()) <= 3:
            return True
            
        # If no intelligence keywords and no general indicators, 
        # do a more advanced check based on query structure
        
        # Questions asking for explanations of non-intelligence concepts
        if query_lower.startswith("what is") or query_lower.startswith("who is") or query_lower.startswith("how does"):
            # If it's a short "what is" question without intelligence keywords, it's likely general
            if len(query.split()) <= 5:
                return True
        
        # Default to considering it intelligence-related if we're unsure
        # This ensures we try to use the RAG system when appropriate
        return False
        
    def process_query(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process a user query through the full RAG pipeline.

        Args:
            query: User query

        Returns:
            Tuple of (generated response, retrieved context)
        """
        # Check if this might be a general question first
        if self.is_general_question(query):
            # For general questions, don't use the RAG context
            results = []
            context = "No relevant information found."
        else:
            # Step 1: Retrieve relevant context
            results = self.retrieve(query)
            
            # Step 2: Format context for the LLM
            context = self.format_context(results)
        
        # Step 3: Generate response using context
        response = self.generate_response(query, context)

        # Update conversation history
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": response})
        
        # Persist the chat history after each interaction
        self.save_history()

        return response, results
