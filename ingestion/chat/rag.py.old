"""
RAG (Retrieval-Augmented Generation) module for Cipher chatbot.

This module implements a RAG system using FAISS vector search
and the Anthropic Claude model for generating contextual responses.
"""

import os
import sys
import json
import hashlib
import datetime
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union, Callable
from functools import lru_cache

# Add project root to the path if needed
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from anthropic import Anthropic
from ingestion.vector.search import VectorSearch


class ResponseCache:
    """
    Caches responses based on semantic similarity of queries.
    
    This cache stores responses and provides lookup based on both exact matches
    and semantic similarity to reduce redundant API calls.
    """
    
    def __init__(self, max_size: int = 100, similarity_threshold: float = 0.92, ttl: int = 3600):
        """
        Initialize the response cache.
        
        Args:
            max_size: Maximum number of entries to store in the cache
            similarity_threshold: Minimum similarity score to consider a cache hit
            ttl: Time-to-live for cache entries in seconds
        """
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.query_embeddings: Dict[str, List[float]] = {}
        
    def get(self, query: str, embedding_fn: Callable[[str], List[float]]) -> Optional[Dict[str, Any]]:
        """
        Get a cached response for a query.
        
        Args:
            query: The query to lookup
            embedding_fn: Function to generate embeddings for semantic comparison
            
        Returns:
            Cached response or None if not found
        """
        # First check for exact match (faster)
        query_hash = self._hash_query(query)
        if query_hash in self.cache:
            entry = self.cache[query_hash]
            # Check if the entry has expired
            if time.time() - entry['timestamp'] < self.ttl:
                # Update access time
                entry['last_accessed'] = time.time()
                print(f"Cache hit: exact match for '{query}'")
                return entry['data']
            else:
                # Remove expired entry
                del self.cache[query_hash]
                if query_hash in self.query_embeddings:
                    del self.query_embeddings[query_hash]
        
        # If no exact match, try semantic matching
        try:
            query_embedding = embedding_fn(query)
            
            # Find most similar cached query
            max_similarity = 0.0
            most_similar_hash = None
            
            for cached_hash, cached_embedding in self.query_embeddings.items():
                # Skip if entry no longer exists or has expired
                if cached_hash not in self.cache or time.time() - self.cache[cached_hash]['timestamp'] >= self.ttl:
                    continue
                
                # Compute cosine similarity
                similarity = self._cosine_similarity(query_embedding, cached_embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_hash = cached_hash
            
            # Return cached response if similarity is above threshold
            if most_similar_hash and max_similarity >= self.similarity_threshold:
                entry = self.cache[most_similar_hash]
                # Update access time
                entry['last_accessed'] = time.time()
                print(f"Cache hit: semantic match ({max_similarity:.3f}) for '{query}'")
                return entry['data']
                
        except Exception as e:
            print(f"Error during semantic cache lookup: {str(e)}")
        
        return None
    
    def set(self, query: str, data: Dict[str, Any], embedding_fn: Callable[[str], List[float]]) -> None:
        """
        Cache a response for a query.
        
        Args:
            query: The query to cache
            data: The data to cache
            embedding_fn: Function to generate embeddings for semantic comparison
        """
        # Check if we need to evict entries
        if len(self.cache) >= self.max_size:
            self._evict_entries()
        
        # Add new entry
        query_hash = self._hash_query(query)
        now = time.time()
        self.cache[query_hash] = {
            'data': data,
            'timestamp': now,
            'last_accessed': now
        }
        
        # Store embedding for semantic matching
        try:
            self.query_embeddings[query_hash] = embedding_fn(query)
        except Exception as e:
            print(f"Error generating embedding for cache: {str(e)}")
    
    def _hash_query(self, query: str) -> str:
        """
        Create a hash of the query for exact match lookup.
        
        Args:
            query: The query to hash
            
        Returns:
            Hash string of the query
        """
        # Normalize the query by lowercasing and removing extra whitespace
        normalized = ' '.join(query.lower().split())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity as a float between -1 and 1
        """
        import numpy as np
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    def _evict_entries(self) -> None:
        """
        Evict least recently used entries when cache is full.
        """
        # Sort entries by last access time
        sorted_entries = sorted(
            self.cache.items(), 
            key=lambda x: x[1]['last_accessed']
        )
        
        # Remove oldest 10% of entries or at least one
        entries_to_remove = max(1, int(len(self.cache) * 0.1))
        for i in range(entries_to_remove):
            if i < len(sorted_entries):
                hash_to_remove = sorted_entries[i][0]
                if hash_to_remove in self.cache:
                    del self.cache[hash_to_remove]
                if hash_to_remove in self.query_embeddings:
                    del self.query_embeddings[hash_to_remove]


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
        enable_cache: bool = True,
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
            enable_cache: Whether to enable response caching
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
        
        # Response cache setup
        self.enable_cache = enable_cache
        if self.enable_cache:
            self.cache = ResponseCache(max_size=100, similarity_threshold=0.92, ttl=3600)

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the query to extract intent, entities, and generate reformulations.
        
        Args:
            query: The user's query
            
        Returns:
            Dictionary containing query analysis information
        """
        if self.demo_mode:
            # Return simplified analysis in demo mode
            return {
                "entities": query.split(),
                "intent": "information",
                "reformulations": [query]
            }
        
        # Use Claude to analyze the query
        system_prompt = """You are an expert intelligence analyst tasked with analyzing queries for an intelligence research system.
Analyze the user's query to identify:
1. Key entities (people, organizations, countries, events)
2. The search intent (e.g., factual information, analysis, comparison, timeline)
3. Generate 3 reformulated search queries that could help retrieve relevant information

Respond in JSON format:
{
  "entities": ["entity1", "entity2", ...],
  "intent": "intent_type",
  "reformulations": ["query1", "query2", "query3"]
}

Make sure that:
- The reformulated queries are diverse and approach the topic from different angles
- Each reformulation should be focused and specific
- Reformulations should capture different aspects of the original query
- Each reformulation should be a complete, well-formed question"""
        
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",  # Use smaller, faster model for analysis
                system=system_prompt,
                messages=[{"role": "user", "content": query}],
                max_tokens=500,
                temperature=0.0  # Deterministic for consistent results
            )
            
            # Extract JSON from the response
            import json
            import re
            
            content = response.content[0].text
            # Find JSON in the content (handling formatting variations)
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = content
                
            try:
                # Try to parse as-is
                analysis = json.loads(json_str)
            except json.JSONDecodeError:
                # Try to find the JSON object using regex if direct parsing fails
                json_pattern = r'\{.*\}'
                json_match = re.search(json_pattern, json_str, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group(0))
                else:
                    raise ValueError("Could not extract valid JSON from LLM response")
            
            return analysis
        except Exception as e:
            print(f"Error during query analysis: {str(e)}")
            # Return basic fallback analysis
            return {
                "entities": [word for word in query.split() if len(word) > 3],
                "intent": "information",
                "reformulations": [query]
            }

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query using enhanced retrieval techniques.

        Args:
            query: User query to find relevant context for

        Returns:
            List of relevant context entries from the vector store
        """
        # Analyze query to get entities and reformulations
        analysis = self.analyze_query(query)
        reformulations = analysis.get("reformulations", [query])
        
        # Make sure original query is included in reformulations
        if query not in reformulations:
            reformulations.append(query)
            
        # Search with all reformulations
        all_results = []
        existing_ids = set()
        
        for q in reformulations:
            # Search with the reformulated query
            results = self.search.search(q, k=self.search_results_count)
            
            # Add any new results that weren't in previous reformulations
            for result in results:
                result_id = result.get('id', hash(result.get('content', '')))
                if result_id not in existing_ids:
                    all_results.append(result)
                    existing_ids.add(result_id)
        
        # Get any missing key entities if needed
        if len(all_results) < self.search_results_count:
            entities = analysis.get("entities", [])
            for entity in entities:
                if len(entity) > 3:  # Only use substantial entities
                    entity_results = self.search.search(entity, k=self.search_results_count // 2)
                    
                    # Add any new entity results
                    for result in entity_results:
                        result_id = result.get('id', hash(result.get('content', '')))
                        if result_id not in existing_ids:
                            all_results.append(result)
                            existing_ids.add(result_id)
                            
                            # Stop if we have enough results
                            if len(all_results) >= self.search_results_count * 2:
                                break
        
        # Sort results by relevance (lowest distance = most relevant)
        sorted_results = sorted(all_results, key=lambda x: x.get('distance', float('inf')))
        
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
        Uses a Chain of Thought (CoT) approach for improved analytical quality.

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
            
        # Check if response is cached
        if self.enable_cache:
            # Use the embedding model from the search index for cache lookup
            def get_embedding(text):
                try:
                    # The model is accessed via FAISSIndex in our search object
                    return self.search.index.model.encode(text).tolist()
                except Exception as e:
                    print(f"Warning: Error generating embedding for cache: {str(e)}")
                    # Return a dummy embedding if model access fails
                    return [0.0] * 384  # Default embedding dimension
                    
            cache_key = {
                "query": query,
                "context_hash": hash(context),
                "with_citations": with_citations
            }
            cached_response = self.cache.get(query, get_embedding)
            if cached_response:
                return cached_response.get("response", f"This is a cached response for: {query}")

        # Step 1: Analyze context for relevance and extract key data points
        context_analysis = self._analyze_context(query, context)
        
        # Step 2: Generate a chain of thought response
        final_response = self._generate_analytical_response(query, context, context_analysis, with_citations)
        
        # Cache the response if caching is enabled
        if self.enable_cache:
            # Use the embedding model from the search index for cache
            def get_embedding(text):
                try:
                    # The model is accessed via FAISSIndex in our search object
                    return self.search.index.model.encode(text).tolist()
                except Exception as e:
                    print(f"Warning: Error generating embedding for cache: {str(e)}")
                    # Return a dummy embedding if model access fails
                    return [0.0] * 384  # Default embedding dimension
                    
            self.cache.set(query, {"response": final_response}, get_embedding)
            
        return final_response
    
    def _analyze_context(self, query: str, context: str) -> Dict[str, Any]:
        """
        First step in CoT: Analyze the retrieved context for relevance and extract key information.
        
        Args:
            query: The user query
            context: The retrieved context
            
        Returns:
            Dictionary with context analysis results
        """
        # Skip analysis for empty context
        if context == "No relevant information found.":
            return {
                "relevance": "none",
                "key_points": [],
                "gaps": ["No relevant information found in the intelligence database."],
                "sources_reliability": []
            }
        
        system_prompt = """You are an expert intelligence analyst specializing in evaluating intelligence information.
Your task is to analyze the retrieved intelligence sources and assess their relevance to the query.

Examine each intelligence source objectively and identify:
1. The relevance of each source to the specific query (high/medium/low/none)
2. Key facts and data points that address the query
3. Information gaps or limitations in the available intelligence
4. Reliability assessment of each source based on confidence levels

Respond in JSON format:
{
  "relevance": "overall_relevance",
  "key_points": ["point1", "point2", ...],
  "gaps": ["gap1", "gap2", ...],
  "sources_reliability": [
    {"source_num": 1, "reliability": "high/medium/low", "relevant_content": "summary of key info"}
  ]
}"""

        try:
            # Use a smaller model for analysis to save tokens
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                system=system_prompt,
                messages=[{
                    "role": "user", 
                    "content": f"QUERY: {query}\n\nINTELLIGENCE SOURCES:\n{context}"
                }],
                max_tokens=1500,
                temperature=0.0  # Deterministic for consistency
            )
            
            # Extract JSON from the response
            import json
            import re
            
            content = response.content[0].text
            # Find JSON in the content (handling formatting variations)
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = content
                
            try:
                # Try to parse as-is
                analysis = json.loads(json_str)
            except json.JSONDecodeError:
                # Try to find the JSON object using regex if direct parsing fails
                json_pattern = r'\{.*\}'
                json_match = re.search(json_pattern, json_str, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group(0))
                else:
                    # Fallback to structured extraction if JSON parsing fails
                    analysis = {
                        "relevance": "medium" if "relevant" in content.lower() else "low",
                        "key_points": re.findall(r'key points?:?(.*?)(?:gaps|$)', content, re.IGNORECASE | re.DOTALL)[0].split('\n') if re.findall(r'key points?:?(.*?)(?:gaps|$)', content, re.IGNORECASE | re.DOTALL) else [],
                        "gaps": re.findall(r'gaps?:?(.*?)(?:sources|$)', content, re.IGNORECASE | re.DOTALL)[0].split('\n') if re.findall(r'gaps?:?(.*?)(?:sources|$)', content, re.IGNORECASE | re.DOTALL) else ["Unable to determine specific gaps"],
                        "sources_reliability": []
                    }
            
            return analysis
            
        except Exception as e:
            print(f"Error during context analysis: {str(e)}")
            # Return basic analysis on error
            return {
                "relevance": "unknown",
                "key_points": [],
                "gaps": ["Error occurred during intelligence analysis."],
                "sources_reliability": []
            }
    
    def _generate_analytical_response(
        self, query: str, context: str, analysis: Dict[str, Any], with_citations: bool = True
    ) -> str:
        """
        Second step in CoT: Generate a structured analytical response based on the context analysis.
        
        Args:
            query: The user query
            context: The retrieved context
            analysis: Context analysis results
            with_citations: Whether to include citations
            
        Returns:
            Generated analytical response
        """
        # Build an enhanced system prompt incorporating analysis
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

        # Context is empty or analysis shows it's not useful
        context_is_empty = (context == "No relevant information found." or 
                           analysis.get("relevance", "").lower() in ["none", "low"] or 
                           not analysis.get("key_points", []))
        
        # Build enhanced user message with analysis insights
        if context_is_empty:
            # For empty/irrelevant context, use conversation history and acknowledge limitations
            user_message = f"""INTELLIGENCE QUERY: {query}

PRELIMINARY INTELLIGENCE ASSESSMENT:
Our intelligence database contains insufficient relevant information on this topic.

ANALYTICAL APPROACH:
1. Acknowledge the intelligence gap clearly to the user
2. If appropriate, suggest a framework for understanding the issue based on established analytical methodologies
3. Do NOT speculate beyond available information or fabricate intelligence
4. Provide a concise response that is honest about the limitations"""
            messages = self.history + [{"role": "user", "content": user_message}]
        else:
            # Format the analysis insights for the LLM
            key_points = "\n".join([f"- {point}" for point in analysis.get("key_points", [])])
            gaps = "\n".join([f"- {gap}" for gap in analysis.get("gaps", [])])
            
            # Format source reliability information
            source_assessments = ""
            for src in analysis.get("sources_reliability", []):
                source_num = src.get("source_num", "Unknown")
                reliability = src.get("reliability", "Unknown")
                content = src.get("relevant_content", "")
                if content:
                    source_assessments += f"- Source {source_num}: {reliability.upper()} reliability - {content}\n"
            
            # Build enhanced prompt with analytical guidance
            user_message = f"""INTELLIGENCE QUERY: {query}

RETRIEVED INTELLIGENCE:
{context}

PRELIMINARY INTELLIGENCE ASSESSMENT:
Overall relevance: {analysis.get("relevance", "unknown").upper()}

Key information points:
{key_points}

Known intelligence gaps:
{gaps}

Source reliability assessment:
{source_assessments}

ANALYTICAL GUIDANCE:
1. Provide a structured intelligence analysis based strictly on the retrieved information.
   - Begin with key judgments or a concise executive summary
   - Focus on the most reliable and relevant information
   - Address the identified intelligence gaps explicitly
   - Be clear about confidence levels in your assessment
   - {f"Include source citations [1], [2], etc. to maintain analytical rigor" if with_citations else "Clearly delineate facts from your assessment"}

2. Maintain appropriate classification handling - treat all information as sensitive but unclassified (SBU)

3. Craft a response that would be valuable to a high-level decision maker."""
            messages = [{"role": "user", "content": user_message}]

        try:
            # Generate final response
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
        Implements an enhanced RAG workflow with query analysis, improved retrieval,
        context assessment, and chain-of-thought generation.

        Args:
            query: User query

        Returns:
            Tuple of (generated response, retrieved context)
        """
        # Check if we have an exact cache hit first (fastest path)
        if self.enable_cache:
            # Use the embedding model from the search index for cache lookup
            def get_embedding(text):
                try:
                    # The model is accessed via FAISSIndex in our search object
                    return self.search.index.model.encode(text).tolist()
                except Exception as e:
                    print(f"Warning: Error generating embedding for cache: {str(e)}")
                    # Return a dummy embedding if model access fails
                    return [0.0] * 384  # Default embedding dimension
                    
            cached_response = self.cache.get(query, get_embedding)
            if cached_response and "response" in cached_response:
                print(f"Using cached response for query: {query}")
                
                # Still update history for tracking
                self.history.append({"role": "user", "content": query})
                self.history.append({"role": "assistant", "content": cached_response["response"]})
                self.save_history()
                
                # Return cached response with empty results (since we don't cache results)
                return cached_response["response"], []
        
        # Check if this might be a general question first
        if self.is_general_question(query):
            # For general questions, don't use the RAG context
            results = []
            context = "No relevant information found."
        else:
            # Step 1: Retrieve relevant context using enhanced query analysis
            results = self.retrieve(query)
            
            # Step 2: Format context for the LLM
            context = self.format_context(results)
        
        # Step 3: Generate response using context with chain-of-thought analysis
        # The generate_response method now handles caching
        response = self.generate_response(query, context)

        # Update conversation history
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": response})
        
        # Persist the chat history after each interaction
        self.save_history()

        return response, results
