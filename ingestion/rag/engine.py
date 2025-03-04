"""
RAG engine implementation using the component-based pipeline.

This module provides the main RAG engine that developers interact with.
It builds and configures the RAG pipeline with appropriate components.
"""

import os
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .pipeline import RAGPipeline, PipelineBuilder
from .components.retriever import BasicRetriever, ReformulationRetriever
from .components.processor import BasicContextProcessor, AnalyticalContextProcessor
from .components.generator import BasicResponseGenerator, AnalyticalResponseGenerator
from .components.cache import SemanticResponseCache
from ingestion.vector.search import VectorSearch


class RAGEngine:
    """
    Retrieval-Augmented Generation (RAG) engine for the Cipher chatbot.
    
    This class provides a high-level interface to the RAG functionality,
    handling configuration and pipeline setup.
    """
    
    def __init__(self, **kwargs):
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
            pipeline_type: Type of pipeline to use ('basic', 'analytical', 'custom')
            **kwargs: Additional configuration parameters
        """
        # Extract configuration
        self.api_key = kwargs.get('api_key') or os.environ.get("ANTHROPIC_API_KEY")
        self.model_name = kwargs.get('model_name', 'claude-3-5-sonnet-20240620')
        self.vector_index_dir = kwargs.get('vector_index_dir', './ingestion/vector/data')
        self.search_results_count = kwargs.get('search_results_count', 5)
        self.max_tokens = kwargs.get('max_tokens', 1024)
        self.temperature = kwargs.get('temperature', 0.7)
        self.demo_mode = not bool(self.api_key) or kwargs.get('demo_mode', False)
        self.enable_cache = kwargs.get('enable_cache', True)
        self.pipeline_type = kwargs.get('pipeline_type', 'analytical')
        
        # Chat history setup
        self.history_dir = Path(kwargs.get('history_dir', './ingestion/chat/history'))
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = kwargs.get('session_id', datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.history_path = self.history_dir / f"session_{self.session_id}.json"
        self.history: List[Dict[str, str]] = []
        
        # Initialize pipeline
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the RAG pipeline with appropriate components."""
        builder = PipelineBuilder()
        
        # Common component config
        common_config = {
            'api_key': self.api_key,
            'model_name': self.model_name,
            'demo_mode': self.demo_mode,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature
        }
        
        # Vector search for embedding function (used by cache)
        self.search = VectorSearch(index_dir=self.vector_index_dir)
        
        # Setup cache if enabled
        cache = None
        if self.enable_cache:
            cache = SemanticResponseCache(
                max_size=100,
                similarity_threshold=0.92,
                ttl=3600  # 1 hour TTL
            )
        
        # Configure the pipeline based on type
        if self.pipeline_type == 'basic':
            # Basic pipeline: simpler components, no analysis
            pipeline = builder.with_retriever(
                BasicRetriever(
                    index_dir=self.vector_index_dir,
                    search_results_count=self.search_results_count
                )
            ).with_processor(
                BasicContextProcessor()
            ).with_generator(
                BasicResponseGenerator(**common_config)
            ).with_cache(
                cache
            ).build()
        elif self.pipeline_type == 'custom':
            # Custom pipeline: use components from kwargs
            retriever = kwargs.get('retriever', ReformulationRetriever(
                index_dir=self.vector_index_dir,
                search_results_count=self.search_results_count
            ))
            processor = kwargs.get('processor', AnalyticalContextProcessor(**common_config))
            generator = kwargs.get('generator', AnalyticalResponseGenerator(**common_config))
            
            pipeline = builder.with_retriever(retriever).with_processor(
                processor
            ).with_generator(
                generator
            ).with_cache(
                cache
            ).build()
        else:
            # Default analytical pipeline: advanced components with chain-of-thought
            # Create the query analyzer (used by reformulation retriever)
            analyzer = AnalyticalContextProcessor(**common_config)
            
            pipeline = builder.with_retriever(
                ReformulationRetriever(
                    index_dir=self.vector_index_dir,
                    search_results_count=self.search_results_count,
                    analyzer=analyzer
                )
            ).with_processor(
                AnalyticalContextProcessor(**common_config)
            ).with_generator(
                AnalyticalResponseGenerator(**common_config)
            ).with_cache(
                cache
            ).build()
        
        self.pipeline = pipeline
    
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
    
    def process_query(self, query: str, **kwargs) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process a user query through the full RAG pipeline.
        
        Args:
            query: User query
            **kwargs: Additional parameters for the pipeline
            
        Returns:
            Tuple of (generated response, retrieved context)
        """
        # Define embedding function for cache
        def get_embedding(text):
            try:
                # The model is accessed via the search object
                return self.search.index.model.encode(text).tolist()
            except Exception as e:
                print(f"Warning: Error generating embedding: {str(e)}")
                # Return a dummy embedding if model access fails
                return [0.0] * 384  # Default embedding dimension
        
        # Check if this might be a general question
        if self.is_general_question(query):
            # For general questions, skip RAG and use a direct response
            if self.demo_mode:
                response = f"This is a demo response for general question: {query}"
            else:
                # Use the generator directly for general questions
                basic_generator = BasicResponseGenerator(
                    api_key=self.api_key,
                    model_name=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    demo_mode=self.demo_mode
                )
                result = basic_generator.execute({'query': query, 'formatted_context': ''})
                response = result.get('response', f"Could not generate response for: {query}")
            
            # Update conversation history
            self.history.append({"role": "user", "content": query})
            self.history.append({"role": "assistant", "content": response})
            self.save_history()
            
            return response, []
        
        # Run the full pipeline for intelligence-related questions
        pipeline_result = self.pipeline.run(
            query, 
            embedding_fn=get_embedding,
            history=self.history,
            **kwargs
        )
        
        # Get the response and state
        response = pipeline_result.get('response', '')
        state = pipeline_result.get('state', {})
        
        # Extract results from state
        results = []
        if 'retrieval' in state and 'results' in state['retrieval']:
            results = state['retrieval']['results']
        
        # Update conversation history
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": response})
        self.save_history()
        
        return response, results
    
    def save_history(self):
        """Save the current conversation history to a JSON file."""
        history_data = {
            "session_id": self.session_id,
            "model": self.model_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "messages": self.history
        }
        
        with open(self.history_path, 'w') as f:
            json.dump(history_data, f, indent=2)