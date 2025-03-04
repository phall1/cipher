"""
Compatibility module for the new RAG architecture.

This module provides backward compatibility with the original RAG implementation,
allowing existing code to use the new modular architecture without changes.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add project root to the path if needed
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from ingestion.rag.engine import RAGEngine as ModularRAGEngine


class RAGEngine:
    """
    Compatibility wrapper for the original RAG implementation.
    
    This class provides the same interface as the original RAGEngine,
    but uses the new modular architecture under the hood.
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
        Initialize the compatibility RAG engine.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model_name: Claude model to use
            vector_index_dir: Directory containing the FAISS index
            search_results_count: Number of search results to retrieve
            max_tokens: Maximum tokens in the response
            temperature: Temperature for response generation
            history_dir: Directory to store chat history
        """
        # Create the modular RAG engine with the same parameters
        self.engine = ModularRAGEngine(
            api_key=api_key,
            model_name=model_name,
            vector_index_dir=vector_index_dir,
            search_results_count=search_results_count,
            max_tokens=max_tokens,
            temperature=temperature,
            history_dir=history_dir,
            pipeline_type='analytical',  # Use the advanced pipeline by default
            enable_cache=True
        )
        
        # Expose properties from the wrapped engine
        self.search = self.engine.search
        self.history = self.engine.history
        self.history_path = self.engine.history_path
        self.demo_mode = self.engine.demo_mode
    
    def is_general_question(self, query: str) -> bool:
        """
        Determine if a query is likely a general question not related to intelligence matters.
        
        Args:
            query: User query
            
        Returns:
            Boolean indicating if this is likely a general question
        """
        return self.engine.is_general_question(query)
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query
            
        Returns:
            List of relevant context entries
        """
        # Run only the retrieval part of the pipeline
        retrieval_input = {'query': query}
        retrieval_result = self.engine.pipeline.retriever.execute(retrieval_input)
        return retrieval_result.get('results', [])
    
    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results into a structured context string.
        
        Args:
            results: List of search results
            
        Returns:
            Formatted context string
        """
        # Use the processor component to format the context
        processor_input = {'results': results}
        processing_result = self.engine.pipeline.processor.execute(processor_input)
        return processing_result.get('formatted_context', 'No relevant information found.')
    
    def generate_response(self, query: str, context: str, with_citations: bool = True) -> str:
        """
        Generate a response using Claude with the given context.
        
        Args:
            query: User query
            context: Context string
            with_citations: Whether to include citations
            
        Returns:
            Generated response
        """
        # Use the generator component directly
        generator_input = {
            'query': query,
            'formatted_context': context
        }
        generation_result = self.engine.pipeline.generator.execute(
            generator_input, with_citations=with_citations
        )
        return generation_result.get('response', f"Error generating response for: {query}")
    
    def process_query(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process a user query through the full RAG pipeline.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (generated response, retrieved context)
        """
        return self.engine.process_query(query)
    
    def save_history(self):
        """Save the current conversation history to a JSON file."""
        self.engine.save_history()