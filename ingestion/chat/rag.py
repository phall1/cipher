"""
RAG (Retrieval-Augmented Generation) module for Cipher chatbot.

This module implements a RAG system using FAISS vector search
and the Anthropic Claude model for generating contextual responses.
"""

import os
import sys
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
        model_name: str = "claude-3-5-sonnet-20240620",
        vector_index_dir: str = "./ingestion/vector/data",
        search_results_count: int = 5,
        max_tokens: int = 1024,
        temperature: float = 0.7,
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
        
        # Initialize conversation history
        self.history: List[Dict[str, str]] = []
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query to find relevant context for
            
        Returns:
            List of relevant context entries from the vector store
        """
        return self.search.search(query, k=self.search_results_count)
    
    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results into a context string for the LLM.
        
        Args:
            results: List of search results from the vector store
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found."
        
        context_parts = []
        
        for i, result in enumerate(results):
            # Format each result into a structured context entry
            context_parts.append(
                f"[{i+1}] Topic: {result['topic']}\n"
                f"Title: {result['title']}\n"
                f"Report: {result['report_id']}\n"
                f"Content: {result['content']}\n"
            )
        
        return "\n".join(context_parts)
    
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
        system_prompt = f"""You are Cipher, an AI assistant specializing in news analysis and information retrieval.
Your responses should be accurate, concise, and helpful.

You have access to a database of news reports. When answering questions,
reference only the provided context information. If the context doesn't 
contain relevant information to fully answer the question, acknowledge 
this limitation and provide the best answer with available information.

{f"When referencing information from the provided context, include citation numbers in brackets like [1], [2], etc." if with_citations else ""}

Be factual and objective. Present multiple perspectives when relevant.
Avoid speculation beyond what's in the information provided."""

        # Build the user message with context
        user_message = f"""Question: {query}

Relevant Context:
{context}

Please provide a comprehensive answer based on the context information above.
{f"Include citation numbers [1], [2], etc. to reference which source you're using for each part of your answer." if with_citations else ""}"""

        try:
            # Generate response
            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            return response.content[0].text
        except Exception as e:
            print(f"Error generating response via API: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def process_query(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process a user query through the full RAG pipeline.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (generated response, retrieved context)
        """
        # Step 1: Retrieve relevant context
        results = self.retrieve(query)
        
        # Step 2: Format context for the LLM
        context = self.format_context(results)
        
        # Step 3: Generate response using context
        response = self.generate_response(query, context)
        
        # Update conversation history
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": response})
        
        return response, results