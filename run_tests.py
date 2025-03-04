#!/usr/bin/env python3
"""
Test runner for the Cipher Intelligence Assistant.

This script runs the test suite and provides a simple way to test specific components.
It can also be used to demonstrate the functionality of the RAG engine with a mock
implementation to avoid API calls.
"""

import os
import sys
from pathlib import Path

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to the path if needed
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Make sure tests directory is in the path
tests_dir = os.path.join(project_root, "tests")
if tests_dir not in sys.path:
    sys.path.append(tests_dir)

from tests.test_rag import run_tests as run_rag_tests


def display_progress(message):
    """Display a progress message to the user."""
    print(f"\n>>> {message}...")


def run_all_tests():
    """Run all test suites."""
    display_progress("Running RAG Engine tests")
    run_rag_tests()
    

def demo_rag():
    """
    Demonstrate the RAG engine functionality with mocks.
    This allows testing the flow without making actual API calls.
    """
    from unittest.mock import MagicMock, patch
    from ingestion.chat.rag_compat import RAGEngine
    
    display_progress("Setting up mock RAG engine")
    
    # Create mock for Anthropic API
    with patch('anthropic.Anthropic') as mock_anthropic, \
         patch('ingestion.vector.search.VectorSearch') as mock_search:
        
        # Setup mock responses
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        # Setup mock message response
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Mock response from Claude")]
        mock_client.messages.create.return_value = mock_message
        
        # Setup mock search
        search_instance = MagicMock()
        mock_search.return_value = search_instance
        
        # Setup mock search results
        search_instance.search.return_value = [
            {
                "id": "doc1",
                "content": "Test content about intelligence",
                "title": "Intelligence Report",
                "topic": "INTELLIGENCE",
                "report_id": "report-123",
                "distance": 0.1
            }
        ]
        
        # Setup mock index and model
        mock_index = MagicMock()
        search_instance.index = mock_index
        mock_model = MagicMock()
        mock_index.model = mock_model
        mock_model.encode.return_value = [0.1, 0.2, 0.3]
        
        # Initialize RAG engine in demo mode (no API calls)
        os.environ["ANTHROPIC_API_KEY"] = ""  # Force demo mode
        rag = RAGEngine(
            api_key="",  # Empty key forces demo mode
            model_name="mock_model",
            enable_cache=True
        )
        # Explicitly set demo mode
        rag.demo_mode = True
        
        # Test simple query
        display_progress("Testing simple query")
        response, results = rag.process_query("Tell me about intelligence")
        
        print("\nQuery: Tell me about intelligence")
        print(f"Response: {response}")
        print(f"Retrieved {len(results)} documents")
        
        # Test cached response
        display_progress("Testing cached response")
        response2, results2 = rag.process_query("Tell me about intelligence")
        
        print("\nCache hit should be used for identical query")
        print(f"Response: {response2}")
        
        # Test different query
        display_progress("Testing different query")
        response3, results3 = rag.process_query("What is cyber security?")
        
        print("\nQuery: What is cyber security?")
        print(f"Response: {response3}")
        print(f"Retrieved {len(results3)} documents")
        
        print("\nDemo completed successfully")


def run_interactive_test():
    """
    Run an interactive test that lets the user try the RAG engine.
    Uses actual API if available, otherwise uses mock.
    """
    from ingestion.chat.rag_compat import RAGEngine
    
    display_progress("Setting up RAG engine for interactive testing")
    
    # Check if API key is available
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("No API key found. Running in demo mode.")
        demo_rag()
        return
    
    # Initialize RAG engine
    rag = RAGEngine(
        api_key=api_key,
        model_name="claude-3-haiku-20240307",  # Use a smaller model for testing
        enable_cache=True
    )
    
    print("\nInteractive RAG Engine Test")
    print("---------------------------")
    print("Type 'exit' to quit")
    
    while True:
        query = input("\nEnter query: ")
        if query.lower() in ("exit", "quit"):
            break
            
        print("Processing query...")
        try:
            response, results = rag.process_query(query)
            
            print(f"\nRetrieved {len(results)} documents")
            print(f"\nResponse:\n{response}")
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test runner for Cipher Intelligence Assistant")
    parser.add_argument("--test", choices=["all", "rag"], default="all", help="Test suite to run")
    parser.add_argument("--demo", action="store_true", help="Run demo with mock implementation")
    parser.add_argument("--interactive", action="store_true", help="Run interactive test")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_rag()
    elif args.interactive:
        run_interactive_test()
    elif args.test == "all":
        run_all_tests()
    elif args.test == "rag":
        run_rag_tests()