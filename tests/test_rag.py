"""
Test suite for the RAG engine and component functionality.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add project root to path for imports
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)

from ingestion.chat.rag import RAGEngine, ResponseCache


class TestResponseCache(unittest.TestCase):
    """Test the ResponseCache functionality."""

    def test_exact_match(self):
        """Test exact match caching."""
        # Create a very strict cache that won't match based on semantic similarity
        cache = ResponseCache(max_size=5, similarity_threshold=1.0)  # Only exact vector matches
        
        # Mock embedding function with deterministic but distinct results
        def mock_response_embed(text):
            if text.lower() == "test query":
                return [0.1, 0.2, 0.3]
            else:
                return [0.9, 0.9, 0.9]  # Very different vector
        
        # Add item to cache
        cache.set("test query", {"response": "test response"}, mock_response_embed)
        
        # Check exact match retrieval
        result = cache.get("test query", mock_response_embed)
        self.assertEqual(result["response"], "test response")
        
        # Check case insensitivity - should work due to normalization
        result = cache.get("Test Query", mock_response_embed)
        self.assertEqual(result["response"], "test response")
        
        # Clear the test by modifying the ResponseCache temporarily
        # This is a bit hacky but necessary for testing
        original_similarity = ResponseCache._cosine_similarity
        
        # Override the cosine similarity to always return low similarity
        def mock_cosine_similarity(self, vec1, vec2):
            return 0.1  # Very low similarity to avoid semantic matches
            
        ResponseCache._cosine_similarity = mock_cosine_similarity
        
        try:
            # Now this should not match semantically
            result = cache.get("nonexistent query", mock_response_embed)
            self.assertIsNone(result)
        finally:
            # Restore original function
            ResponseCache._cosine_similarity = original_similarity

    def test_cache_eviction(self):
        """Test cache eviction when full."""
        # Create a test-specific version of the cache for this test
        cache = ResponseCache(max_size=2, similarity_threshold=0.99)  # Higher threshold to avoid semantic matches
        
        # Create unique embedding functions for each query to avoid semantic matching
        def mock_embed1(text):
            return [0.1, 0.2, 0.3]
            
        def mock_embed2(text):
            return [0.4, 0.5, 0.6]
            
        def mock_embed3(text):
            return [0.7, 0.8, 0.9]
        
        # Add items to cache with different embeddings
        cache.set("query1", {"response": "response1"}, mock_embed1)
        cache.set("query2", {"response": "response2"}, mock_embed2)
        
        # Add another item to trigger eviction (should replace oldest - query1)
        cache.set("query3", {"response": "response3"}, mock_embed3)
        
        # Check query1 (oldest) was evicted
        self.assertIsNone(cache.get("query1", mock_embed1))
        
        # And the other two remain
        self.assertEqual(cache.get("query2", mock_embed2)["response"], "response2")
        self.assertEqual(cache.get("query3", mock_embed3)["response"], "response3")


class TestRAGEngine(unittest.TestCase):
    """Test the RAG engine functionality."""

    @patch('ingestion.vector.search.VectorSearch')
    @patch('anthropic.Anthropic')
    def setUp(self, mock_anthropic, mock_vector_search):
        """Set up the test environment."""
        # Mock the Anthropic client
        self.mock_anthropic_instance = MagicMock()
        mock_anthropic.return_value = self.mock_anthropic_instance
        
        # Mock the VectorSearch
        self.mock_search = MagicMock()
        mock_vector_search.return_value = self.mock_search
        
        # Mock index and model for the search
        self.mock_index = MagicMock()
        self.mock_search.index = self.mock_index
        self.mock_model = MagicMock()
        self.mock_index.model = self.mock_model
        
        # Create temporary directory for history
        import tempfile
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Initialize RAG engine with mocks
        self.rag = RAGEngine(
            api_key="test_key",
            model_name="test_model",
            vector_index_dir="test_dir",
            history_dir=self.temp_dir.name
        )

    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()

    def test_is_general_question(self):
        """Test detection of general questions."""
        # General questions
        self.assertTrue(self.rag.is_general_question("hello"))
        self.assertTrue(self.rag.is_general_question("how are you"))
        self.assertTrue(self.rag.is_general_question("tell me a joke"))
        
        # Intelligence questions
        self.assertFalse(self.rag.is_general_question("russia military capabilities"))
        self.assertFalse(self.rag.is_general_question("cyber threats from China"))
        self.assertFalse(self.rag.is_general_question("intelligence on North Korea nuclear program"))

    @patch('ingestion.chat.rag.RAGEngine._analyze_context')
    @patch('ingestion.chat.rag.RAGEngine._generate_analytical_response')
    def test_generate_response_caching(self, mock_generate, mock_analyze):
        """Test response generation with caching."""
        # Setup mocks
        mock_analyze.return_value = {"relevance": "high", "key_points": ["test point"]}
        mock_generate.return_value = "test response"
        
        # Setup model mocking
        self.mock_model.encode.return_value = [0.1, 0.2, 0.3]
        
        # First call should use the actual generation
        response = self.rag.generate_response("test query", "test context")
        self.assertEqual(response, "test response")
        mock_generate.assert_called_once()
        
        # Reset mocks
        mock_generate.reset_mock()
        
        # Second call with same query should use cache
        response = self.rag.generate_response("test query", "test context")
        self.assertEqual(response, "test response")
        mock_generate.assert_not_called()  # Should not call generate again

    def test_format_context(self):
        """Test context formatting for the LLM"""
        # Create some test results
        test_results = [
            {
                "content": "Test content about intelligence",
                "title": "Intelligence Report",
                "topic": "INTELLIGENCE",
                "report_id": "report-123",
                "distance": 0.1,
                "date": "2025-03-01"
            },
            {
                "content": "Military operations in region X",
                "title": "Military Analysis",
                "topic": "MILITARY",
                "report_id": "report-456",
                "distance": 0.2
            }
        ]
        
        # Format the context
        context = self.rag.format_context(test_results)
        
        # Check that the context contains essential elements
        self.assertIn("INTELLIGENCE BRIEFING MATERIALS", context)
        self.assertIn("CLASSIFICATION: UNCLASSIFIED", context)
        self.assertIn("Intelligence Report", context)
        self.assertIn("Military Analysis", context)
        self.assertIn("Test content about intelligence", context)
        self.assertIn("Military operations in region X", context)
        self.assertIn("CONFIDENCE: HIGH", context)  # Should be high for distance 0.1


def run_tests():
    """Run the test suite."""
    print("\n=== Running RAG Engine Tests ===\n")
    # Create a test loader and test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestResponseCache)
    suite.addTests(loader.loadTestsFromTestCase(TestRAGEngine))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == '__main__':
    run_tests()