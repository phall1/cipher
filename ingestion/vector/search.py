"""
Vector search API module for the Cipher project.

This module provides a simple API for searching the FAISS index
without having to interact with the CLI tools directly.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to the path if needed
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from ingestion.vector.index import FAISSIndex


class VectorSearch:
    """
    API class for searching embedded paragraphs using vector similarity.
    """
    
    def __init__(
        self,
        index_dir: str = "./ingestion/vector/data",
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the vector search with an existing index.
        
        Args:
            index_dir: Path to the directory containing the FAISS index
            model_name: Name of the sentence transformer model to use
        """
        self.index_dir = index_dir
        self.model_name = model_name
        self._index = None
    
    @property
    def index(self) -> FAISSIndex:
        """
        Lazy-load the FAISS index when first needed.
        
        Returns:
            The loaded FAISS index
        """
        if self._index is None:
            self._index = FAISSIndex(
                model_name=self.model_name,
                index_path=self.index_dir
            )
        return self._index
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_topic: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for paragraphs similar to the query.
        
        Args:
            query: The search query
            k: Number of results to return
            filter_topic: Optional topic name to filter results by
            
        Returns:
            List of dictionaries containing search results with metadata
        """
        # Get raw results from index
        results = self.index.search(query, k=k if not filter_topic else k*3)
        
        # Apply topic filter if specified
        if filter_topic:
            results = [r for r in results if r["topic"] == filter_topic][:k]
            
        return results
    
    def get_topics(self) -> List[str]:
        """
        Get a list of all unique topics in the index.
        
        Returns:
            List of topic names
        """
        return list(set(self.index.topic_names))
    
    def get_reports(self) -> List[str]:
        """
        Get a list of all unique report IDs in the index.
        
        Returns:
            List of report IDs
        """
        return list(set(self.index.report_ids))
    
    def search_by_report(self, query: str, report_id: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search within a specific report.
        
        Args:
            query: The search query
            report_id: ID of the report to search within
            k: Number of results to return
            
        Returns:
            List of dictionaries containing search results with metadata
        """
        # Get a larger set of raw results
        results = self.index.search(query, k=k*3)
        
        # Filter by report ID
        results = [r for r in results if r["report_id"] == report_id][:k]
        
        return results


# Simple usage example
def main():
    # Create search API
    search = VectorSearch()
    
    # Example search
    query = "Trump negotiations with Russia"
    results = search.search(query, k=3)
    
    # Print topics available in the index
    print(f"Available topics: {search.get_topics()}\n")
    
    # Print results
    print(f"Top results for query: '{query}'\n")
    for i, result in enumerate(results):
        print(f"\nResult {i+1} (Distance: {result['distance']:.4f}):")
        print(f"Topic: {result['topic']}")
        print(f"Title: {result['title']}")
        print(f"From Report: {result['report_id']}")
        print(f"Content: {result['content'][:150]}...")
    
    # Example of filtering by topic
    print("\n\nFiltered results (AMERICAS topic only):")
    filtered_results = search.search(query, filter_topic="AMERICAS")
    for i, result in enumerate(filtered_results):
        print(f"\nResult {i+1} (Distance: {result['distance']:.4f}):")
        print(f"Title: {result['title']}")
        print(f"Content: {result['content'][:100]}...")


if __name__ == "__main__":
    main()