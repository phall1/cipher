"""
Retriever components for the RAG system.

This module defines the Retriever interface and implementations for retrieving
relevant context from various data sources.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from .base import Component
from ingestion.vector.search import VectorSearch


class Retriever(Component):
    """
    Interface for retriever components.
    
    Retrievers are responsible for fetching relevant context based on a query.
    They accept a query string and return a list of relevant documents or passages.
    """
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute the retrieval operation.
        
        Args:
            input_data: Must contain 'query' key with the user's query
            **kwargs: Additional runtime parameters
            
        Returns:
            Dictionary with 'results' key containing retrieved documents
        """
        pass


class BasicRetriever(Retriever):
    """
    Simple retriever that performs direct semantic search.
    
    This retriever uses the vector search index to find semantically
    similar documents without additional processing.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the basic retriever.
        
        Args:
            index_dir: Directory containing the vector index
            search_results_count: Number of results to retrieve (default: 5)
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.index_dir = kwargs.get('index_dir', './ingestion/vector/data')
        self.search_results_count = kwargs.get('search_results_count', 5)
        self.search = VectorSearch(index_dir=self.index_dir)
    
    def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Perform basic semantic search.
        
        Args:
            input_data: Must contain 'query' key with the user's query
            **kwargs: Additional runtime parameters
            
        Returns:
            Dictionary with 'results' key containing retrieved documents
        """
        query = input_data.get('query', '')
        filter_topic = input_data.get('filter_topic')
        
        # Override the result count if specified at runtime
        results_count = kwargs.get('results_count', self.search_results_count)
        
        # Perform the search
        if filter_topic:
            results = self.search.search(query, k=results_count, filter_topic=filter_topic)
        elif 'report_id' in input_data:
            results = self.search.search_by_report(query, report_id=input_data['report_id'], k=results_count)
        else:
            results = self.search.search(query, k=results_count)
        
        # Sort by relevance
        sorted_results = sorted(results, key=lambda x: x.get('distance', float('inf')))
        
        return {'results': sorted_results[:results_count]}


class ReformulationRetriever(Retriever):
    """
    Advanced retriever that reformulates queries for better retrieval.
    
    This retriever analyzes the query, generates multiple reformulations,
    and combines results from all reformulations for more comprehensive retrieval.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the reformulation retriever.
        
        Args:
            index_dir: Directory containing the vector index
            search_results_count: Number of results to retrieve (default: 5)
            analyzer: Component for analyzing and reformulating queries
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.index_dir = kwargs.get('index_dir', './ingestion/vector/data')
        self.search_results_count = kwargs.get('search_results_count', 5)
        self.search = VectorSearch(index_dir=self.index_dir)
        
        # The analyzer is injected from the pipeline
        self.analyzer = kwargs.get('analyzer')
    
    def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Perform retrieval with query reformulation.
        
        Args:
            input_data: Must contain 'query' key with the user's query
            **kwargs: Additional runtime parameters
            
        Returns:
            Dictionary with 'results' key containing retrieved documents and
            'analysis' key containing query analysis information
        """
        query = input_data.get('query', '')
        filter_topic = input_data.get('filter_topic')
        results_count = kwargs.get('results_count', self.search_results_count)
        
        # First analyze the query if we have an analyzer
        if self.analyzer:
            analysis_result = self.analyzer.execute({'query': query})
            analysis = analysis_result.get('analysis', {
                'entities': [],
                'intent': 'information',
                'reformulations': [query]
            })
        else:
            # Basic fallback if no analyzer
            analysis = {
                'entities': [w for w in query.split() if len(w) > 3],
                'intent': 'information',
                'reformulations': [query]
            }
        
        # Get reformulations from analysis
        reformulations = analysis.get('reformulations', [query])
        
        # Make sure original query is included
        if query not in reformulations:
            reformulations.append(query)
        
        # Search with all reformulations
        all_results = []
        existing_ids = set()
        
        for q in reformulations:
            # Search with each reformulated query
            if filter_topic:
                results = self.search.search(q, k=results_count, filter_topic=filter_topic)
            elif 'report_id' in input_data:
                results = self.search.search_by_report(q, report_id=input_data['report_id'], k=results_count)
            else:
                results = self.search.search(q, k=results_count)
            
            # Add any new results
            for result in results:
                result_id = result.get('id', hash(result.get('content', '')))
                if result_id not in existing_ids:
                    all_results.append(result)
                    existing_ids.add(result_id)
        
        # If we don't have enough results, try searching with key entities
        if len(all_results) < results_count:
            entities = analysis.get('entities', [])
            for entity in entities:
                if len(entity) > 3:  # Only use substantial entities
                    entity_results = self.search.search(entity, k=results_count // 2)
                    
                    # Add any new entity results
                    for result in entity_results:
                        result_id = result.get('id', hash(result.get('content', '')))
                        if result_id not in existing_ids:
                            all_results.append(result)
                            existing_ids.add(result_id)
                            
                            # Stop if we have enough results
                            if len(all_results) >= results_count * 2:
                                break
        
        # Sort by relevance and limit to the top results
        sorted_results = sorted(all_results, key=lambda x: x.get('distance', float('inf')))
        limited_results = sorted_results[:results_count]
        
        return {
            'results': limited_results,
            'analysis': analysis
        }