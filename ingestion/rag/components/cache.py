"""
Cache components for the RAG system.

This module defines cache strategies for optimizing performance by storing
and reusing previously computed results.
"""

import hashlib
import time
import numpy as np
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable

from .base import Component


class CacheStrategy(Component):
    """
    Interface for cache strategy components.
    
    Cache strategies provide mechanisms for storing and retrieving
    previously computed results to improve performance.
    """
    
    @abstractmethod
    def get(self, key: str, embedding_fn: Callable[[str], List[float]]) -> Optional[Dict[str, Any]]:
        """
        Retrieve a value from the cache.
        
        Args:
            key: The key to look up
            embedding_fn: Function to generate embeddings for semantic comparison
            
        Returns:
            Cached value or None if not found
        """
        pass
    
    @abstractmethod
    def set(self, key: str, value: Dict[str, Any], embedding_fn: Callable[[str], List[float]]) -> None:
        """
        Store a value in the cache.
        
        Args:
            key: The key to store
            value: The value to cache
            embedding_fn: Function to generate embeddings for semantic comparison
        """
        pass


class SemanticResponseCache(CacheStrategy):
    """
    Cache that matches queries based on semantic similarity.
    
    This cache can retrieve values based on exact key matches or
    semantic similarity of query embeddings, allowing it to reuse
    responses for semantically equivalent queries.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the semantic response cache.
        
        Args:
            max_size: Maximum number of entries to store (default: 100)
            similarity_threshold: Minimum similarity score for a cache hit (default: 0.92)
            ttl: Time-to-live for cache entries in seconds (default: 3600)
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.max_size = kwargs.get('max_size', 100)
        self.similarity_threshold = kwargs.get('similarity_threshold', 0.92)
        self.ttl = kwargs.get('ttl', 3600)
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.query_embeddings: Dict[str, List[float]] = {}
    
    def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute the cache component.
        
        Args:
            input_data: Must contain either 'get' or 'set' operation and relevant data
            **kwargs: Additional runtime parameters
            
        Returns:
            Dictionary with operation results
        """
        operation = input_data.get('operation')
        if operation == 'get':
            key = input_data.get('key', '')
            embedding_fn = input_data.get('embedding_fn')
            result = self.get(key, embedding_fn)
            return {'hit': result is not None, 'value': result}
        elif operation == 'set':
            key = input_data.get('key', '')
            value = input_data.get('value', {})
            embedding_fn = input_data.get('embedding_fn')
            self.set(key, value, embedding_fn)
            return {'success': True}
        else:
            return {'error': 'Invalid operation'}
    
    def get(self, key: str, embedding_fn: Callable[[str], List[float]]) -> Optional[Dict[str, Any]]:
        """
        Get a cached response for a query.
        
        Args:
            key: The query to lookup
            embedding_fn: Function to generate embeddings for semantic comparison
            
        Returns:
            Cached response or None if not found
        """
        # First check for exact match (faster)
        query_hash = self._hash_query(key)
        if query_hash in self.cache:
            entry = self.cache[query_hash]
            # Check if the entry has expired
            if time.time() - entry['timestamp'] < self.ttl:
                # Update access time
                entry['last_accessed'] = time.time()
                print(f"Cache hit: exact match for '{key}'")
                return entry['data']
            else:
                # Remove expired entry
                del self.cache[query_hash]
                if query_hash in self.query_embeddings:
                    del self.query_embeddings[query_hash]
        
        # If no exact match, try semantic matching
        try:
            query_embedding = embedding_fn(key)
            
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
                print(f"Cache hit: semantic match ({max_similarity:.3f}) for '{key}'")
                return entry['data']
                
        except Exception as e:
            print(f"Error during semantic cache lookup: {str(e)}")
        
        return None
    
    def set(self, key: str, value: Dict[str, Any], embedding_fn: Callable[[str], List[float]]) -> None:
        """
        Cache a response for a query.
        
        Args:
            key: The query to cache
            value: The data to cache
            embedding_fn: Function to generate embeddings for semantic comparison
        """
        # Check if we need to evict entries
        if len(self.cache) >= self.max_size:
            self._evict_entries()
        
        # Add new entry
        query_hash = self._hash_query(key)
        now = time.time()
        self.cache[query_hash] = {
            'data': value,
            'timestamp': now,
            'last_accessed': now
        }
        
        # Store embedding for semantic matching
        try:
            self.query_embeddings[query_hash] = embedding_fn(key)
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