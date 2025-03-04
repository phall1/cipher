"""
RAG pipeline implementation.

This module implements a flexible RAG pipeline that allows composing
different components for retrieval, context processing, and response generation.
"""

from typing import Any, Dict, List, Optional, Union, Type

from .components import Component, Retriever, ContextProcessor, ResponseGenerator, CacheStrategy


class RAGPipeline:
    """
    Configurable RAG pipeline for composing RAG components.
    
    The pipeline orchestrates the flow of data between components
    and tracks the execution state. It provides a modular way to
    assemble different retrieval, processing, and generation strategies.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the RAG pipeline.
        
        Args:
            retriever: Retriever component for finding relevant documents
            processor: ContextProcessor component for processing retrieved documents
            generator: ResponseGenerator component for generating responses
            cache: Optional CacheStrategy component for caching
            **kwargs: Additional configuration parameters
        """
        self.retriever = kwargs.get('retriever')
        self.processor = kwargs.get('processor')
        self.generator = kwargs.get('generator')
        self.cache = kwargs.get('cache')
        
        self.config = kwargs
        self.state = {}
    
    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Run the complete RAG pipeline.
        
        Args:
            query: User query
            **kwargs: Additional runtime parameters
            
        Returns:
            Output dictionary with the response and execution state
        """
        # Initialize the pipeline state
        self.state = {
            'query': query,
            'start_time': __import__('time').time(),
            'cache_hit': False
        }
        
        # Step 1: Check cache if available
        if self.cache:
            embedding_fn = kwargs.get('embedding_fn')
            cached_result = self.cache.get(query, embedding_fn)
            if cached_result:
                self.state['cache_hit'] = True
                self.state['response'] = cached_result.get('response')
                self.state['end_time'] = __import__('time').time()
                return {
                    'response': self.state['response'],
                    'state': self.state,
                    'from_cache': True
                }
        
        # Step 2: Retrieve relevant documents
        retrieval_input = {'query': query}
        if kwargs.get('filter_topic'):
            retrieval_input['filter_topic'] = kwargs.get('filter_topic')
        if kwargs.get('report_id'):
            retrieval_input['report_id'] = kwargs.get('report_id')
            
        retrieval_result = self.retriever.execute(retrieval_input, **kwargs)
        self.state['retrieval'] = retrieval_result
        
        # Step 3: Process the context
        processor_input = {'query': query, **retrieval_result}
        if kwargs.get('history'):
            processor_input['history'] = kwargs.get('history')
            
        processing_result = self.processor.execute(processor_input, **kwargs)
        self.state['processing'] = processing_result
        
        # Step 4: Generate the response
        generator_input = {'query': query, **processing_result}
        if kwargs.get('history'):
            generator_input['history'] = kwargs.get('history')
            
        generation_result = self.generator.execute(generator_input, **kwargs)
        self.state['generation'] = generation_result
        self.state['response'] = generation_result.get('response')
        
        # Step 5: Cache the result if applicable
        if self.cache:
            embedding_fn = kwargs.get('embedding_fn')
            self.cache.set(query, {'response': self.state['response']}, embedding_fn)
        
        # Finalize and return
        self.state['end_time'] = __import__('time').time()
        self.state['execution_time'] = self.state['end_time'] - self.state['start_time']
        
        return {
            'response': self.state['response'],
            'state': self.state,
            'from_cache': False
        }


class PipelineBuilder:
    """
    Builder pattern for constructing RAG pipelines.
    
    This class provides a fluent interface for building RAG pipelines
    with different components and configurations.
    """
    
    def __init__(self):
        """Initialize an empty pipeline configuration."""
        self.config = {}
    
    def with_retriever(self, retriever: Retriever) -> 'PipelineBuilder':
        """
        Set the retriever component.
        
        Args:
            retriever: Retriever component for finding relevant documents
            
        Returns:
            Self for method chaining
        """
        self.config['retriever'] = retriever
        return self
    
    def with_processor(self, processor: ContextProcessor) -> 'PipelineBuilder':
        """
        Set the context processor component.
        
        Args:
            processor: ContextProcessor component for processing retrieved documents
            
        Returns:
            Self for method chaining
        """
        self.config['processor'] = processor
        return self
    
    def with_generator(self, generator: ResponseGenerator) -> 'PipelineBuilder':
        """
        Set the response generator component.
        
        Args:
            generator: ResponseGenerator component for generating responses
            
        Returns:
            Self for method chaining
        """
        self.config['generator'] = generator
        return self
    
    def with_cache(self, cache: CacheStrategy) -> 'PipelineBuilder':
        """
        Set the cache strategy component.
        
        Args:
            cache: CacheStrategy component for caching
            
        Returns:
            Self for method chaining
        """
        self.config['cache'] = cache
        return self
    
    def with_config(self, **kwargs) -> 'PipelineBuilder':
        """
        Set additional configuration parameters.
        
        Args:
            **kwargs: Additional configuration parameters
            
        Returns:
            Self for method chaining
        """
        self.config.update(kwargs)
        return self
    
    def build(self) -> RAGPipeline:
        """
        Build the RAG pipeline.
        
        Returns:
            Configured RAGPipeline instance
        """
        # Validate required components
        if 'retriever' not in self.config:
            raise ValueError("Retriever component is required")
        if 'processor' not in self.config:
            raise ValueError("Processor component is required")
        if 'generator' not in self.config:
            raise ValueError("Generator component is required")
        
        return RAGPipeline(**self.config)