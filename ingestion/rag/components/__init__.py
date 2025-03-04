"""
Component interfaces and implementations for the RAG system.

This package contains the core components of the RAG system:
- Retrievers: Components that retrieve relevant context from data sources
- Processors: Components that process and format context
- Generators: Components that generate responses using context
- Cache: Components for caching responses and embeddings
"""

from .base import Component
from .retriever import Retriever
from .processor import ContextProcessor
from .generator import ResponseGenerator
from .cache import CacheStrategy