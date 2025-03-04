"""
Modular RAG (Retrieval-Augmented Generation) system for the Cipher application.

This package implements a flexible, composable RAG system with pluggable components
for retrieval, context processing, and response generation.
"""

from .pipeline import RAGPipeline
from .engine import RAGEngine