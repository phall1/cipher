"""
Base component interface for the RAG system.

This module defines the base Component interface that all RAG components implement.
Components follow a consistent initialization and execution pattern, making them
composable and testable.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Component(ABC):
    """
    Base interface for all RAG components.
    
    Components are the building blocks of the RAG pipeline. Each component
    takes a set of configuration parameters at initialization and implements
    an execute method with a consistent interface.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the component with configuration parameters.
        
        Args:
            **kwargs: Configuration parameters specific to the component
        """
        self.config = kwargs
        self.name = kwargs.get('name', self.__class__.__name__)
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute the component's functionality.
        
        Args:
            input_data: Input data dictionary
            **kwargs: Additional runtime parameters
            
        Returns:
            Output data dictionary with component results
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the component."""
        return f"{self.name}"
    
    def __repr__(self) -> str:
        """Detailed representation of the component."""
        return f"{self.name}(config={self.config})"