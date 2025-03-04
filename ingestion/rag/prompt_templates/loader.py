"""
Prompt template loader implementation.

This module provides functionality for loading and rendering prompt templates
from files or strings.
"""

import os
import json
from typing import Any, Dict, List, Optional, Union
from string import Template as StringTemplate


class PromptTemplate:
    """
    Template for LLM prompts with variable substitution.
    
    This class handles loading templates from files or strings and
    rendering them with variable substitution.
    """
    
    def __init__(self, template: str, template_type: str = 'string'):
        """
        Initialize the prompt template.
        
        Args:
            template: Template string or file path
            template_type: Type of template ('string', 'file', or 'json')
        """
        self.template_type = template_type
        
        if template_type == 'file':
            # Load template from file
            with open(template, 'r') as f:
                self.template = f.read()
        elif template_type == 'json':
            # Load template from JSON file
            with open(template, 'r') as f:
                data = json.load(f)
                self.template = data.get('template', '')
                self.metadata = data.get('metadata', {})
        else:
            # Use template string directly
            self.template = template
            
        # Create template object for rendering
        self.template_obj = StringTemplate(self.template)
    
    def render(self, **kwargs) -> str:
        """
        Render the template with variable substitution.
        
        Args:
            **kwargs: Variables to substitute in the template
            
        Returns:
            Rendered template string
        """
        return self.template_obj.safe_substitute(**kwargs)


def load_template(template_path: str) -> PromptTemplate:
    """
    Load a prompt template from a file.
    
    Args:
        template_path: Path to the template file
        
    Returns:
        PromptTemplate instance
    """
    extension = os.path.splitext(template_path)[1].lower()
    
    if extension == '.json':
        return PromptTemplate(template_path, template_type='json')
    else:
        return PromptTemplate(template_path, template_type='file')