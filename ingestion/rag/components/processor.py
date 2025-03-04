"""
Context processor components for the RAG system.

This module defines the ContextProcessor interface and implementations
for processing retrieved context before using it for response generation.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional

from .base import Component
from anthropic import Anthropic


class ContextProcessor(Component):
    """
    Interface for context processor components.
    
    Context processors transform retrieved documents into a format suitable
    for the response generator, potentially adding analysis or filtering.
    """
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process the retrieved context.
        
        Args:
            input_data: Must contain 'results' key with retrieved documents
            **kwargs: Additional runtime parameters
            
        Returns:
            Dictionary with 'context' key containing processed context
        """
        pass


class BasicContextProcessor(ContextProcessor):
    """
    Simple context processor that formats retrieved documents for the LLM.
    
    This processor converts retrieved documents into a structured format
    suitable for the response generator.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the basic context processor.
        
        Args:
            **kwargs: Configuration parameters
        """
        super().__init__(**kwargs)
    
    def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Format retrieved documents into structured context.
        
        Args:
            input_data: Must contain 'results' key with retrieved documents
            **kwargs: Additional runtime parameters
            
        Returns:
            Dictionary with 'context' key containing formatted context string
        """
        results = input_data.get('results', [])
        
        if not results:
            return {'context': "No relevant information found."}
        
        context_parts = []
        
        for i, result in enumerate(results):
            # Calculate confidence score (transform distance to confidence)
            confidence = 1.0 - (result.get('distance', 0.5) / 2)
            confidence_level = "HIGH" if confidence > 0.8 else "MEDIUM" if confidence > 0.6 else "LOW"
            
            # Format timestamp if available, otherwise use generic date
            report_date = result.get('date', 'N/A')
            
            # Format each result into a structured intelligence entry
            context_parts.append(
                f"[SOURCE {i+1}] {'-'*50}\n"
                f"CLASSIFICATION: UNCLASSIFIED // FOR OFFICIAL USE ONLY\n"
                f"SUBJECT: {result['title']}\n"
                f"REFERENCE ID: {result['report_id']}\n"
                f"TOPIC: {result['topic']}\n"
                f"DATE: {report_date}\n" 
                f"CONFIDENCE: {confidence_level} ({confidence:.2f})\n"
                f"CONTENT:\n{result['content']}\n"
                f"{'='*60}\n"
            )
        
        # Add header to the context
        header = f"""
INTELLIGENCE BRIEFING MATERIALS
CLASSIFICATION: UNCLASSIFIED // FOR OFFICIAL USE ONLY
RETRIEVED: {len(results)} RELEVANT SOURCES
{'='*60}
"""
        
        # Add footer with handling instructions
        footer = f"""
{'='*60}
HANDLING INSTRUCTIONS: Information is UNCLASSIFIED but sensitive. 
Provide appropriate context and source citation in analysis.
"""
        
        return {'context': header + "\n".join(context_parts) + footer}


class AnalyticalContextProcessor(ContextProcessor):
    """
    Advanced context processor that analyzes retrieved documents.
    
    This processor evaluates the relevance and reliability of each document,
    extracts key information, and identifies gaps to produce a comprehensive
    analysis for the response generator.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the analytical context processor.
        
        Args:
            api_key: Anthropic API key
            model_name: Model to use for analysis
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.api_key = kwargs.get('api_key')
        self.model_name = kwargs.get('model_name', 'claude-3-haiku-20240307')
        self.demo_mode = kwargs.get('demo_mode', False)
        
        # Initialize Anthropic client if we have an API key
        if self.api_key and not self.demo_mode:
            self.client = Anthropic(api_key=self.api_key)
        else:
            self.client = None
            self.demo_mode = True
    
    def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process and analyze retrieved context.
        
        Args:
            input_data: Must contain 'results' key with retrieved documents and 'query' key
            **kwargs: Additional runtime parameters
            
        Returns:
            Dictionary with 'context', 'formatted_context', and 'analysis' keys
        """
        results = input_data.get('results', [])
        query = input_data.get('query', '')
        
        # Format the context first (we'll use this regardless of analysis)
        formatted_context = self._format_context(results)
        
        # If we're in demo mode or have no results, skip the analysis
        if self.demo_mode or not results:
            return {
                'context': results,
                'formatted_context': formatted_context,
                'analysis': {
                    'relevance': 'unknown' if results else 'none',
                    'key_points': [],
                    'gaps': ["No relevant information found."] if not results else ["Unable to analyze in demo mode."],
                    'sources_reliability': []
                }
            }
        
        # Analyze the context using the LLM
        analysis = self._analyze_context(query, formatted_context)
        
        return {
            'context': results,
            'formatted_context': formatted_context,
            'analysis': analysis
        }
    
    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format the retrieved results into a structured context string.
        
        Args:
            results: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found."
        
        context_parts = []
        
        for i, result in enumerate(results):
            # Calculate confidence score
            confidence = 1.0 - (result.get('distance', 0.5) / 2)
            confidence_level = "HIGH" if confidence > 0.8 else "MEDIUM" if confidence > 0.6 else "LOW"
            
            # Format timestamp if available
            report_date = result.get('date', 'N/A')
            
            # Format each result
            context_parts.append(
                f"[SOURCE {i+1}] {'-'*50}\n"
                f"CLASSIFICATION: UNCLASSIFIED // FOR OFFICIAL USE ONLY\n"
                f"SUBJECT: {result['title']}\n"
                f"REFERENCE ID: {result['report_id']}\n"
                f"TOPIC: {result['topic']}\n"
                f"DATE: {report_date}\n" 
                f"CONFIDENCE: {confidence_level} ({confidence:.2f})\n"
                f"CONTENT:\n{result['content']}\n"
                f"{'='*60}\n"
            )
        
        # Add header and footer
        header = f"""
INTELLIGENCE BRIEFING MATERIALS
CLASSIFICATION: UNCLASSIFIED // FOR OFFICIAL USE ONLY
RETRIEVED: {len(results)} RELEVANT SOURCES
{'='*60}
"""
        
        footer = f"""
{'='*60}
HANDLING INSTRUCTIONS: Information is UNCLASSIFIED but sensitive. 
Provide appropriate context and source citation in analysis.
"""
        
        return header + "\n".join(context_parts) + footer
    
    def _analyze_context(self, query: str, context: str) -> Dict[str, Any]:
        """
        Analyze the context for relevance and extract key information.
        
        Args:
            query: The user's query
            context: Formatted context string
            
        Returns:
            Analysis dictionary with relevance, key points, gaps, and source reliability
        """
        # Skip analysis for empty context
        if context == "No relevant information found.":
            return {
                'relevance': 'none',
                'key_points': [],
                'gaps': ["No relevant information found in the intelligence database."],
                'sources_reliability': []
            }
        
        system_prompt = """You are an expert intelligence analyst specializing in evaluating intelligence information.
Your task is to analyze the retrieved intelligence sources and assess their relevance to the query.

Examine each intelligence source objectively and identify:
1. The relevance of each source to the specific query (high/medium/low/none)
2. Key facts and data points that address the query
3. Information gaps or limitations in the available intelligence
4. Reliability assessment of each source based on confidence levels

Respond in JSON format:
{
  "relevance": "overall_relevance",
  "key_points": ["point1", "point2", ...],
  "gaps": ["gap1", "gap2", ...],
  "sources_reliability": [
    {"source_num": 1, "reliability": "high/medium/low", "relevant_content": "summary of key info"}
  ]
}"""

        try:
            # Use the API to analyze the context
            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=[{
                    "role": "user", 
                    "content": f"QUERY: {query}\n\nINTELLIGENCE SOURCES:\n{context}"
                }],
                max_tokens=1500,
                temperature=0.0  # Deterministic for consistency
            )
            
            # Extract JSON from the response
            import json
            import re
            
            content = response.content[0].text
            # Find JSON in the content (handling formatting variations)
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = content
                
            try:
                # Try to parse as-is
                analysis = json.loads(json_str)
            except json.JSONDecodeError:
                # Try to find the JSON object using regex if direct parsing fails
                json_pattern = r'\{.*\}'
                json_match = re.search(json_pattern, json_str, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group(0))
                else:
                    # Fallback to structured extraction if JSON parsing fails
                    analysis = {
                        "relevance": "medium" if "relevant" in content.lower() else "low",
                        "key_points": re.findall(r'key points?:?(.*?)(?:gaps|$)', content, re.IGNORECASE | re.DOTALL)[0].split('\n') if re.findall(r'key points?:?(.*?)(?:gaps|$)', content, re.IGNORECASE | re.DOTALL) else [],
                        "gaps": re.findall(r'gaps?:?(.*?)(?:sources|$)', content, re.IGNORECASE | re.DOTALL)[0].split('\n') if re.findall(r'gaps?:?(.*?)(?:sources|$)', content, re.IGNORECASE | re.DOTALL) else ["Unable to determine specific gaps"],
                        "sources_reliability": []
                    }
            
            return analysis
            
        except Exception as e:
            print(f"Error during context analysis: {str(e)}")
            # Return basic analysis on error
            return {
                "relevance": "unknown",
                "key_points": [],
                "gaps": ["Error occurred during intelligence analysis."],
                "sources_reliability": []
            }