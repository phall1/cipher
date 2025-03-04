"""
Response generator components for the RAG system.

This module defines the ResponseGenerator interface and implementations
for generating responses based on queries and processed context.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from .base import Component
from anthropic import Anthropic


class ResponseGenerator(Component):
    """
    Interface for response generator components.
    
    Response generators create responses to user queries based on
    processed context and optionally analysis information.
    """
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Generate a response.
        
        Args:
            input_data: Must contain 'query' and processed context
            **kwargs: Additional runtime parameters
            
        Returns:
            Dictionary with 'response' key containing the generated response
        """
        pass


class BasicResponseGenerator(ResponseGenerator):
    """
    Simple response generator that uses the formatted context.
    
    This generator creates responses based on the formatted context without
    additional processing or analysis.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the basic response generator.
        
        Args:
            api_key: Anthropic API key
            model_name: Model to use for generation
            max_tokens: Maximum tokens in response (default: 1024)
            temperature: Temperature for generation (default: 0.7)
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.api_key = kwargs.get('api_key')
        self.model_name = kwargs.get('model_name', 'claude-3-sonnet-20240229')
        self.max_tokens = kwargs.get('max_tokens', 1024)
        self.temperature = kwargs.get('temperature', 0.7)
        self.demo_mode = kwargs.get('demo_mode', False)
        
        # Initialize Anthropic client if we have an API key
        if self.api_key and not self.demo_mode:
            self.client = Anthropic(api_key=self.api_key)
        else:
            self.client = None
            self.demo_mode = True
    
    def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Generate a response using formatted context.
        
        Args:
            input_data: Must contain 'query' and 'formatted_context' keys
            **kwargs: Additional runtime parameters
            
        Returns:
            Dictionary with 'response' key containing the generated response
        """
        query = input_data.get('query', '')
        context = input_data.get('formatted_context', 'No relevant information found.')
        with_citations = kwargs.get('with_citations', True)
        
        # Check if we're in demo mode
        if self.demo_mode:
            return {'response': f"This is a demo response for: {query}"}
        
        # Build the system prompt
        system_prompt = f"""You are Cipher, an elite AI analyst for The Cipher Brief, specializing in geopolitical intelligence, threat analysis, and national security insights.
Your responses should be precise, authoritative, and insightful - emulating the analytical rigor of intelligence professionals.

CORE CAPABILITIES:
- Geopolitical Analysis: Explain complex international relations, conflicts, and diplomatic developments
- National Security: Analyze threats to national security, defense strategies, and security implications
- Intelligence Tradecraft: Provide expert insights on intelligence operations, trends, and methodologies
- Military Analysis: Assess military capabilities, strategies, and emerging defense technologies
- Cyber Intelligence: Analyze cyber threats, operations, and their implications for national security

When answering questions about geopolitics, security, intelligence matters, or current events from your database,
use only the provided context information. If the context doesn't contain relevant information,
acknowledge this limitation clearly and provide a framework for understanding the issue
based on established analytical methodologies without speculating beyond your knowledge.

However, if the user is asking a general question unrelated to intelligence matters
(such as coding help, math problems, creative writing, or casual conversation),
respond normally as a helpful assistant without the intelligence analyst persona.

{f"When referencing specific intelligence from the context, include citation numbers in brackets like [1], [2], etc." if with_citations else ""}

ANALYTICAL STYLE:
- Structured Analysis: Present information in a clear, logical flow with key judgments clearly identified
- Nuanced Assessment: Highlight degrees of confidence in your analysis and present alternative viewpoints
- Contextual Understanding: Place events in their proper historical and strategic context
- Objective Analysis: Maintain analytical objectivity while presenting multiple perspectives
- Forward-Looking: Identify potential implications and future scenarios when appropriate

For intelligence-related questions, model your responses after high-quality intelligence briefings - factual, concise, and focused on insights that would be valuable to decision-makers."""

        # Check if context is empty or not useful
        context_is_empty = context == "No relevant information found."
        
        # Build the user message
        if context_is_empty:
            # For empty context, just answer normally
            user_message = query
        else:
            # For queries with relevant context, include the context
            user_message = f"""INTELLIGENCE QUERY: {query}

RETRIEVED INTELLIGENCE:
{context}

ANALYTICAL GUIDANCE:
1. If this query relates to geopolitics, national security, intelligence matters, or current events, provide a structured intelligence analysis based strictly on the retrieved information.
   - Begin with key judgments or a concise executive summary
   - Analyze the implications and significance of the information
   - Identify any intelligence gaps or areas of uncertainty
   - Place the information in relevant geopolitical/historical context
   - {f"Include source citations [1], [2], etc. to maintain analytical rigor" if with_citations else "Clearly delineate facts from your assessment"}

2. If this is a general question unrelated to intelligence matters, respond as a helpful assistant without the intelligence analyst persona or references to the retrieved information.

3. Maintain appropriate classification handling - treat all information as sensitive but unclassified (SBU)."""

        try:
            # Generate response
            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            return {'response': response.content[0].text}
        except Exception as e:
            print(f"Error generating response via API: {str(e)}")
            return {'response': f"Error generating response: {str(e)}"}


class AnalyticalResponseGenerator(ResponseGenerator):
    """
    Advanced response generator that uses analysis for Chain-of-Thought generation.
    
    This generator creates responses based on both the formatted context and
    the analysis produced by an analytical context processor. It implements a
    Chain-of-Thought approach for more structured and comprehensive responses.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the analytical response generator.
        
        Args:
            api_key: Anthropic API key
            model_name: Model to use for generation
            max_tokens: Maximum tokens in response (default: 1024)
            temperature: Temperature for generation (default: 0.7)
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.api_key = kwargs.get('api_key')
        self.model_name = kwargs.get('model_name', 'claude-3-sonnet-20240229')
        self.max_tokens = kwargs.get('max_tokens', 1024)
        self.temperature = kwargs.get('temperature', 0.7)
        self.demo_mode = kwargs.get('demo_mode', False)
        
        # Initialize Anthropic client if we have an API key
        if self.api_key and not self.demo_mode:
            self.client = Anthropic(api_key=self.api_key)
        else:
            self.client = None
            self.demo_mode = True
            
        # Templates for different response types can be configured here
        self.prompt_templates = kwargs.get('prompt_templates', {})
    
    def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Generate an analytical response using context and analysis.
        
        Args:
            input_data: Must contain 'query', 'formatted_context', and 'analysis' keys
            **kwargs: Additional runtime parameters
            
        Returns:
            Dictionary with 'response' key containing the generated response
        """
        query = input_data.get('query', '')
        context = input_data.get('formatted_context', 'No relevant information found.')
        analysis = input_data.get('analysis', {})
        with_citations = kwargs.get('with_citations', True)
        history = input_data.get('history', [])
        
        # Check if we're in demo mode
        if self.demo_mode:
            return {'response': f"This is a demo response for: {query}"}
        
        # Build the system prompt
        system_prompt = f"""You are Cipher, an elite AI analyst for The Cipher Brief, specializing in geopolitical intelligence, threat analysis, and national security insights.
Your responses should be precise, authoritative, and insightful - emulating the analytical rigor of intelligence professionals.

CORE CAPABILITIES:
- Geopolitical Analysis: Explain complex international relations, conflicts, and diplomatic developments
- National Security: Analyze threats to national security, defense strategies, and security implications
- Intelligence Tradecraft: Provide expert insights on intelligence operations, trends, and methodologies
- Military Analysis: Assess military capabilities, strategies, and emerging defense technologies
- Cyber Intelligence: Analyze cyber threats, operations, and their implications for national security

When answering questions about geopolitics, security, intelligence matters, or current events from your database,
use only the provided context information. If the context doesn't contain relevant information,
acknowledge this limitation clearly and provide a framework for understanding the issue
based on established analytical methodologies without speculating beyond your knowledge.

However, if the user is asking a general question unrelated to intelligence matters
(such as coding help, math problems, creative writing, or casual conversation),
respond normally as a helpful assistant without the intelligence analyst persona.

{f"When referencing specific intelligence from the context, include citation numbers in brackets like [1], [2], etc." if with_citations else ""}

ANALYTICAL STYLE:
- Structured Analysis: Present information in a clear, logical flow with key judgments clearly identified
- Nuanced Assessment: Highlight degrees of confidence in your analysis and present alternative viewpoints
- Contextual Understanding: Place events in their proper historical and strategic context
- Objective Analysis: Maintain analytical objectivity while presenting multiple perspectives
- Forward-Looking: Identify potential implications and future scenarios when appropriate

For intelligence-related questions, model your responses after high-quality intelligence briefings - factual, concise, and focused on insights that would be valuable to decision-makers."""

        # Context is empty or analysis shows it's not useful
        context_is_empty = (context == "No relevant information found." or 
                           analysis.get("relevance", "").lower() in ["none", "low"] or 
                           not analysis.get("key_points", []))
        
        # Build enhanced user message with analysis insights
        if context_is_empty:
            # For empty/irrelevant context, acknowledge limitations
            user_message = f"""INTELLIGENCE QUERY: {query}

PRELIMINARY INTELLIGENCE ASSESSMENT:
Our intelligence database contains insufficient relevant information on this topic.

ANALYTICAL APPROACH:
1. Acknowledge the intelligence gap clearly to the user
2. If appropriate, suggest a framework for understanding the issue based on established analytical methodologies
3. Do NOT speculate beyond available information or fabricate intelligence
4. Provide a concise response that is honest about the limitations"""
            messages = history + [{"role": "user", "content": user_message}]
        else:
            # Format the analysis insights for the LLM
            key_points = "\n".join([f"- {point}" for point in analysis.get("key_points", [])])
            gaps = "\n".join([f"- {gap}" for gap in analysis.get("gaps", [])])
            
            # Format source reliability information
            source_assessments = ""
            for src in analysis.get("sources_reliability", []):
                source_num = src.get("source_num", "Unknown")
                reliability = src.get("reliability", "Unknown")
                content = src.get("relevant_content", "")
                if content:
                    source_assessments += f"- Source {source_num}: {reliability.upper()} reliability - {content}\n"
            
            # Build enhanced prompt with analytical guidance
            user_message = f"""INTELLIGENCE QUERY: {query}

RETRIEVED INTELLIGENCE:
{context}

PRELIMINARY INTELLIGENCE ASSESSMENT:
Overall relevance: {analysis.get("relevance", "unknown").upper()}

Key information points:
{key_points}

Known intelligence gaps:
{gaps}

Source reliability assessment:
{source_assessments}

ANALYTICAL GUIDANCE:
1. Provide a structured intelligence analysis based strictly on the retrieved information.
   - Begin with key judgments or a concise executive summary
   - Focus on the most reliable and relevant information
   - Address the identified intelligence gaps explicitly
   - Be clear about confidence levels in your assessment
   - {f"Include source citations [1], [2], etc. to maintain analytical rigor" if with_citations else "Clearly delineate facts from your assessment"}

2. Maintain appropriate classification handling - treat all information as sensitive but unclassified (SBU)

3. Craft a response that would be valuable to a high-level decision maker."""
            messages = [{"role": "user", "content": user_message}]

        try:
            # Generate response
            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            return {'response': response.content[0].text}
        except Exception as e:
            print(f"Error generating response via API: {str(e)}")
            return {'response': f"Error generating response: {str(e)}"}