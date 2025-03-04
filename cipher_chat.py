#!/usr/bin/env python3
"""
Cipher News Assistant - Interactive CLI Chatbot

This script provides a conversational interface to a news database
using RAG (Retrieval-Augmented Generation) with FAISS vector search
and the Anthropic Claude 3.5 model.

Usage:
    python cipher_chat.py [options]

Options:
    --api-key KEY          Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
    --model MODEL          Claude model name (default: claude-3-5-sonnet-20240620)
    --index-dir DIR        FAISS index directory (default: ./ingestion/vector/data)
    --results N            Number of search results to retrieve (default: 5)
    --max-tokens N         Maximum tokens in response (default: 1024)
    --temperature FLOAT    Temperature for response generation (default: 0.7)
    --verbose              Show retrieved context
    --demo                 Run in demo mode (non-interactive)
"""

import sys
import os
from pathlib import Path

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to the path if needed
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from ingestion.chat.cli import main

if __name__ == "__main__":
    # Automatically use demo mode when running in Claude Code environment
    # or if CLAUDE_CODE_ENV environment variable is set
    is_claude_env = os.environ.get("CLAUDE_CODE_ENV") == "1"
    
    # Add --demo flag to arguments if running in Claude Code environment
    if is_claude_env and "--demo" not in sys.argv:
        sys.argv.append("--demo")
        
    main()