# Cipher News Assistant - RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for interacting with news data using Anthropic's Claude 3.5 model.

## Overview

This chatbot combines vector search technology with state-of-the-art language models to provide accurate, context-aware responses to questions about news topics. The system uses:

- **FAISS Vector Search**: For retrieving relevant news paragraphs based on semantic similarity
- **Anthropic Claude 3.5**: For generating natural language responses based on retrieved context
- **RAG Architecture**: To combine retrieval and generation for accurate, grounded answers

## Installation

1. Ensure you have the required dependencies:
```bash
pip install anthropic faiss-cpu sentence-transformers
```

2. Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

## Usage

### Running the Chatbot

```bash
python cipher_chat.py
```

### Command-line Options

- `--api-key KEY`: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
- `--model MODEL`: Claude model to use (default: claude-3-5-sonnet-20240620)
- `--index-dir DIR`: FAISS index directory (default: ./ingestion/vector/data)
- `--results N`: Number of search results to retrieve (default: 5)
- `--max-tokens N`: Maximum tokens in response (default: 1024)
- `--temperature FLOAT`: Temperature for response generation (default: 0.7)
- `--verbose`: Show retrieved context

### Interactive Commands

While using the chatbot, you have access to several commands:

- `/help`: Show available commands
- `/exit`: Exit the chatbot
- `/verbose`: Toggle showing retrieved context
- `/clear`: Clear the screen
- `/topics`: Show available news topics
- `/reports`: Show available reports
- `/topic <name>`: Filter next query by topic
- `/report <id>`: Filter next query by report

## Architecture

### RAGEngine Class

The core engine that combines retrieval and generation:

```python
from ingestion.chat.rag import RAGEngine

# Initialize the engine
engine = RAGEngine(api_key="your_api_key")

# Process a query
response, context = engine.process_query("What's happening in Ukraine?")
```

Key methods:
- `retrieve(query)`: Get relevant context from vector store
- `format_context(results)`: Format retrieved results for the LLM
- `generate_response(query, context)`: Generate response with Claude
- `process_query(query)`: End-to-end RAG pipeline

### CipherCLI Class

The interactive command-line interface:

```python
from ingestion.chat.cli import CipherCLI

# Initialize the CLI
cli = CipherCLI(api_key="your_api_key", verbose=True)

# Run the interactive loop
cli.run()
```

Features:
- Colorful, user-friendly interface
- Command history and editing
- Help system and commands
- Topic and report filtering

## Customization

To modify the system prompt or other generation parameters, edit the `generate_response` method in the `RAGEngine` class.