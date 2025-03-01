# Cipher News Assistant

Cipher is an interactive CLI chatbot that allows you to query a news database using RAG (Retrieval-Augmented Generation) technology. It combines FAISS vector search with the Anthropic Claude API to provide informative responses about current events.

## Features

- Natural language queries about news topics
- Vector search for relevant context
- Topic and report filtering
- Interactive command-line interface
- Citation of sources

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install anthropic faiss-cpu sentence-transformers
   ```

## Configuration

Set your Anthropic API key as an environment variable:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Usage

Run the interactive chat:

```bash
python cipher_chat.py
```

### Command-line options

```
Options:
  --api-key KEY         Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
  --model MODEL         Claude model name (default: claude-3-5-sonnet-20240620)
  --index-dir DIR       FAISS index directory (default: ./ingestion/vector/data)
  --results N           Number of search results to retrieve (default: 5)
  --max-tokens N        Maximum tokens in response (default: 1024)
  --temperature FLOAT   Temperature for response generation (default: 0.7)
  --verbose             Show retrieved context
  --demo                Run in demo mode (non-interactive)
```

### Chat commands

- `/help` - Show help information
- `/exit` - Exit the chatbot
- `/verbose` - Toggle showing retrieved context
- `/clear` - Clear the screen
- `/topics` - Show available topics
- `/reports` - Show available reports
- `/topic <name>` - Filter next query by topic
- `/report <id>` - Filter next query by report

## Demo Mode

For environments where interactive input is problematic, you can use demo mode:

```bash
python cipher_chat.py --demo
```

## Project Structure

- `cipher_chat.py` - Main entry point
- `ingestion/` - Core functionality
  - `chat/` - CLI interface and RAG implementation
  - `vector/` - FAISS vector search and indexing
  - `models.py` - Data models for reports and topics
- `design.md` - Project design documentation

## License

This project is for demonstration purposes only.