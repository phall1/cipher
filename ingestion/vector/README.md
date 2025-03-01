# Vector Search for News Reports

This module provides FAISS-based vector search functionality for news reports. It embeds paragraphs from news reports and enables semantic search across the content.

## Setup

1. Install dependencies:
```
pip install faiss-cpu sentence-transformers
```

2. Build the index:
```
python ingestion/vector/build_index.py
```

## Usage Options

### 1. CLI Query Tool

Search the index using the CLI tool:
```
python ingestion/vector/query_index.py --query "Your search query here"
```

Options:
- `--index_dir`: Directory containing the FAISS index (default: ./ingestion/vector/data)
- `--query`: Query text to search for (required)
- `--k`: Number of results to return (default: 5)
- `--model`: Sentence transformer model to use (default: all-MiniLM-L6-v2)

### 2. Python API

Import and use the `VectorSearch` class in your application:

```python
from ingestion.vector.search import VectorSearch

# Initialize the search API
search = VectorSearch()

# Get available topics
topics = search.get_topics()
print(f"Available topics: {topics}")

# Basic search
results = search.search("Trump negotiations with Russia", k=3)
for result in results:
    print(f"Title: {result['title']}")
    print(f"Topic: {result['topic']}")
    print(f"Content: {result['content'][:100]}...")

# Search with topic filter
filtered_results = search.search(
    "Ukraine conflict", 
    filter_topic="THE_UKRAINE_UPDATE", 
    k=5
)

# Search within a specific report
report_results = search.search_by_report(
    "China tariffs", 
    report_id="2025-02-27", 
    k=3
)
```

### 3. Interactive Example

Run the example interactive search application:
```
python example_search.py
```

## Available Scripts

- `build_index.py`: Creates and populates a FAISS index from report JSON files
  - `--reports_dir`: Directory containing report JSON files (default: ./ingestion/reports)
  - `--output_dir`: Directory to save the FAISS index (default: ./ingestion/vector/data)
  - `--model`: Sentence transformer model to use (default: all-MiniLM-L6-v2)

- `query_index.py`: CLI tool to search the FAISS index

- `search.py`: API module for programmatic access to the index
  - `VectorSearch.search()`: Search across all content
  - `VectorSearch.search_by_report()`: Search within a specific report
  - `VectorSearch.get_topics()`: Get list of available topics
  - `VectorSearch.get_reports()`: Get list of available reports

## Core Classes

### 1. FAISSIndex (index.py)
Lower-level index management:
- Initialize a new index or load an existing one
- Add reports from JSON files
- Search for paragraphs similar to a query
- Save and load the index and metadata

### 2. VectorSearch (search.py)
High-level search API:
- User-friendly interface for applications
- Topic and report filtering
- Lazy-loading of index