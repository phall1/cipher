#!/usr/bin/env python3
"""
Example script demonstrating how to use the VectorSearch API in an application.
"""

import sys
from pathlib import Path

# Add project root to path if needed
project_root = str(Path(__file__).resolve().parents[0])
if project_root not in sys.path:
    sys.path.append(project_root)

from ingestion.vector.search import VectorSearch


def main():
    # Create the search object - this will lazy-load the index when needed
    search = VectorSearch()
    
    print("=== Cipher Vector Search Example ===\n")
    
    # Display available topics
    topics = search.get_topics()
    print("Available topics:")
    for topic in topics:
        print(f"  - {topic}")
    
    # Display available reports
    reports = search.get_reports()
    print("\nAvailable reports:")
    for report in reports:
        print(f"  - {report}")
    
    while True:
        # Get user query
        print("\nEnter a search query (or 'q' to quit):")
        query = input("> ")
        
        if query.lower() in ('q', 'quit', 'exit'):
            break
        
        # Get optional topic filter
        print("\nFilter by topic? Enter topic name or leave blank for all:")
        for i, topic in enumerate(topics):
            print(f"  {i+1}. {topic}")
        
        topic_input = input("> ")
        topic_filter = None
        
        if topic_input.isdigit() and 1 <= int(topic_input) <= len(topics):
            topic_filter = topics[int(topic_input) - 1]
        elif topic_input in topics:
            topic_filter = topic_input
        
        # Perform search
        results = search.search(query, filter_topic=topic_filter)
        
        # Display results
        if not results:
            print("\nNo results found.")
            continue
            
        print(f"\nTop {len(results)} results for: '{query}'")
        if topic_filter:
            print(f"(Filtered by topic: {topic_filter})")
        
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} (Relevance: {1.0 - result['distance']/2:.2f}) ---")
            print(f"Topic: {result['topic']}")
            print(f"Title: {result['title']}")
            print(f"From: {result['report_id']}")
            
            # Show a snippet of the content
            content = result['content']
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"\n{content}")
    
    print("\nThank you for using Cipher Vector Search!")


if __name__ == "__main__":
    main()