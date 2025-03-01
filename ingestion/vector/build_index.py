import os
import sys
import argparse
from pathlib import Path

# Add project root to the path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from ingestion.vector.index import FAISSIndex
from ingestion.main import load_report_from_json

def main():
    parser = argparse.ArgumentParser(description='Build FAISS index from report files')
    parser.add_argument('--reports_dir', type=str, default='./ingestion/reports',
                        help='Directory containing report JSON files')
    parser.add_argument('--output_dir', type=str, default='./ingestion/vector/data',
                        help='Directory to save the FAISS index')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence transformer model to use')
    args = parser.parse_args()
    
    # Create the FAISS index
    print(f"Initializing FAISS index with model {args.model}")
    index = FAISSIndex(model_name=args.model)
    
    # Add reports from directory
    reports_path = Path(args.reports_dir)
    if not reports_path.exists():
        print(f"Reports directory {args.reports_dir} does not exist!")
        return
    
    # Process all JSON files
    print(f"Processing reports from {args.reports_dir}")
    index.add_reports_from_directory(args.reports_dir)
    
    # Save the index
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving index to {args.output_dir}")
    index.save(args.output_dir)
    
    # Test the index with a sample query
    print("\nTesting index with sample queries:")
    
    test_queries = [
        "Ukraine and Russia negotiations",
        "Trump tariffs on China",
        "European Union security policy",
        "Middle East peace process"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = index.search(query, k=3)
        
        for i, result in enumerate(results):
            print(f"\nResult {i+1} (Distance: {result['distance']:.4f}):")
            print(f"Topic: {result['topic']}")
            print(f"Title: {result['title']}")
            print(f"From Report: {result['report_id']}")
            print(f"Content: {result['content'][:150]}...")

if __name__ == "__main__":
    main()