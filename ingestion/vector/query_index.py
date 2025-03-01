import sys
import argparse
from pathlib import Path

# Add project root to the path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from ingestion.vector.index import FAISSIndex

def main():
    parser = argparse.ArgumentParser(description='Query FAISS index')
    parser.add_argument('--index_dir', type=str, default='./ingestion/vector/data',
                        help='Directory containing the FAISS index')
    parser.add_argument('--query', type=str, required=True,
                        help='Query text to search for')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of results to return')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence transformer model to use')
    args = parser.parse_args()
    
    # Check if index exists
    index_path = Path(args.index_dir)
    if not index_path.exists():
        print(f"Index directory {args.index_dir} does not exist!")
        print("Please run build_index.py first to create the index.")
        return
    
    # Load the index
    print(f"Loading index from {args.index_dir}")
    index = FAISSIndex(model_name=args.model, index_path=args.index_dir)
    
    # Search the index
    results = index.search(args.query, k=args.k)
    
    # Display results
    print(f"\nTop {len(results)} results for query: '{args.query}'\n")
    for i, result in enumerate(results):
        print(f"\nResult {i+1} (Distance: {result['distance']:.4f}):")
        print(f"Topic: {result['topic']}")
        print(f"Title: {result['title']}")
        print(f"From Report: {result['report_id']}")
        print(f"Content: {result['content'][:200]}...")
    
    if not results:
        print("No results found.")

if __name__ == "__main__":
    main()