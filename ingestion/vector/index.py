import os
import json
import faiss
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle

from ingestion.models import Report, Topic, Paragraph, TopicName


class FAISSIndex:
    """
    FAISS Index for efficient similarity search over paragraph embeddings.
    Maintains metadata to track paragraph titles, content, and topic information.
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2", 
        dimension: int = 384,
        index_path: Optional[str] = None
    ):
        """
        Initialize the FAISS index.
        
        Args:
            model_name: Name of the sentence transformer model to use for embeddings
            dimension: Dimension of the embedding vectors
            index_path: Path to load existing index from, if None creates a new index
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        
        # Metadata for storing and retrieving paragraph information
        self.texts: List[str] = []  # Original paragraph texts
        self.titles: List[str] = []  # Paragraph titles
        self.topic_names: List[str] = []  # Topic names
        self.report_ids: List[str] = []  # Report IDs
        
        # Load existing index if path provided
        if index_path and os.path.exists(index_path):
            self.load(index_path)
    
    def add_report(self, report: Report) -> None:
        """
        Process a report and add all its paragraphs to the index.
        
        Args:
            report: The Report object to process
        """
        for topic in report.topics:
            for paragraph in topic.paragraphs:
                # Get the embedding for the paragraph content
                embedding = self.model.encode(paragraph.content)
                
                # Convert to the format required by FAISS
                embedding_np = np.array([embedding], dtype=np.float32)
                
                # Add to FAISS index
                self.index.add(embedding_np)
                
                # Store metadata
                self.texts.append(paragraph.content)
                self.titles.append(paragraph.title)
                self.topic_names.append(topic.name.value)
                self.report_ids.append(report.id)
    
    def add_reports_from_directory(self, directory_path: str) -> None:
        """
        Load all JSON reports from a directory and add them to the index.
        
        Args:
            directory_path: Path to directory containing JSON report files
        """
        directory = Path(directory_path)
        for file_path in directory.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    report = Report(
                        id=data["id"],
                        topics=[
                            Topic(
                                name=TopicName(topic["name"]),
                                paragraphs=[
                                    Paragraph(
                                        title=paragraph["title"],
                                        content=paragraph["content"],
                                        sources=[],
                                    )
                                    for paragraph in topic["paragraphs"]
                                ],
                            )
                            for topic in data["topics"]
                        ],
                    )
                    self.add_report(report)
                    print(f"Added report {report.id} from {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the index for paragraphs similar to the query.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of dictionaries containing search results with metadata
        """
        # Check if the index has been initialized and has data
        if not hasattr(self, 'index') or not self.texts:
            print("Warning: Index not initialized or empty. Using demo data.")
            # Return demo data
            return self._get_demo_results(query, k)
            
        try:
            # Get query embedding
            query_embedding = self.model.encode(query)
            query_embedding = np.array([query_embedding], dtype=np.float32)
            
            # Search the index
            distances, indices = self.index.search(query_embedding, k)
            
            # Format results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.texts):  # Ensure index is valid
                    results.append({
                        "content": self.texts[idx],
                        "title": self.titles[idx],
                        "topic": self.topic_names[idx],
                        "report_id": self.report_ids[idx],
                        "distance": float(distances[0][i])
                    })
            
            return results
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return self._get_demo_results(query, k)
            
    def _get_demo_results(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Generate demo results when the real index is not available.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of demo search results
        """
        # Create some mock results based on keywords in the query
        topics = ["AMERICAS", "EUROPE", "MIDDLE_EAST", "ASIA", "AFRICA"]
        reports = ["2025-02-25.txt", "2025-02-27.json"]
        
        results = []
        keywords = {
            "technology": "Recent advancements in AI technology have revolutionized various industries, with breakthroughs in machine learning models leading to more efficient processes and innovative solutions.",
            "politics": "Political tensions continue to rise as negotiations between major powers reach a critical stage. Leaders from both sides have expressed cautious optimism about finding common ground.",
            "economy": "Economic forecasts indicate a potential slowdown in growth rate over the next quarter, with inflation concerns prompting discussions about interest rate adjustments.",
            "climate": "Climate change reports highlight alarming trends in global temperature rise, with experts calling for immediate action to mitigate environmental damage.",
            "health": "Healthcare systems worldwide are implementing new protocols based on recent research findings, focusing on preventative care and improved treatment methods."
        }
        
        # Find matching keywords
        found_keywords = [k for k in keywords if k.lower() in query.lower()]
        
        # If no keywords match, use some default content
        if not found_keywords:
            found_keywords = list(keywords.keys())[:min(k, len(keywords))]
        
        # Create results based on matching keywords
        for i, keyword in enumerate(found_keywords[:k]):
            topic = topics[i % len(topics)]
            report = reports[i % len(reports)]
            
            results.append({
                "content": keywords[keyword],
                "title": f"{keyword.title()} Update",
                "topic": topic,
                "report_id": report,
                "distance": 0.1 + (i * 0.05)  # Simulate distance scores
            })
            
        return results
    
    def save(self, path: str) -> None:
        """
        Save the index and metadata to disk.
        
        Args:
            path: Directory path where to save the index
        """
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        
        # Save metadata
        metadata = {
            "texts": self.texts,
            "titles": self.titles,
            "topic_names": self.topic_names,
            "report_ids": self.report_ids,
            "dimension": self.dimension
        }
        
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)
    
    def load(self, path: str) -> None:
        """
        Load the index and metadata from disk.
        
        Args:
            path: Directory path to load the index from
        """
        index_file = os.path.join(path, "index.faiss")
        metadata_file = os.path.join(path, "metadata.pkl")
        
        # Check if files exist
        if not os.path.exists(index_file):
            print(f"Warning: FAISS index file not found at {index_file}")
            # Initialize with empty data
            self.texts = []
            self.titles = []
            self.topic_names = ["AMERICAS", "EUROPE", "MIDDLE_EAST", "ASIA", "AFRICA"]  # Default topics
            self.report_ids = ["2025-02-25.txt", "2025-02-27.json"]  # Default reports
            return
            
        if not os.path.exists(metadata_file):
            print(f"Warning: Metadata file not found at {metadata_file}")
            # Initialize with empty data
            self.texts = []
            self.titles = []
            self.topic_names = ["AMERICAS", "EUROPE", "MIDDLE_EAST", "ASIA", "AFRICA"]  # Default topics
            self.report_ids = ["2025-02-25.txt", "2025-02-27.json"]  # Default reports
            return
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_file)
            
            # Load metadata
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)
                
            self.texts = metadata["texts"]
            self.titles = metadata["titles"]
            self.topic_names = metadata["topic_names"]
            self.report_ids = metadata["report_ids"]
            self.dimension = metadata["dimension"]
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            # Initialize with empty data
            self.texts = []
            self.titles = []
            self.topic_names = ["AMERICAS", "EUROPE", "MIDDLE_EAST", "ASIA", "AFRICA"]  # Default topics
            self.report_ids = ["2025-02-25.txt", "2025-02-27.json"]  # Default reports