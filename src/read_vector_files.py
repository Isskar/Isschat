#!/usr/bin/env python3
"""
A utility script to read and inspect .faiss and .pkl files that are
commonly used for vector databases and embeddings.
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add parent directory to Python path to allow importing from project
sys.path.append(str(Path(__file__).parent.parent))

# Optional imports that might be needed depending on the file content
try:
    import numpy as np
    from langchain_community.vectorstores import FAISS
    import faiss
    from langchain_core.embeddings import Embeddings
except ImportError:
    print("Warning: Some dependencies are missing. Install with:")
    print("pip install langchain langchain-community langchain-core faiss-cpu numpy")


class DummyEmbeddings(Embeddings):
    """A dummy embeddings class for loading FAISS indexes without actual embeddings."""
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return empty embeddings for documents."""
        # Return a dummy embedding vector of dimension 1536 (common for OpenAI embeddings)
        return [[0.0] * 1536 for _ in texts]
        
    def embed_query(self, text: str) -> List[float]:
        """Return an empty embedding for a query."""
        # Return a dummy embedding vector of dimension 1536
        return [0.0] * 1536


def read_pickle_file(file_path: str) -> Any:
    """Read and return the contents of a pickle file."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error reading pickle file: {e}")
        return None


def inspect_pickle_content(data: Any, num_items: int = 5) -> None:
    """
    Print information about the content of a pickle file.
    
    Args:
        data: The loaded pickle data
        num_items: Number of items to display for collections (default: 5)
    """
    print("\n===== Pickle File Content Information =====")
    
    if data is None:
        print("No data could be loaded from the pickle file.")
        return
        
    print(f"Data type: {type(data)}")
    
    if hasattr(data, "__len__"):
        try:
            print(f"Length: {len(data)}")
        except:
            print("Length: Unable to determine")
    
    # Try to determine if it's a dictionary or has attributes
    if isinstance(data, dict):
        print("\nKeys in dictionary:")
        for key in data.keys():
            print(f"  - {key} ({type(data[key])})")
            
        # For FAISS index.pkl files, usually contains document IDs to UUIDs mapping
        # Try to display some of these mappings
        if len(data) > 0:
            print(f"\nShowing first {min(num_items, len(data))} items:")
            
            for i, (key, value) in enumerate(list(data.items())[:num_items]):
                print(f"  {key}: {value}")
                
            if len(data) > num_items:
                print(f"  ... and {len(data) - num_items} more items")
                
    elif hasattr(data, "__dict__"):
        print("\nAttributes:")
        for attr in dir(data):
            if not attr.startswith("_") and not callable(getattr(data, attr)):
                try:
                    attr_value = getattr(data, attr)
                    print(f"  - {attr} ({type(attr_value)})")
                except:
                    print(f"  - {attr} (Unable to access)")
    
    # For numpy arrays or other array-like structures
    if hasattr(data, "shape"):
        try:
            print(f"\nShape: {data.shape}")
        except:
            pass
    
    # If it's a list or other collection, show sample items
    if isinstance(data, (list, tuple)) and len(data) > 0:
        print(f"\nShowing first {min(num_items, len(data))} items:")
        for i, item in enumerate(data[:num_items]):
            print(f"  Item {i}: {type(item)}")
            
            # Try to display some content if it's a simple type
            if isinstance(item, (str, int, float, bool)):
                print(f"    Value: {item}")
            
        if len(data) > num_items:
            print(f"  ... and {len(data) - num_items} more items")


def inspect_faiss_pkl(file_path: str, num_items: int = 5) -> None:
    """
    Directly inspect a .pkl file that's part of a FAISS index.
    This is specifically for the index.pkl file that accompanies a FAISS index.
    
    Args:
        file_path: Path to the pickle file
        num_items: Number of items to display (default: 5)
    """
    try:
        data = read_pickle_file(file_path)
        inspect_pickle_content(data, num_items)
    except Exception as e:
        print(f"Error inspecting FAISS pickle file: {e}")


def read_faiss_file(directory_path: str) -> Optional[FAISS]:
    """
    Read a FAISS index from a directory.
    The directory should contain both index.faiss and index.pkl files.
    
    Args:
        directory_path: Path to the directory containing the FAISS index
    """
    try:
        print(f"Attempting to read FAISS index from: {directory_path}")
        
        # Check if the path exists and is a directory
        if not os.path.isdir(directory_path):
            print(f"Error: {directory_path} is not a directory")
            return None
        
        # Check for index files
        index_faiss_path = os.path.join(directory_path, "index.faiss")
        index_pkl_path = os.path.join(directory_path, "index.pkl")
        
        if not os.path.exists(index_faiss_path):
            print(f"Error: index.faiss not found in {directory_path}")
            if os.path.exists(index_pkl_path):
                print(f"Note: index.pkl exists but index.faiss is missing")
            return None
        
        if not os.path.exists(index_pkl_path):
            print(f"Error: index.pkl not found in {directory_path}")
            return None
            
        # Create a dummy embeddings instance to load the index
        # The real embedding function isn't needed just to inspect the index
        dummy_embeddings = DummyEmbeddings()
        
        # Load the FAISS index
        db = FAISS.load_local(
            directory_path,
            dummy_embeddings,
            allow_dangerous_deserialization=True
        )
        
        return db
        
    except Exception as e:
        print(f"Error reading FAISS directory: {e}")
        import traceback
        traceback.print_exc()
        return None


def inspect_faiss_index(db: FAISS, num_docs: int = 5) -> None:
    """
    Print information about a FAISS index.
    
    Args:
        db: The FAISS index to inspect
        num_docs: Number of documents to display (default: 5)
    """
    if db is None:
        print("No FAISS index could be loaded.")
        return
        
    print("\n===== FAISS Index Information =====")
    
    # Basic information
    print(f"FAISS index type: {type(db)}")
    
    # Try to access the underlying index
    try:
        index = db.index
        print(f"Index type: {type(index)}")
        
        # Get dimensionality if possible
        if hasattr(index, "d"):
            print(f"Vector dimension: {index.d}")
        
        # Get number of vectors if possible
        if hasattr(index, "ntotal"):
            print(f"Number of vectors: {index.ntotal}")
    except Exception as e:
        print(f"Could not access underlying index: {e}")
    
    # Try to access docstore
    try:
        docstore = db.docstore
        print(f"Docstore type: {type(docstore)}")
        
        # Try to get document count
        if hasattr(docstore, "_dict"):
            print(f"Number of documents: {len(docstore._dict)}")
            
            # Show a sample document if available
            if len(docstore._dict) > 0:
                sample_id = next(iter(docstore._dict.keys()))
                sample_doc = docstore._dict[sample_id]
                print("\nSample document:")
                print(f"  ID: {sample_id}")
                print(f"  Type: {type(sample_doc)}")
                
                # Print metadata if available
                if hasattr(sample_doc, "metadata"):
                    print("  Metadata fields:")
                    for key in sample_doc.metadata:
                        print(f"    - {key}")
                
                # Display the documents in a table
                print(f"\n===== First {num_docs} Documents =====")
                
                # Get the document IDs (limited by num_docs)
                doc_ids = list(docstore._dict.keys())[:num_docs]
                
                # Determine max width for each column to format a nice table
                id_width = 8  # Truncated ID width
                title_width = 30
                content_width = 50
                
                # Print table header
                header = f"| {'ID':<{id_width}} | {'Title':<{title_width}} | {'Content (truncated)':<{content_width}} |"
                separator = f"|-{'-'*id_width}-|-{'-'*title_width}-|-{'-'*content_width}-|"
                print(header)
                print(separator)
                
                # Print each document
                for doc_id in doc_ids:
                    doc = docstore._dict[doc_id]
                    
                    # Extract data (safely)
                    truncated_id = str(doc_id)[:id_width] if len(str(doc_id)) > id_width else str(doc_id)
                    title = doc.metadata.get('title', '')[:title_width] if 'title' in doc.metadata else 'N/A'
                    # Truncate content and replace newlines
                    content = doc.page_content.replace('\n', ' ')[:content_width] if hasattr(doc, 'page_content') else 'N/A'
                    
                    # Print row
                    row = f"| {truncated_id:<{id_width}} | {title:<{title_width}} | {content:<{content_width}} |"
                    print(row)
    except Exception as e:
        print(f"Could not access docstore: {e}")


def main():
    parser = argparse.ArgumentParser(description="Read and inspect .faiss and .pkl files")
    parser.add_argument("file_path", help="Path to the file or directory to read")
    parser.add_argument("--type", choices=["auto", "faiss", "pickle", "faiss-pkl"], default="auto",
                        help="Type of file to read: faiss, pickle, faiss-pkl (for index.pkl file only), or auto-detect (default)")
    parser.add_argument("-n", "--num-docs", type=int, default=5,
                        help="Number of documents or items to display (default: 5)")
    
    args = parser.parse_args()
    
    # Determine the file type
    file_type = args.type
    if file_type == "auto":
        if os.path.isdir(args.file_path):
            file_type = "faiss"
        elif args.file_path.endswith(".pkl") or args.file_path.endswith(".pickle"):
            file_type = "pickle"
        else:
            print(f"Could not auto-detect file type for {args.file_path}. Please specify with --type.")
            return
    
    print(f"Reading {file_type} file: {args.file_path}")
    
    # Process according to file type
    if file_type == "faiss":
        db = read_faiss_file(args.file_path)
        inspect_faiss_index(db, args.num_docs)
    elif file_type == "faiss-pkl":
        # For directly examining the index.pkl file
        if not args.file_path.endswith('.pkl'):
            if os.path.isdir(args.file_path):
                args.file_path = os.path.join(args.file_path, 'index.pkl')
            else:
                print(f"Error: {args.file_path} is not a .pkl file or a directory containing index.pkl")
                return
        inspect_faiss_pkl(args.file_path, args.num_docs)
    elif file_type == "pickle":
        data = read_pickle_file(args.file_path)
        inspect_pickle_content(data, args.num_docs)


if __name__ == "__main__":
    main()
    
    # Examples for user reference
    print("\n===== Example Usage =====")
    print("# To view 100 documents in the FAISS index:")
    print("uv run src/read_vector_files.py db --num-docs 100")
    print("\n# To view all documents (if index is not too large):")
    print("uv run src/read_vector_files.py db --num-docs 1000000")
    print("\n# To view just the pickle file (general):")
    print("uv run src/read_vector_files.py db/index.pkl --type pickle")
    print("\n# To view just the FAISS index.pkl file (ID mappings):")
    print("uv run src/read_vector_files.py db --type faiss-pkl --num-docs 20")