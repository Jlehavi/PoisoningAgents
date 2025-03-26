import os
from RAG import HighwayRAG
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")

def count_memories():
    # Initialize RAG with the API key from environment
    rag = HighwayRAG(openai_api_key=api_key)
    
    # Get the vector store
    vector_store = rag.vector_store
    
    # Get the number of documents in the vector store
    num_memories = len(vector_store.docstore._dict)
    
    print(f"Number of memories in the RAG system: {num_memories}")
    
    # Print a few example memories if they exist
    if num_memories > 0:
        print("\nExample memories:")
        for i, (doc_id, doc) in enumerate(list(vector_store.docstore._dict.items())[:5]):  # Show first 5 memories
            print(f"\nMemory {i+1}:")
            print(f"Content: {doc.page_content}")
            print("-" * 50)

if __name__ == "__main__":
    count_memories() 