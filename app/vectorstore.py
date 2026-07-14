import os
import json
import numpy as np

# Define storage directory path relative to this file (app/storage)
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")

def _get_doc_path(doc_id: str) -> str:
    """Helper to get the JSON file path for a given doc_id."""
    return os.path.join(STORAGE_DIR, f"{doc_id}.json")

def save_chunks(doc_id: str, chunks: list[str], embeddings: list[list[float]]):
    """
    Saves document chunks and their corresponding embeddings into a local JSON file.
    Creates the storage directory if it does not exist.
    """
    os.makedirs(STORAGE_DIR, exist_ok=True)
    file_path = _get_doc_path(doc_id)
    
    data = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        data.append({
            "doc_id": doc_id,
            "chunk_id": idx,
            "text": chunk,
            "embedding": embedding
        })
        
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def search(doc_id: str, query_embedding: list[float], top_k: int = 5) -> list[str]:
    """
    Searches the stored chunks of a document using cosine similarity against the query embedding.
    Returns the top_k most similar chunk texts.
    """
    file_path = _get_doc_path(doc_id)
    if not os.path.exists(file_path):
        print(f"Error: Document with ID {doc_id} does not exist.")
        return []
        
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if not data:
        return []
        
    # Extract texts and embeddings
    texts = [item["text"] for item in data]
    embeddings_list = [item["embedding"] for item in data]
    
    # Convert to numpy arrays for calculation
    embeddings_arr = np.array(embeddings_list)
    query_arr = np.array(query_embedding)
    
    # Calculate cosine similarity: A . B / (||A|| * ||B||)
    dot_products = np.dot(embeddings_arr, query_arr)
    embedding_norms = np.linalg.norm(embeddings_arr, axis=1)
    query_norm = np.linalg.norm(query_arr)
    
    # Handle possible division by zero
    norms = embedding_norms * query_norm
    similarities = np.zeros_like(dot_products)
    non_zero_indices = norms > 0
    similarities[non_zero_indices] = dot_products[non_zero_indices] / norms[non_zero_indices]
    
    # Sort indices by similarity in descending order
    sorted_indices = np.argsort(similarities)[::-1]
    
    # Get top_k indices
    top_indices = sorted_indices[:top_k]
    
    # Return top_k chunk texts
    return [texts[idx] for idx in top_indices]

def doc_exists(doc_id: str) -> bool:
    """
    Checks if a document JSON store exists locally.
    """
    return os.path.exists(_get_doc_path(doc_id))
