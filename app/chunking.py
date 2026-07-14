from typing import List

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Chunks text into character-based overlapping segments.
    Skips appending a chunk if it is empty or whitespace-only after stripping.
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Skip if empty or whitespace-only after stripping
        if chunk.strip():
            chunks.append(chunk)
            
        step = max(1, chunk_size - overlap)
        start += step
        if start >= text_len or end >= text_len:
            break
            
    return chunks
