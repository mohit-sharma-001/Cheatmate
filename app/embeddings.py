import os
import time
import google.generativeai as genai
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env
load_dotenv()

# Configure the Gemini API client
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("Warning: GEMINI_API_KEY not found in environment. Please set it in your .env file.")

def embed_text(text: str) -> list[float]:
    """
    Embeds a single text chunk using the Google Gemini gemini-embedding-001 model.
    Retries up to 3 times on failure.
    """
    max_attempts = 3
    delay = 1.0  # seconds between retries
    
    for attempt in range(max_attempts):
        try:
            response = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text,
                output_dimensionality=768
            )
            # Handle different SDK response formats safely
            embedding = response["embedding"] if isinstance(response, dict) else response.embedding
            # Some model versions nest the values under "values", others return the list directly
            if isinstance(embedding, dict):
                return embedding["values"]
            elif hasattr(embedding, "values"):
                return embedding.values
            else:
                return embedding
        except Exception as e:
            print(f"Embedding attempt {attempt + 1} failed: {e}")
            if attempt == max_attempts - 1:
                raise e
            time.sleep(delay * (attempt + 1))  # linear backoff
            
    raise RuntimeError("Failed to embed text after retries")

def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embeds a list of text chunks in parallel using ThreadPoolExecutor.
    
    ThreadPoolExecutor is chosen here because calls to the Gemini API are network
    I/O-bound rather than CPU-bound. Utilizing threads allows multiple network
    requests to run concurrently, yielding a significant speedup without being
    limited by Python's Global Interpreter Lock (GIL).
    
    Uses max_workers=5 to balance performance and Gemini API rate limits.
    """
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(embed_text, texts))
    return results
