import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

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
                content=text
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
    Embeds a list of text chunks by calling embed_text individually.
    Adds a 0.1-second delay between calls to prevent rate limiting.
    """
    results = []
    for text in texts:
        embedding = embed_text(text)
        results.append(embedding)
        time.sleep(0.1)
    return results
