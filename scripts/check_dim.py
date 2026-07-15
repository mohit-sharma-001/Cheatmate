import sys
sys.path.append("..")  # so we can import from app/
from app.embeddings import embed_text

values = embed_text("test sentence")
print("Dimension:", len(values))