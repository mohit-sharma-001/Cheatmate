import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from app.pdf_utils import extract_text_from_pdf
from app.chunking import chunk_text
from app.embeddings import embed_batch
from app.vectorstore import save_chunks, doc_exists
from app.generation import generate_notes

# Initialize FastAPI app
app = FastAPI(
    title="CheatMate Backend MVP",
    description="FastAPI + RAG study assistant backend with Gemini API",
    version="1.0.0"
)

# Enable CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema for generation
class GenerateRequest(BaseModel):
    doc_id: str
    feature: str
    instruction: str

@app.get("/health")
def health_check():
    """
    Simple health check endpoint to verify backend service status.
    """
    return {"status": "ok"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF upload, extracts the text, chunks it, generates embeddings for each chunk,
    saves the data in the local vector store, and returns a unique document ID.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    try:
        # Read the file content
        file_bytes = await file.read()
        
        # 1. Extract text from PDF
        text = extract_text_from_pdf(file_bytes)
        if not text.strip():
            raise HTTPException(status_code=400, detail="The uploaded PDF has no extractable text.")
            
        # 2. Chunk text
        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="PDF content resulted in no valid text chunks.")
            
        # 3. Generate doc_id
        doc_id = str(uuid.uuid4())
        
        # 4. Embed chunks
        embeddings_list = embed_batch(chunks)
        
        # 5. Save chunk embeddings to local store
        save_chunks(doc_id, chunks, embeddings_list)
        
        return {
            "doc_id": doc_id,
            "num_chunks": len(chunks)
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF upload: {str(e)}")

@app.post("/generate")
def generate_study_material(payload: GenerateRequest):
    """
    Retrieves relevant context for the doc_id using the user instruction, 
    and generates study materials based on the chosen feature template.
    """
    # 1. Verify doc exists
    if not doc_exists(payload.doc_id):
        raise HTTPException(status_code=404, detail=f"Document with ID {payload.doc_id} not found.")
        
    try:
        # 2. Generate grounded study notes/flashcards/quiz
        result = generate_notes(
            doc_id=payload.doc_id,
            feature=payload.feature,
            user_instruction=payload.instruction
        )
        return {"result": result}
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate material: {str(e)}")

if __name__ == "__main__":
    # Run uvicorn on port 8000 with reload=True
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
