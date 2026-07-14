import os
from dotenv import load_dotenv
import google.generativeai as genai
from app import embeddings
from app import vectorstore

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("Warning: GEMINI_API_KEY not found in environment. Please set it in your .env file.")

# Templates for each feature type
TEMPLATES = {
    "long_notes": """You are an expert study assistant. Generate detailed, comprehensive notes based on the provided CONTEXT and user instructions.
Use headings, subheadings, and clear paragraphs. Explain concepts thoroughly and logically.

User Instruction: {user_instruction}""",

    "short_notes": """You are an expert study assistant. Generate short, condensed notes based on the provided CONTEXT and user instructions.
Use bullet points, bold key terms, and summary lists. Keep it highly readable and concise.

User Instruction: {user_instruction}""",

    "cheat_sheet": """You are an expert study assistant. Generate a study cheat sheet based on the provided CONTEXT and user instructions.
Focus on key terms, definitions, formulas, and critical concepts in a highly dense format suitable for a quick reference.

User Instruction: {user_instruction}""",

    "flashcards": """You are an expert study assistant. Generate flashcards based on the provided CONTEXT and user instructions.
Return the output strictly as a JSON array of objects, where each object has exactly two fields: "question" and "answer".
Do not include any introductory or concluding text. Return ONLY the raw JSON array.

User Instruction: {user_instruction}""",

    "quiz": """You are an expert study assistant. Generate a multiple-choice quiz based on the provided CONTEXT and user instructions.
Return the output strictly as a JSON array of objects, where each object has exactly three fields:
- "question" (string)
- "options" (list of strings, containing exactly 4 choices)
- "correct_answer" (string, which must match one of the options exactly)
Do not include any introductory or concluding text. Return ONLY the raw JSON array.

User Instruction: {user_instruction}"""
}

def _strip_markdown_code_fences(text: str) -> str:
    """
    Strips markdown code fences (e.g. ```json ... ```) from a model's response.
    This ensures that the output is a valid raw JSON string.
    """
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        else:
            text = text[3:]
            
    if text.endswith("```"):
        text = text[:-3]
        
    return text.strip()

def generate_notes(doc_id: str, feature: str, user_instruction: str) -> str:
    """
    Retrieves the top 5 relevant document chunks for the user instruction, 
    constructs a feature-specific prompt, and calls Gemini (gemini-flash-lite-latest) 
    to generate the final grounded study materials.
    """
    # 1. Validate doc exists
    if not vectorstore.doc_exists(doc_id):
        raise ValueError(f"Document with ID {doc_id} does not exist.")
        
    # 2. Embed user instruction
    query_embedding = embeddings.embed_text(user_instruction)
    
    # 3. Retrieve relevant chunks (top 5)
    relevant_chunks = vectorstore.search(doc_id, query_embedding, top_k=5)
    if not relevant_chunks:
        return "No relevant context found in the document to answer the request."
        
    # Combine chunks to form context block
    context = "\n---\n".join(relevant_chunks)
    
    # 4. Retrieve and format prompt template
    template = TEMPLATES.get(feature)
    if not template:
        valid_features = list(TEMPLATES.keys())
        raise ValueError(f"Unknown feature: '{feature}'. Must be one of: {valid_features}")
        
    feature_prompt = template.format(user_instruction=user_instruction)
    
    # 5. Build full prompt with strict grounding guidelines
    full_prompt = f"""CONTEXT:
{context}

---

GROUNDING RULES:
Only use the provided CONTEXT. If the context doesn't contain enough information, say so — do not make things up.

---

INSTRUCTIONS:
{feature_prompt}
"""

    # 6. Generate output using gemini-2.0-flash
    model = genai.GenerativeModel("gemini-flash-lite-latest")
    response = model.generate_content(full_prompt)
    text_response = response.text
    
    # 7. Post-process to strip markdown code fences for JSON outputs
    if feature in ("flashcards", "quiz"):
        text_response = _strip_markdown_code_fences(text_response)
        
    return text_response
