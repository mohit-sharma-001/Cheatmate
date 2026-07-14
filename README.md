# CheatMate Backend MVP

CheatMate is an AI-powered study assistant backend designed to help students learn more efficiently. Using FastAPI, a Retrieval-Augmented Generation (RAG) architecture, and the Google Gemini API, CheatMate allows users to upload study documents (PDFs), extract and chunk the text, compute embeddings, and generate grounded study materials like detailed notes, short summaries, cheat sheets, flashcards, and quizzes.

---

## 🚀 Key Features

* **PDF Text Extraction**: Extracts text content page-by-page from study notes, textbooks, or syllabi.
* **Smart Chunking**: Splits extracted documents into character-based overlapping chunks, filtering out empty or whitespace-only chunks.
* **Vector Embeddings**: Computes text embeddings for each chunk using the **Google Gemini Embedding Model**.
* **Local JSON Vector Store**: Stores chunks and embeddings locally in JSON format, performing semantic search using `numpy` cosine similarity without needing an external database.
* **Grounded Study Material Generation**: Leverages **Google Gemini Flash Lite** to generate responses strictly grounded on the provided document context to prevent hallucinations.
* **Support for 5 Feature Modes**:
  1. `long_notes`: Detailed, comprehensive structured study notes.
  2. `short_notes`: Highly condensed bullet points.
  3. `cheat_sheet`: High-density key terms, formulas, and definitions.
  4. `flashcards`: Question-and-answer pairs returned as raw JSON arrays.
  5. `quiz`: Multiple-choice questions with option lists and correct answers returned as raw JSON arrays.

---

## 🛠️ Tech Stack & Model Information

* **Framework**: FastAPI (Python 3.9+)
* **Dependencies**: `google-generativeai`, `numpy`, `pypdf`, `python-dotenv`, `python-multipart`, `uvicorn`
* **Models**:
  * **Embeddings**: `models/gemini-embedding-001` (specifically chosen for low resource and high reliability)
  * **Generation**: `models/gemini-flash-lite-latest` (optimized for fast grounded queries and compatible with the free tier limits without quota errors)

---

## 📥 Getting Started

### 1. Clone & Navigate to Project Directory
```bash
cd cheatmate
```

### 2. Set Up Virtual Environment
Create and activate a virtual environment to manage dependencies:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the template `.env.example` file and create a `.env` file:
```bash
cp .env.example .env
```
Open `.env` in a text editor and paste your Google Gemini API Key:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 5. Run the Server
Launch the FastAPI development server:
```bash
python app/main.py
```
By default, the server will run on `http://127.0.0.1:8000` with hot-reloading enabled.

---

## 🧪 Running the Tests
To verify all utility modules, routing, embeddings, vector stores, and generation logic without consuming Gemini API tokens, run the mock-based test suite:
```bash
python -m unittest test_app.py
```

---

## 📡 API Usage & cURL Commands

### 🩺 Health Check
Verify the server is running properly:
```bash
curl http://127.0.0.1:8000/health
```
**Response:**
```json
{"status": "ok"}
```

---

### 1. Document Upload (`/upload`)
Upload a PDF document. This endpoint extracts the text, chunks it, generates embeddings, and saves it.
```bash
curl -X POST -F "file=@/path/to/your/study_notes.pdf" http://127.0.0.1:8000/upload
```
**Response:**
```json
{
  "doc_id": "4a12fcb2-8417-48f8-8bb1-ec2e31e5f0ea",
  "num_chunks": 15
}
```
*(Copy the generated `doc_id` to use in the generation endpoints).*

---

### 2. Grounded Material Generation (`/generate`)
Make requests to `/generate` using the retrieved `doc_id` and the specific feature mode.

#### 📝 Feature 1: Long Notes (`long_notes`)
Generates comprehensive, detailed notes:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"doc_id": "YOUR_DOC_ID_HERE", "feature": "long_notes", "instruction": "Explain the main processes in detail"}' \
  http://127.0.0.1:8000/generate
```

#### 📌 Feature 2: Short Notes (`short_notes`)
Generates bullet points and summaries:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"doc_id": "YOUR_DOC_ID_HERE", "feature": "short_notes", "instruction": "Summarize the key facts"}' \
  http://127.0.0.1:8000/generate
```

#### 📑 Feature 3: Cheat Sheet (`cheat_sheet`)
Generates high-density keywords, definitions, and formulas:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"doc_id": "YOUR_DOC_ID_HERE", "feature": "cheat_sheet", "instruction": "Extract definitions and formulas"}' \
  http://127.0.0.1:8000/generate
```

#### 🗂️ Feature 4: Flashcards (`flashcards`)
Generates flashcards returned as a raw JSON array of objects `[{"question": "...", "answer": "..."}]`:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"doc_id": "YOUR_DOC_ID_HERE", "feature": "flashcards", "instruction": "Create flashcards for key terms"}' \
  http://127.0.0.1:8000/generate
```

#### ❓ Feature 5: Quiz (`quiz`)
Generates multiple-choice questions returned as a raw JSON array `[{"question": "...", "options": ["A: ...", "B: ...", "C: ...", "D: ..."], "correct_answer": "..."}]`:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"doc_id": "YOUR_DOC_ID_HERE", "feature": "quiz", "instruction": "Create a 3-question quiz to test comprehension"}' \
  http://127.0.0.1:8000/generate
```

---

## 🔒 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
