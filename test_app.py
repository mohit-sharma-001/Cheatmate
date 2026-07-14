import os
import json
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Setup dummy environment key before importing app components
os.environ["GEMINI_API_KEY"] = "dummy_key_for_testing"

from app.chunking import chunk_text
from app.pdf_utils import extract_text_from_pdf
from app.embeddings import embed_text, embed_batch
from app import vectorstore
from app.generation import generate_notes, _strip_markdown_code_fences
from app.main import app

class TestChunking(unittest.TestCase):
    def test_chunk_text_basic(self):
        text = "Hello world! This is a simple test text to verify chunking behavior."
        # chunk_size=20, overlap=5
        # "Hello world! This is" (20 chars) -> step = 15
        # start 15: " is a simple test te" (20 chars) -> step = 15
        # start 30: "st text to verify ch" ...
        chunks = chunk_text(text, chunk_size=20, overlap=5)
        self.assertTrue(len(chunks) > 0)
        for c in chunks:
            self.assertTrue(len(c) <= 20)

    def test_chunk_text_whitespace_skip(self):
        # Text with some blocks that are whitespace only
        # We expect only non-whitespace chunks to be returned
        text = "Hello.                   "
        chunks = chunk_text(text, chunk_size=10, overlap=2)
        # First chunk: "Hello.    " (has characters, strip() is "Hello.")
        # Second chunk: "       " (whitespace only, strip() is empty -> skipped)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].strip(), "Hello.")

class TestPDFUtils(unittest.TestCase):
    @patch("app.pdf_utils.PdfReader")
    def test_extract_text_from_pdf(self, mock_pdf_reader):
        # Mock pages
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 content"
        
        mock_reader_instance = MagicMock()
        mock_reader_instance.pages = [mock_page1, mock_page2]
        mock_pdf_reader.return_value = mock_reader_instance
        
        result = extract_text_from_pdf(b"dummy pdf bytes")
        self.assertIn("Page 1 content", result)
        self.assertIn("Page 2 content", result)

class TestEmbeddings(unittest.TestCase):
    @patch("google.generativeai.embed_content")
    def test_embed_text_success(self, mock_embed_content):
        # Mock returning an object style first, then dict style
        mock_embed_content.return_value = {"embedding": {"values": [0.1, 0.2, 0.3]}}
        emb = embed_text("hello")
        self.assertEqual(emb, [0.1, 0.2, 0.3])

    @patch("google.generativeai.embed_content")
    def test_embed_text_retry(self, mock_embed_content):
        # Mock failed calls twice, then success on third
        mock_embed_content.side_effect = [
            Exception("API Rate Limit"),
            Exception("Transient Network Error"),
            {"embedding": {"values": [0.5, 0.6, 0.7]}}
        ]
        emb = embed_text("retry test")
        self.assertEqual(emb, [0.5, 0.6, 0.7])
        self.assertEqual(mock_embed_content.call_count, 3)

    @patch("app.embeddings.embed_text")
    @patch("time.sleep") # Mock sleep to speed up tests
    def test_embed_batch(self, mock_sleep, mock_embed_text):
        mock_embed_text.return_value = [0.9, 0.8]
        batch = ["text1", "text2", "text3"]
        embeddings = embed_batch(batch)
        self.assertEqual(len(embeddings), 3)
        self.assertEqual(embeddings[0], [0.9, 0.8])
        self.assertEqual(mock_embed_text.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 3)

class TestVectorStore(unittest.TestCase):
    def setUp(self):
        # Use a separate test doc_id
        self.test_doc_id = "test-doc-123"
        self.test_file_path = vectorstore._get_doc_path(self.test_doc_id)
        # Ensure clean state
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

    def tearDown(self):
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

    def test_save_and_search_and_exists(self):
        chunks = [
            "FastAPI is a modern, fast (high-performance) web framework for building APIs with Python.",
            "Retrieval-Augmented Generation (RAG) is a technique for grounding large language models.",
            "Numpy is a library for the Python programming language, adding support for large, multi-dimensional arrays."
        ]
        # Embeddings as simple unit vectors to test cosine similarity logic
        embeddings = [
            [1.0, 0.0, 0.0], # FastAPI
            [0.0, 1.0, 0.0], # RAG
            [0.0, 0.0, 1.0]  # Numpy
        ]
        
        # Save
        vectorstore.save_chunks(self.test_doc_id, chunks, embeddings)
        self.assertTrue(vectorstore.doc_exists(self.test_doc_id))
        
        # Search query matching first document (FastAPI)
        query = [0.9, 0.1, 0.0]
        results = vectorstore.search(self.test_doc_id, query, top_k=2)
        
        self.assertEqual(len(results), 2)
        # The closest should be the FastAPI chunk
        self.assertEqual(results[0], chunks[0])

class TestGeneration(unittest.TestCase):
    def test_strip_markdown_code_fences(self):
        # Test cleaning json code fences
        input_text = "```json\n[\n  {\"question\": \"Q?\", \"answer\": \"A\"}\n]\n```"
        output = _strip_markdown_code_fences(input_text)
        self.assertEqual(output, "[\n  {\"question\": \"Q?\", \"answer\": \"A\"}\n]")

        # Test cleaning raw code fences
        input_text2 = "```\nhello\n```"
        output2 = _strip_markdown_code_fences(input_text2)
        self.assertEqual(output2, "hello")

    @patch("app.generation.genai.GenerativeModel")
    @patch("app.embeddings.embed_text")
    @patch("app.vectorstore.search")
    @patch("app.vectorstore.doc_exists")
    def test_generate_notes_grounding_and_fences(self, mock_doc_exists, mock_search, mock_embed, mock_gen_model):
        mock_doc_exists.return_value = True
        mock_embed.return_value = [0.1, 0.2]
        mock_search.return_value = ["This is context 1", "This is context 2"]
        
        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.text = "```json\n[{\"question\": \"Q\", \"answer\": \"A\"}]\n```"
        
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = mock_response
        mock_gen_model.return_value = mock_model_instance
        
        # Generate flashcards
        result = generate_notes(
            doc_id="test-doc",
            feature="flashcards",
            user_instruction="Generate 1 flashcard"
        )
        
        # Verify markdown fences stripped
        self.assertEqual(result, "[{\"question\": \"Q\", \"answer\": \"A\"}]")
        
        # Verify search was called
        mock_search.assert_called_once_with("test-doc", [0.1, 0.2], top_k=5)
        
        # Verify prompt had context and instruction
        call_args = mock_model_instance.generate_content.call_args[0][0]
        self.assertIn("This is context 1", call_args)
        self.assertIn("GROUNDING RULES:", call_args)
        self.assertIn("Only use the provided CONTEXT", call_args)

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    @patch("app.main.extract_text_from_pdf")
    @patch("app.main.embed_batch")
    @patch("app.main.save_chunks")
    def test_upload_endpoint(self, mock_save, mock_embed_batch, mock_extract):
        mock_extract.return_value = "Extracted PDF contents."
        mock_embed_batch.return_value = [[0.1, 0.2]]
        
        pdf_content = b"%PDF-1.4 mock pdf content"
        files = {"file": ("test.pdf", pdf_content, "application/pdf")}
        
        response = self.client.post("/upload", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("doc_id", data)
        self.assertEqual(data["num_chunks"], 1)

    @patch("app.main.doc_exists")
    @patch("app.main.generate_notes")
    def test_generate_endpoint(self, mock_generate_notes, mock_doc_exists):
        mock_doc_exists.return_value = True
        mock_generate_notes.return_value = "Detailed notes content"
        
        payload = {
            "doc_id": "existing-doc-uuid",
            "feature": "long_notes",
            "instruction": "Explain everything"
        }
        
        response = self.client.post("/generate", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"result": "Detailed notes content"})

if __name__ == "__main__":
    unittest.main()
