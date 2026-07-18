import os
import json
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Setup dummy environment key before importing app components
os.environ["GEMINI_API_KEY"] = "dummy_key_for_testing"
os.environ["SUPABASE_DB_URL"] = "postgresql://postgres:dummy@localhost:5432/postgres"

from app.chunking import chunk_text
from app.pdf_utils import extract_text_from_pdf
from app.embeddings import embed_text, embed_batch
from app import vectorstore
from app.generation import generate_notes, _strip_markdown_code_fences
from app.main import app
from app import chat
from app.extract import extract_text_from_docx, extract_text_from_txt, extract_text_from_image



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

class TestExtraction(unittest.TestCase):
    @patch("app.extract.Document")
    def test_extract_text_from_docx(self, mock_document):
        # Mock paragraph structure
        mock_p1 = MagicMock()
        mock_p1.text = "Hello world paragraph."
        mock_p2 = MagicMock()
        mock_p2.text = "Second paragraph."
        
        # Mock table structure
        mock_cell1 = MagicMock()
        mock_cell1.text = "Cell1"
        mock_cell2 = MagicMock()
        mock_cell2.text = "Cell2"
        mock_row1 = MagicMock()
        mock_row1.cells = [mock_cell1, mock_cell2]
        mock_table1 = MagicMock()
        mock_table1.rows = [mock_row1]
        
        mock_doc_instance = MagicMock()
        mock_doc_instance.paragraphs = [mock_p1, mock_p2]
        mock_doc_instance.tables = [mock_table1]
        mock_document.return_value = mock_doc_instance
        
        result = extract_text_from_docx(b"mock docx bytes")
        self.assertIn("Hello world paragraph.", result)
        self.assertIn("Second paragraph.", result)
        self.assertIn("Cell1 | Cell2", result)

    def test_extract_text_from_txt(self):
        # Test valid utf-8 decoding
        result = extract_text_from_txt(b"Hello world from text.")
        self.assertEqual(result, "Hello world from text.")
        
        # Test invalid bytes fallback (errors="ignore")
        result_corrupted = extract_text_from_txt(b"Hello \xff world.")
        self.assertEqual(result_corrupted, "Hello  world.")

    @patch("app.extract.genai.GenerativeModel")
    def test_extract_text_from_image(self, mock_gen_model):
        mock_response = MagicMock()
        mock_response.text = "Text from image transcription"
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = mock_response
        mock_gen_model.return_value = mock_model_instance
        
        result = extract_text_from_image(b"fake image bytes", "image/png")
        self.assertEqual(result, "Text from image transcription")
        
        # Verify model and arguments
        mock_gen_model.assert_called_once_with("models/gemini-flash-lite-latest")
        call_args = mock_model_instance.generate_content.call_args[0][0]
        self.assertEqual(call_args[0]["mime_type"], "image/png")
        self.assertEqual(call_args[0]["data"], b"fake image bytes")
        self.assertIn("Extract all text", call_args[1])


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
        mock_sleep.assert_not_called()

class TestVectorStore(unittest.TestCase):
    @patch("psycopg2.connect")
    @patch("app.vectorstore.execute_values")
    def test_save_chunks(self, mock_execute_values, mock_connect):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.__enter__.return_value = mock_cur
        mock_conn.cursor.return_value = mock_cur
        mock_connect.return_value = mock_conn
        
        chunks = ["chunk1", "chunk2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        
        vectorstore.save_chunks("test-doc", chunks, embeddings)
        
        mock_connect.assert_called_once_with(vectorstore.DB_URL)
        mock_execute_values.assert_called_once()
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("psycopg2.connect")
    def test_search(self, mock_connect):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.__enter__.return_value = mock_cur
        mock_cur.fetchall.return_value = [("result text 1",), ("result text 2",)]
        mock_conn.cursor.return_value = mock_cur
        mock_connect.return_value = mock_conn
        
        results = vectorstore.search(["test-doc"], [0.1, 0.2], top_k=2)
        
        self.assertEqual(results, ["result text 1", "result text 2"])
        self.assertTrue(mock_cur.execute.called)
        mock_conn.close.assert_called_once()

    @patch("psycopg2.connect")
    def test_doc_exists(self, mock_connect):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.__enter__.return_value = mock_cur
        mock_cur.fetchone.return_value = (True,)
        mock_conn.cursor.return_value = mock_cur
        mock_connect.return_value = mock_conn
        
        exists = vectorstore.doc_exists("test-doc")
        self.assertTrue(exists)
        mock_conn.close.assert_called_once()

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
            doc_ids=["test-doc"],
            feature="flashcards",
            user_instruction="Generate 1 flashcard"
        )
        
        # Verify markdown fences stripped
        self.assertEqual(result, "[{\"question\": \"Q\", \"answer\": \"A\"}]")
        
        # Verify search was called
        mock_search.assert_called_once_with(["test-doc"], [0.1, 0.2], top_k=5)
        
        # Verify prompt had context and instruction
        call_args = mock_model_instance.generate_content.call_args[0][0]
        self.assertIn("This is context 1", call_args)
        self.assertIn("GROUNDING RULES:", call_args)
        self.assertIn("Only use the provided CONTEXT", call_args)

class TestChatModule(unittest.TestCase):
    @patch("app.chat._get_connection")
    def test_get_or_create_conversation_existing(self, mock_get_conn):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = (True,)
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_get_conn.return_value = mock_conn

        res = chat.get_or_create_conversation("existing-uuid")
        self.assertEqual(res, "existing-uuid")
        mock_cur.execute.assert_called_once_with(
            "SELECT EXISTS(SELECT 1 FROM conversations WHERE id = %s)", 
            ("existing-uuid",)
        )

    @patch("app.chat._get_connection")
    def test_get_or_create_conversation_new(self, mock_get_conn):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        # EXISTS query returns False, INSERT DEFAULT VALUES query returns new ID
        mock_cur.fetchone.side_effect = [(False,), ("new-uuid-123",)]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_get_conn.return_value = mock_conn

        res = chat.get_or_create_conversation("non-existent-uuid")
        self.assertEqual(res, "new-uuid-123")
        self.assertEqual(mock_cur.execute.call_count, 2)
        mock_conn.commit.assert_called_once()

    @patch("app.chat._get_connection")
    def test_get_recent_messages(self, mock_get_conn):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        # Mock recent messages returned from DB (newest first, which is what query fetches)
        mock_cur.fetchall.return_value = [
            ("assistant", "response 1"),
            ("user", "message 1")
        ]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_get_conn.return_value = mock_conn

        res = chat.get_recent_messages("some-uuid", limit=6)
        # Verify chronological order (reversed in python to user message first)
        self.assertEqual(res, [
            {"role": "user", "content": "message 1"},
            {"role": "assistant", "content": "response 1"}
        ])

    @patch("app.chat._get_connection")
    def test_save_message(self, mock_get_conn):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_get_conn.return_value = mock_conn

        chat.save_message("some-uuid", "user", "hello")
        mock_cur.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

    @patch("app.chat.save_message")
    @patch("app.chat.get_recent_messages")
    @patch("app.chat.get_or_create_conversation")
    @patch("app.chat.genai.GenerativeModel")
    @patch("app.embeddings.embed_text")
    @patch("app.vectorstore.search")
    @patch("app.vectorstore.doc_exists")
    def test_chat_without_doc(self, mock_doc_exists, mock_search, mock_embed, mock_gen_model, 
                              mock_get_or_create, mock_get_recent, mock_save_msg):
        mock_doc_exists.return_value = False
        mock_get_or_create.return_value = "verified-uuid"
        mock_get_recent.return_value = [{"role": "user", "content": "hello"}]
        
        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.text = "Hello! I am CheatMate."
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = mock_response
        mock_gen_model.return_value = mock_model_instance
        
        response, returned_id = chat.chat("test-conv-2", "Hello study assistant", doc_ids=None)
        
        self.assertEqual(response, "Hello! I am CheatMate.")
        self.assertEqual(returned_id, "verified-uuid")
        
        # Verify messages saved
        self.assertEqual(mock_save_msg.call_count, 2)
        mock_save_msg.assert_any_call("verified-uuid", "user", "Hello study assistant")
        mock_save_msg.assert_any_call("verified-uuid", "assistant", "Hello! I am CheatMate.")
        
        # Verify GenerativeModel was initialized with correct system instruction
        mock_gen_model.assert_called_once_with(
            model_name="models/gemini-flash-lite-latest",
            system_instruction=chat.SYSTEM_INSTRUCTION
        )
        # Verify search was not called
        mock_search.assert_not_called()

    @patch("app.chat.save_message")
    @patch("app.chat.get_recent_messages")
    @patch("app.chat.get_or_create_conversation")
    @patch("app.chat.genai.GenerativeModel")
    @patch("app.embeddings.embed_text")
    @patch("app.vectorstore.search")
    @patch("app.vectorstore.doc_exists")
    def test_chat_with_doc(self, mock_doc_exists, mock_search, mock_embed, mock_gen_model,
                           mock_get_or_create, mock_get_recent, mock_save_msg):
        mock_doc_exists.return_value = True
        mock_get_or_create.return_value = "verified-uuid"
        mock_get_recent.return_value = []
        mock_embed.return_value = [0.1, 0.2]
        mock_search.return_value = ["Grounded chunk 1", "Grounded chunk 2"]
        
        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.text = "Grounded response."
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = mock_response
        mock_gen_model.return_value = mock_model_instance
        
        response, returned_id = chat.chat("test-conv-3", "Explain photosynthesis", doc_ids=["my-doc"])
        
        self.assertEqual(response, "Grounded response.")
        self.assertEqual(returned_id, "verified-uuid")
        
        # Verify vector store query was performed
        mock_doc_exists.assert_called_once_with("my-doc")
        mock_embed.assert_called_once_with("Explain photosynthesis")
        mock_search.assert_called_once_with(["my-doc"], [0.1, 0.2], top_k=5)
        
        # Verify prompt details
        call_args = mock_model_instance.generate_content.call_args[0][0]
        self.assertIn("CONTEXT:\nGrounded chunk 1\n---\nGrounded chunk 2", call_args)
        self.assertIn("User: Explain photosynthesis", call_args)

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    @patch("app.main.usage.check_and_add_document_to_conversation")
    @patch("app.main.usage.check_and_increment_upload")
    @patch("app.main.extract_text_from_pdf")
    @patch("app.main.embed_batch")
    @patch("app.main.save_chunks")
    def test_upload_endpoint(self, mock_save, mock_embed_batch, mock_extract, mock_check_upload, mock_add_doc):
        mock_extract.return_value = "Extracted PDF contents."
        mock_embed_batch.return_value = [[0.1, 0.2]]
        
        pdf_content = b"%PDF-1.4 mock pdf content"
        files = {"file": ("test.pdf", pdf_content, "application/pdf")}
        
        response = self.client.post("/upload", files=files, headers={"X-Guest-Id": "test-guest"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("doc_id", data)
        self.assertEqual(data["num_chunks"], 1)

    @patch("app.main.usage.check_and_add_document_to_conversation")
    @patch("app.main.usage.check_and_increment_upload")
    @patch("app.main.extract_text_from_docx")
    @patch("app.main.embed_batch")
    @patch("app.main.save_chunks")
    def test_upload_endpoint_docx(self, mock_save, mock_embed_batch, mock_extract_docx, mock_check_upload, mock_add_doc):
        mock_extract_docx.return_value = "Extracted DOCX content."
        mock_embed_batch.return_value = [[0.1, 0.2]]
        
        files = {"file": ("test.docx", b"dummy docx bytes", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        response = self.client.post("/upload", files=files, headers={"X-Guest-Id": "test-guest"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("doc_id", data)
        mock_extract_docx.assert_called_once()

    @patch("app.main.usage.check_and_add_document_to_conversation")
    @patch("app.main.usage.check_and_increment_upload")
    @patch("app.main.extract_text_from_txt")
    @patch("app.main.embed_batch")
    @patch("app.main.save_chunks")
    def test_upload_endpoint_txt(self, mock_save, mock_embed_batch, mock_extract_txt, mock_check_upload, mock_add_doc):
        mock_extract_txt.return_value = "Extracted TXT content."
        mock_embed_batch.return_value = [[0.1, 0.2]]
        
        files = {"file": ("test.txt", b"dummy txt content", "text/plain")}
        response = self.client.post("/upload", files=files, headers={"X-Guest-Id": "test-guest"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("doc_id", data)
        mock_extract_txt.assert_called_once()

    @patch("app.main.usage.check_and_add_document_to_conversation")
    @patch("app.main.usage.check_and_increment_upload")
    @patch("app.main.extract_text_from_image")
    @patch("app.main.embed_batch")
    @patch("app.main.save_chunks")
    def test_upload_endpoint_image(self, mock_save, mock_embed_batch, mock_extract_img, mock_check_upload, mock_add_doc):
        mock_extract_img.return_value = "Extracted image text content."
        mock_embed_batch.return_value = [[0.1, 0.2]]
        
        files = {"file": ("test.png", b"fake image bytes", "image/png")}
        response = self.client.post("/upload", files=files, headers={"X-Guest-Id": "test-guest"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("doc_id", data)
        mock_extract_img.assert_called_once_with(b"fake image bytes", "image/png")

    def test_upload_endpoint_unsupported(self):
        files = {"file": ("test.exe", b"binary executable", "application/octet-stream")}
        response = self.client.post("/upload", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Unsupported file type. Supported formats: PDF, DOCX, TXT, JPG, PNG")


    @patch("app.main.doc_exists")
    @patch("app.main.generate_notes")
    def test_generate_endpoint(self, mock_generate_notes, mock_doc_exists):
        mock_doc_exists.return_value = True
        mock_generate_notes.return_value = "Detailed notes content"
        
        payload = {
            "doc_ids": ["existing-doc-uuid"],
            "feature": "long_notes",
            "instruction": "Explain everything"
        }
        
        response = self.client.post("/generate", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"result": "Detailed notes content"})

    @patch("app.main.chat.chat")
    def test_chat_endpoint_new_conversation(self, mock_chat):
        mock_chat.return_value = ("Response from model", "generated-uuid-5678")
        
        payload = {
            "message": "Hello study assistant"
        }
        
        response = self.client.post("/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("conversation_id", data)
        self.assertEqual(data["response"], "Response from model")
        
        # Verify conversation ID returned from mock_chat is used
        self.assertEqual(data["conversation_id"], "generated-uuid-5678")

    @patch("app.main.chat.chat")
    def test_chat_endpoint_existing_conversation(self, mock_chat):
        mock_chat.return_value = ("Response with history", "existing-uuid-1234")
        
        payload = {
            "conversation_id": "existing-uuid-1234",
            "message": "Continue discussing",
            "doc_ids": ["doc-uuid-5678"]
        }
        
        response = self.client.post("/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "conversation_id": "existing-uuid-1234",
            "response": "Response with history"
        })
        mock_chat.assert_called_once_with(conversation_id="existing-uuid-1234", message="Continue discussing", doc_ids=["doc-uuid-5678"], user_id=None)

class TestAuth(unittest.TestCase):
    @patch("app.auth.jwks_client")
    @patch("jwt.decode")
    def test_get_user_id_valid_token(self, mock_jwt_decode, mock_jwks_client):
        from app import auth
        mock_signing_key = MagicMock()
        mock_signing_key.key = "public_key"
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwt_decode.return_value = {"sub": "user-uuid-123"}

        res = auth.get_user_id("Bearer valid.jwt.token")
        self.assertEqual(res, "user-uuid-123")
        mock_jwt_decode.assert_called_once()

    def test_get_user_id_missing_or_invalid_header(self):
        from app import auth
        self.assertIsNone(auth.get_user_id(None))
        self.assertIsNone(auth.get_user_id("InvalidHeader"))
        self.assertIsNone(auth.get_user_id("Bearer"))

    @patch("app.auth.jwks_client")
    @patch("jwt.decode")
    def test_get_user_id_expired(self, mock_jwt_decode, mock_jwks_client):
        from app import auth
        import jwt
        mock_signing_key = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwt_decode.side_effect = jwt.ExpiredSignatureError("Token expired")

        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            auth.get_user_id("Bearer expired.jwt.token")
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Invalid or expired session", ctx.exception.detail)

    @patch("app.auth.get_user_id")
    def test_get_identifier_logged_in(self, mock_get_user_id):
        from app import auth
        mock_get_user_id.return_value = "user-uuid-123"
        res = auth.get_identifier("Bearer some-token", "guest-id")
        self.assertEqual(res, "user-uuid-123")

    @patch("app.auth.get_user_id")
    def test_get_identifier_guest(self, mock_get_user_id):
        from app import auth
        mock_get_user_id.return_value = None
        res = auth.get_identifier(None, "guest-id-123")
        self.assertEqual(res, "guest-id-123")

    @patch("app.auth.get_user_id")
    def test_get_identifier_missing(self, mock_get_user_id):
        from app import auth
        mock_get_user_id.return_value = None
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            auth.get_identifier(None, None)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Missing guest identifier", ctx.exception.detail)

class TestUsage(unittest.TestCase):
    @patch("app.usage._get_connection")
    def test_check_increment_upload_under_limit(self, mock_get_conn):
        from app import usage
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = (1,)
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_get_conn.return_value = mock_conn

        usage.check_and_increment_upload("user-123", is_guest=False)
        self.assertEqual(mock_cur.execute.call_count, 2)
        mock_conn.commit.assert_called_once()

    @patch("app.usage._get_connection")
    def test_check_increment_upload_guest_at_limit(self, mock_get_conn):
        from app import usage
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = (2,)
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_get_conn.return_value = mock_conn

        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            usage.check_and_increment_upload("guest-123", is_guest=True)
        self.assertEqual(ctx.exception.status_code, 429)
        self.assertIn("higher limit", ctx.exception.detail)

    @patch("app.usage._get_connection")
    def test_check_increment_upload_user_at_limit(self, mock_get_conn):
        from app import usage
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = (5,)
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_get_conn.return_value = mock_conn

        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            usage.check_and_increment_upload("user-123", is_guest=False)
        self.assertEqual(ctx.exception.status_code, 429)
        self.assertNotIn("higher limit", ctx.exception.detail)

    def test_check_add_document_to_conversation_none(self):
        from app import usage
        # should do nothing if conversation_id is None
        usage.check_and_add_document_to_conversation(None, "doc-123")

    @patch("app.usage._get_connection")
    def test_check_add_document_to_conversation_under_limit(self, mock_get_conn):
        from app import usage
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = (3,)
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_get_conn.return_value = mock_conn

        usage.check_and_add_document_to_conversation("conv-123", "doc-123")
        self.assertEqual(mock_cur.execute.call_count, 2)
        mock_conn.commit.assert_called_once()

    @patch("app.usage._get_connection")
    def test_check_add_document_to_conversation_at_limit(self, mock_get_conn):
        from app import usage
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = (5,)
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_get_conn.return_value = mock_conn

        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            usage.check_and_add_document_to_conversation("conv-123", "doc-123")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("maximum of 5 documents", ctx.exception.detail)

class TestChatUpdates(unittest.TestCase):
    @patch("app.chat._get_connection")
    def test_get_or_create_conversation_new_with_user(self, mock_get_conn):
        from app import chat
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.side_effect = [(False,), ("new-uuid-456",)]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_get_conn.return_value = mock_conn

        res = chat.get_or_create_conversation("non-existent-uuid", user_id="user-uuid-123")
        self.assertEqual(res, "new-uuid-456")
        self.assertEqual(mock_cur.execute.call_count, 3)
        mock_cur.execute.assert_any_call("UPDATE conversations SET user_id = %s WHERE id = %s", ("user-uuid-123", "new-uuid-456"))

    @patch("app.chat._get_connection")
    def test_get_user_conversations(self, mock_get_conn):
        from app import chat
        import datetime
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            ("conv-uuid-1", datetime.datetime(2026, 7, 17, 12, 0, 0), "Preview msg 1"),
            ("conv-uuid-2", None, None)
        ]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_get_conn.return_value = mock_conn

        res = chat.get_user_conversations("user-uuid-123")
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["id"], "conv-uuid-1")
        self.assertEqual(res[0]["preview"], "Preview msg 1")
        self.assertEqual(res[1]["preview"], "")

class TestAPIEndpointsNew(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("app.main.auth.get_user_id")
    @patch("app.main.auth.get_identifier")
    @patch("app.main.usage.check_and_increment_upload")
    @patch("app.main.extract_text_from_pdf")
    @patch("app.main.embed_batch")
    @patch("app.main.save_chunks")
    @patch("app.main.usage.check_and_add_document_to_conversation")
    def test_upload_endpoint_with_auth_and_limit(self, mock_add_doc, mock_save, mock_embed, mock_extract, mock_check_upload, mock_get_id, mock_get_uid):
        mock_get_uid.return_value = "user-uuid"
        mock_get_id.return_value = "user-uuid"
        mock_extract.return_value = "Extracted PDF contents."
        mock_embed.return_value = [[0.1, 0.2]]
        
        pdf_content = b"%PDF-1.4 mock pdf content"
        files = {"file": ("test.pdf", pdf_content, "application/pdf")}
        data = {"conversation_id": "conv-uuid-123"}
        headers = {"Authorization": "Bearer valid-jwt-token"}

        response = self.client.post("/upload", files=files, data=data, headers=headers)
        self.assertEqual(response.status_code, 200)
        mock_check_upload.assert_called_once_with("user-uuid", False)
        mock_add_doc.assert_called_once_with("conv-uuid-123", response.json()["doc_id"])

    @patch("app.main.auth.get_user_id")
    @patch("app.main.auth.get_identifier")
    @patch("app.main.usage.check_and_increment_upload")
    def test_upload_endpoint_rate_limited(self, mock_check_upload, mock_get_id, mock_get_uid):
        mock_get_uid.return_value = None
        mock_get_id.return_value = "guest-uuid"
        from fastapi import HTTPException
        mock_check_upload.side_effect = HTTPException(status_code=429, detail="Limit exceeded")

        files = {"file": ("test.pdf", b"pdf content", "application/pdf")}
        headers = {"X-Guest-Id": "guest-uuid"}
        response = self.client.post("/upload", files=files, headers=headers)
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["detail"], "Limit exceeded")

    @patch("app.main.auth.get_user_id")
    @patch("app.main.chat.chat")
    def test_chat_endpoint_quota_limit(self, mock_chat, mock_get_uid):
        mock_get_uid.return_value = "user-uuid"
        from fastapi import HTTPException
        mock_chat.side_effect = HTTPException(status_code=429, detail="Quota exceeded")

        payload = {"message": "Hello"}
        headers = {"Authorization": "Bearer valid-token"}
        response = self.client.post("/chat", json=payload, headers=headers)
        self.assertEqual(response.status_code, 429)

    @patch("app.main.doc_exists")
    @patch("app.main.generate_notes")
    def test_generate_endpoint_quota_limit(self, mock_generate, mock_doc_exists):
        mock_doc_exists.return_value = True
        from fastapi import HTTPException
        mock_generate.side_effect = HTTPException(status_code=429, detail="Quota exceeded")

        payload = {"doc_ids": ["doc-uuid"], "feature": "quiz", "instruction": "Make quiz"}
        response = self.client.post("/generate", json=payload)
        self.assertEqual(response.status_code, 429)

    def test_get_conversations_missing_auth(self):
        response = self.client.get("/conversations")
        self.assertEqual(response.status_code, 401)
        self.assertIn("Please log in to view chat history", response.json()["detail"])

    @patch("app.main.auth.get_user_id")
    def test_get_conversations_invalid_auth(self, mock_get_uid):
        from fastapi import HTTPException
        mock_get_uid.side_effect = HTTPException(status_code=401, detail="Expired token")

        headers = {"Authorization": "Bearer invalid-token"}
        response = self.client.get("/conversations", headers=headers)
        self.assertEqual(response.status_code, 401)
        # Note: endpoint overrides error detail
        self.assertEqual(response.json()["detail"], "Please log in to view chat history")

    @patch("app.main.auth.get_user_id")
    @patch("app.main.chat.get_user_conversations")
    def test_get_conversations_success(self, mock_get_convs, mock_get_uid):
        mock_get_uid.return_value = "user-uuid"
        mock_get_convs.return_value = [{"id": "c1", "created_at": "2026-07-17T12:00:00Z", "preview": "Hello"}]

        headers = {"Authorization": "Bearer token"}
        response = self.client.get("/conversations", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)
        self.assertEqual(response.json()[0]["id"], "c1")

if __name__ == "__main__":
    unittest.main()
